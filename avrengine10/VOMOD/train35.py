# 
# Я не знаю ваши питоны, мне 80% аишка помогала
# потому что я хрен моржовый в ИИ и питоне, а вот математику я захуярил лопатой
# Если кто разбирается пишите в проект на Дискорд сервере
# Я тут совсем немного прошарил с обучением но если кто реально разбирается и у кого есть жесткие идеи
# то пмргите пж буду благодарен 
#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
import glob
import struct
import os
import random
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# 128 маг + 128 текстура + 128 фаза + 4 метрик
# ============================================================================
SINGLE_INPUT_SIZE = 388      
HIDDEN_SIZE = 256              # Размер одного слоя
NUM_LAYERS = 6                  # 7 слоев
SINGLE_OUTPUT_SIZE = 259        # 64 Gain + 64 Struct + 64 Coh + 64 Phase + 1 сдвиг + 1 ширина + 1 режим
MODEL_PATH = "conv_taka.vonn"

SEQ_LEN = 16 # 16, вроде нужно 32 но я не знаю, вроде при 16 все ресурсы используются но не уверен

WINDOW_SIZE = 4096
HOP_SIZE = 1024
CUTOFF_MIN = 10000        
CUTOFF_MAX = 15000       
TARGET_MAX_FREQ = 20000 
EPOCHS = 800      # при этой версии 3.5 нет смысла ставить выше 500 эпох     
BATCH_SIZE = 64

# =============================
# НАСТРОЙКИ РЕЖИМА МИКШИРОВАНИЯ
# Костыль какой-то :) - Нужно избавиться и автоматизировать
# по карте бинов, 25% логарифмического спада от 10кГц до 5кГц делая фазу менее хаотичной
# но модель будет вести себя как тварь поэтому хз пока

MODE_FLATNESS_MULT = 1.15         
MODE_HFC_MULT = 0.85              
MODE_HIGH_CUTOFF_THRESH = 6000.0  
MODE_HIGH_CUTOFF_BOOST = 0.333    

# =============================

AUDIO_DIRECTORY = "data"
CHECKPOINT_DIR = "checkpoints35_conv"
SAVE_EVERY_N_EPOCHS = 5  
RESUME_TRAINING = True
CACHE_PATH = os.path.join(CHECKPOINT_DIR, f"dataset_cache_seq{SEQ_LEN}.pt")

class VOMOD30Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Входной слой 
        self.layers.append(nn.Conv1d(SINGLE_INPUT_SIZE, HIDDEN_SIZE, kernel_size=3, padding=1))
        
        # Скрытые слои (kernel_size=3 для учета временного контекста)
        for _ in range(NUM_LAYERS - 2):
            self.layers.append(nn.Conv1d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding=1))
            
        # Выходной слой
        self.layers.append(nn.Conv1d(HIDDEN_SIZE, SINGLE_OUTPUT_SIZE, kernel_size=1))
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [B, SINGLE_INPUT_SIZE, SEQ_LEN]
        for layer in self.layers:
            x = self.relu(layer(x)) if layer != self.layers[-1] else layer(x)
        return x # output shape: [B, SINGLE_OUTPUT_SIZE, SEQ_LEN]

def load_wav_scipy(filepath):
    sr, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return sr, torch.FloatTensor(data)

# Найти мразь и устранить, или её уже нет?
def extract_dataset_from_audio(waveform, sr, cutoff_hz):
    hann_win = torch.hann_window(WINDOW_SIZE)
    stft = torch.stft(waveform, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, window=hann_win, return_complex=True)
    mags = torch.abs(stft)
    
    bin_cutoff = int(cutoff_hz * WINDOW_SIZE / sr)
    bin_cutoff = max(8, bin_cutoff)
    half_window = WINDOW_SIZE // 2

    max_bin = int(TARGET_MAX_FREQ * WINDOW_SIZE / sr)
    max_bin = min(max_bin, half_window)
    
    if bin_cutoff >= half_window - 4:
        return torch.empty(0, SINGLE_INPUT_SIZE, SEQ_LEN), torch.empty(0, SINGLE_OUTPUT_SIZE, SEQ_LEN)
        
    live_mags = mags[1:bin_cutoff+1, :] 
    sum_mag = torch.sum(live_mags, dim=0) 
    mask = sum_mag >= 1e-5
    
    if not torch.any(mask):
        return torch.empty(0, SINGLE_INPUT_SIZE, SEQ_LEN), torch.empty(0, SINGLE_OUTPUT_SIZE, SEQ_LEN)
        
    mags = mags[:, mask]
    live_mags = live_mags[:, mask]
    sum_mag = sum_mag[mask]
    
    mean_mag_lf = sum_mag / bin_cutoff
    num_frames = mags.shape[1]

    kernel_size = 9
    padding = kernel_size // 2
    live_t = live_mags.t().unsqueeze(1)
    kernel = torch.ones(1, 1, kernel_size, device=mags.device) / kernel_size
    padded_live = F.pad(live_t, (padding, padding), mode='replicate')
    smooth_live_t = F.conv1d(padded_live, kernel)
    smooth_live = smooth_live_t.squeeze(1).t()
    smooth_live = torch.clamp(smooth_live, min=1e-7)

    num_bands = (SINGLE_INPUT_SIZE - 4) // 3

    norm_pos = torch.linspace(0, 1, num_bands)
    bin_pos = norm_pos * (bin_cutoff - 1)
    idx_low = torch.floor(bin_pos).long()
    idx_high = torch.clamp(idx_low + 1, max=bin_cutoff - 1)
    weight = (bin_pos - idx_low).unsqueeze(1)

    interp_mag = (1.0 - weight) * live_mags[idx_low, :] + weight * live_mags[idx_high, :]
    features_mag = interp_mag / mean_mag_lf.unsqueeze(0)
    
    interp_smooth = (1.0 - weight) * smooth_live[idx_low, :] + weight * smooth_live[idx_high, :]
    features_texture = (interp_mag + 1e-7) / (interp_smooth + 1e-7)
    features_texture = torch.clamp(features_texture, 0.0, 5.0) / 5.0
    
    angles = torch.angle(stft)
    angles = angles[:, mask]
    live_angles = angles[1:bin_cutoff+1, :]
    padded_angles = F.pad(live_angles.unsqueeze(0), (0,0,1,0), mode='replicate').squeeze(0)
    phase_diff = live_angles - padded_angles[:-1, :]
    phase_diff = (phase_diff + torch.pi) % (2 * torch.pi) - torch.pi 
    phase_diff_norm = phase_diff / torch.pi
    features_phase = (1.0 - weight) * phase_diff_norm[idx_low, :] + weight * phase_diff_norm[idx_high, :]
    
    metrics = torch.zeros(4, num_frames)
    k_indices = torch.arange(1, bin_cutoff + 1).unsqueeze(1).float()
    centroid = torch.sum(k_indices * live_mags, dim=0) / (sum_mag + 1e-7)
    metrics[0, :] = centroid / bin_cutoff
    
    hfc_start = int(bin_cutoff * 0.85)
    hfc_energy = torch.sum(live_mags[hfc_start:, :], dim=0)
    metrics[1, :] = hfc_energy / (sum_mag + 1e-7)
    
    log_mean = torch.mean(torch.log(live_mags + 1e-7), dim=0)
    flatness = torch.exp(log_mean) / (mean_mag_lf + 1e-7)
    metrics[2, :] = flatness
    metrics[3, :] = cutoff_hz / (sr / 2.0) 
    
    inputs = torch.cat([features_mag, features_texture, features_phase, metrics], dim=0).t()
    
    num_macro_bands = (SINGLE_OUTPUT_SIZE - 3) // 4
    targets = torch.zeros(SINGLE_OUTPUT_SIZE, num_frames)
    # костыли костыли
    donor_ratio = 0.65 if cutoff_hz < 6000.0 else 0.5
    patch_start = int(bin_cutoff * donor_ratio)
    base_patch_width = bin_cutoff - patch_start
    if base_patch_width < 4: base_patch_width = 4
        
    hf_mags = mags[bin_cutoff + 1 : max_bin + 1, :]
    hf_bins_count = hf_mags.shape[0]
    
    width_mults = [0.6, 0.8, 1.0, 1.2, 1.4]
    best_width_idx = torch.zeros(num_frames, dtype=torch.long)
    best_width_corr = torch.full((num_frames,), -1.0)
    hf_norms_w = torch.norm(hf_mags, dim=0) + 1e-9
    
    for w_idx, w_mult in enumerate(width_mults):
        test_width = max(4, int(base_patch_width * w_mult))
        test_donor = torch.zeros_like(hf_mags)
        for k in range(hf_bins_count):
            src_k = patch_start + (k % test_width) + 1
            if src_k < mags.shape[0]:
                test_donor[k, :] = mags[src_k, :]
        
        corr = torch.sum(test_donor * hf_mags, dim=0) / (torch.norm(test_donor, dim=0) * hf_norms_w + 1e-9)
        better_mask = corr > best_width_corr
        best_width_corr[better_mask] = corr[better_mask]
        best_width_idx[better_mask] = w_idx
        
    targets[-3, :] = best_width_idx.float() / (len(width_mults) - 1.0)
    
    test_widths_tensor = torch.tensor([max(4, int(base_patch_width * m)) for m in width_mults], device=mags.device)
    frame_widths = test_widths_tensor[best_width_idx]
    
    max_shift = max(1, base_patch_width // 4)
    best_shifts = torch.zeros(num_frames)
    L = min(hf_bins_count, bin_cutoff - (patch_start + max_shift))
    
    if L >= 4 and max_shift > 0:
        hf_patches = hf_mags[:L, :]
        hf_norms = torch.norm(hf_patches, dim=0) + 1e-9
        best_corrs = torch.full((num_frames,), -1.0)
        best_s_vals = torch.zeros(num_frames)
        
        for s in range(-max_shift, max_shift + 1):
            curr_start = patch_start + s
            donor_patches = mags[curr_start : curr_start + L, :]
            corr = torch.sum(donor_patches * hf_patches, dim=0) / (torch.norm(donor_patches, dim=0) * hf_norms + 1e-9)
            better_mask = corr > best_corrs
            best_corrs[better_mask] = corr[better_mask]
            best_s_vals[better_mask] = s
            
        best_shifts = (best_s_vals / max_shift + 1.0) / 2.0
    else:
        best_shifts = torch.full((num_frames,), 0.5)
        
    targets[-3, :] = best_width_idx.float() / (len(width_mults) - 1.0)
    targets[-2, :] = best_shifts
    
    flatness_metric = metrics[2, :]
    hfc_metric = metrics[1, :]
    
    shaped_flatness = torch.pow(torch.clamp(flatness_metric, 0.0, 1.0), 1.4)
    mode_target = torch.clamp((shaped_flatness * MODE_FLATNESS_MULT) + (hfc_metric * MODE_HFC_MULT), 0.0, 1.0)
    
    if cutoff_hz > MODE_HIGH_CUTOFF_THRESH:
        mode_target = torch.clamp(mode_target * MODE_HIGH_CUTOFF_BOOST, 0.0, 1.0)
        
    targets[-1, :] = mode_target
    
    donor_env_t = F.conv1d(mags.t().unsqueeze(1), torch.ones(1, 1, 15, device=mags.device)/15, padding=7)
    donor_env = torch.clamp(donor_env_t.squeeze(1).t(), min=1e-7)
    
    donor_mags = torch.zeros_like(hf_mags)
    for k in range(hf_bins_count):
        src_k = patch_start + (k % frame_widths) + 1
        src_k = torch.clamp(src_k, max=mags.shape[0]-1)
        donor_mags[k, :] = (mags[src_k, torch.arange(num_frames)] / donor_env[src_k, torch.arange(num_frames)]) * mean_mag_lf

    bins_per_band = hf_bins_count / float(num_macro_bands)
    
    for b in range(num_macro_bands):
        b_start = int(b * bins_per_band)
        b_end = int((b + 1) * bins_per_band)
        if b_end <= b_start:
            targets[b, :] = 0.0
            targets[b + num_macro_bands, :] = 1.0
            targets[b + num_macro_bands * 2, :] = 0.5
            targets[b + num_macro_bands * 3, :] = 1.0
            continue

        band_hf = hf_mags[b_start:b_end, :]
        band_donor = donor_mags[b_start:b_end, :]
        
        mean_orig = torch.mean(band_hf, dim=0) 
        mean_donor = torch.mean(band_donor, dim=0)
        
        targets[b, :] = torch.clamp(mean_orig / (mean_mag_lf + 1e-7), 0.0, 100.0)
        
        if b_end - b_start > 1:
            dev_orig = band_hf - mean_orig.unsqueeze(0)
            dev_src = band_donor - mean_donor.unsqueeze(0)
            norm_orig = torch.norm(dev_orig, dim=0)
            norm_src = torch.norm(dev_src, dim=0)
            valid_mask = (norm_orig > 1e-6) & (norm_src > 1e-6)
            
            corr = torch.sum(dev_orig * dev_src, dim=0) / (norm_orig * norm_src + 1e-9)
            struct_target = torch.clamp((corr + 1.0) / 2.0, 0.0, 1.0)
            
            band_max, _ = torch.max(band_hf, dim=0)
            local_crest = band_max / (mean_orig + 1e-9)
            
            coh_target = torch.clamp((local_crest - 1.1) / 2.2, 0.0, 1.0)
            
            quiet_mask = mean_orig < (mean_mag_lf * 0.005 + 1e-7)
            coh_target = torch.where(quiet_mask, torch.zeros_like(coh_target), coh_target)
            struct_target = torch.where(quiet_mask, torch.zeros_like(struct_target), struct_target)
            
            targets[b + num_macro_bands, :] = torch.where(valid_mask, struct_target, torch.zeros_like(struct_target))
            targets[b + num_macro_bands * 2, :] = coh_target
            
            band_energy = torch.mean(band_hf, dim=0)
            padded_en = F.pad(band_energy.unsqueeze(0), (1,0), mode='replicate').squeeze(0)
            flux = torch.relu(band_energy - padded_en[:-1])
            attack_ratio = flux / (band_energy + 1e-7)
            
            phase_t = targets[b + num_macro_bands, :] * (1.0 - torch.clamp(attack_ratio * 1.5, 0.0, 1.0))
            targets[b + num_macro_bands * 3, :] = phase_t
                
        else:
            targets[b + num_macro_bands, :] = 0.5
            targets[b + num_macro_bands * 2, :] = 0.5
            
    # Переводим в формат [Channels, Time]
    inputs_single = inputs.t()  # [388, num_frames]
    targets_single = targets    # [259, num_frames]

    num_frames = inputs_single.shape[1]
    if num_frames < SEQ_LEN:
        return torch.empty(0, SINGLE_INPUT_SIZE, SEQ_LEN), torch.empty(0, SINGLE_OUTPUT_SIZE, SEQ_LEN)

    packed_inputs = []
    packed_targets = []

    # Шагаем с перекрытием (SEQ_LEN // 2)
    step = max(1, SEQ_LEN // 2)
    for i in range(0, num_frames - SEQ_LEN + 1, step):
        inp_seq = inputs_single[:, i : i + SEQ_LEN]  # [388, SEQ_LEN]
        tar_seq = targets_single[:, i : i + SEQ_LEN]  # [259, SEQ_LEN]
        packed_inputs.append(inp_seq)
        packed_targets.append(tar_seq)

    inputs_packed = torch.stack(packed_inputs, dim=0)   # [N, 388, SEQ_LEN]
    targets_packed = torch.stack(packed_targets, dim=0) # [N, 259, SEQ_LEN]

    return inputs_packed, targets_packed

def process_single_file(filepath):
    try:
        torch.set_num_threads(1)
        sr, waveform = load_wav_scipy(filepath)
        cutoff_hz = random.uniform(CUTOFF_MIN, CUTOFF_MAX)
        x, y = extract_dataset_from_audio(waveform, sr, cutoff_hz)
        return x, y
    except Exception as e:
        print(f"[ОШИБКА] {filepath}: {e}")
        return None

def build_dataset(audio_dir):
    if os.path.exists(CACHE_PATH) and RESUME_TRAINING:
        print(f"[КЭШ] Загрузка датасета из {CACHE_PATH}...")
        return torch.load(CACHE_PATH)

    print(f"[ДАТАСЕТ] Сбор данных из {audio_dir}...")
    files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
    if not files:
        raise FileNotFoundError(f"В папке '{audio_dir}' нет .wav файлов!")

    X_all, Y_all = [], []
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_single_file, files)
        for res in results:
            if res is not None:
                x, y = res
                if x.shape[0] > 0:
                    X_all.append(x)
                    Y_all.append(y)

    X_tensor = torch.cat(X_all, dim=0)
    Y_tensor = torch.cat(Y_all, dim=0)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save((X_tensor, Y_tensor), CACHE_PATH)
    print(f"[ДАТАСЕТ] Собрано {X_tensor.shape[0]} блоков по {SEQ_LEN} кадров.")
    return X_tensor, Y_tensor

def save_vonn_weights(model, filepath):
    model.eval()
    with torch.no_grad():
        with open(filepath, "wb") as f:
            magic = 0x4E4E4F56 # тут VONN написано
            version = 35
            
            f.write(struct.pack('II', magic, version))
            f.write(struct.pack('IIIIII', 
                                SEQ_LEN, 
                                SINGLE_INPUT_SIZE, 
                                SINGLE_OUTPUT_SIZE, 
                                HIDDEN_SIZE, 
                                NUM_LAYERS,
                                TARGET_MAX_FREQ))
            
            for layer in model.layers:
                w = layer.weight.detach().cpu().numpy().astype(np.float64) 
                b = layer.bias.detach().cpu().numpy().astype(np.float64)   
                f.write(w.tobytes())
                f.write(b.tobytes())
            
    print(f"[ЭКСПОРТ] Модель сохранена: {filepath}")

def asymmetric_gain_loss(pred, target):
    diff = target - pred
    squared_diff = torch.where(diff > 0, (diff ** 2) * 1.15, diff ** 2)
    return torch.mean(squared_diff)

def train():
    X, Y = build_dataset(AUDIO_DIRECTORY) 
    dataset_size = X.shape[0]

    model = VOMOD30Net()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    # малоэффективен сейчас, нужно контрастировать или менять технику обучения и сбора датасета
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    criterion_l1 = nn.L1Loss()
    criterion_shift = nn.MSELoss()
    
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "training_state.pt")
    start_epoch = 0
    
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_FILE):
        print(f" Загрузка чекпоинта из {CHECKPOINT_FILE}...")
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 1e-5:
                print(f"[ВНИМАНИЕ] Обнаружен мертвый LR ({param_group['lr']}). Сброс до 1e-4.")
                param_group['lr'] = 1e-4
        scheduler._reset()

    num_macro_bands = (SINGLE_OUTPUT_SIZE - 3) // 4
    print(f"[СТАРТ] Обучение Conv1D (SEQ_LEN = {SEQ_LEN})...")
    
    for epoch in range(start_epoch, EPOCHS):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batches = 0
        
        model.train()
        for i in range(0, dataset_size, BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X[indices], Y[indices]
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            
            # Разворачиваем оси: [B, C, SEQ_LEN] -> [B, SEQ_LEN, C] -> [B*SEQ_LEN, C]
            predictions_flat = predictions.transpose(1, 2).reshape(-1, SINGLE_OUTPUT_SIZE)
            targets_flat = batch_y.transpose(1, 2).reshape(-1, SINGLE_OUTPUT_SIZE)
            
            pred_gain = predictions_flat[:, 0:num_macro_bands]
            target_gain = targets_flat[:, 0:num_macro_bands]
            
            pred_struct = predictions_flat[:, num_macro_bands:num_macro_bands*2]
            target_struct = targets_flat[:, num_macro_bands:num_macro_bands*2]
            
            pred_coh = predictions_flat[:, num_macro_bands*2:num_macro_bands*3]
            target_coh = targets_flat[:, num_macro_bands*2:num_macro_bands*3]
            
            pred_phase = predictions_flat[:, num_macro_bands*3:num_macro_bands*4]
            target_phase = targets_flat[:, num_macro_bands*3:num_macro_bands*4]
            
            pred_width = predictions_flat[:, -3]
            target_width = targets_flat[:, -3]
            pred_shift = predictions_flat[:, -2]
            target_shift = targets_flat[:, -2]
            pred_mode = predictions_flat[:, -1]
            target_mode = targets_flat[:, -1]
            
            loss_g = asymmetric_gain_loss(pred_gain, target_gain)
            loss_s = criterion_l1(pred_struct, target_struct)
            loss_c = criterion_l1(pred_coh, target_coh)
            loss_p = criterion_l1(pred_phase, target_phase)
            
            loss_width = criterion_shift(pred_width, target_width) 
            loss_shift = criterion_shift(pred_shift, target_shift)
            
            loss_mode = criterion_l1(pred_mode, target_mode)
            
            loss = loss_g + 1.2 * loss_s + 1.0 * loss_c + 0.8 * loss_p + 0.5 * loss_width + loss_shift + 1.2 * loss_mode
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Эпоха [{epoch+1}/{EPOCHS}] | Лосс: {avg_loss:.5f} | LR: {current_lr:.2e}")
        
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }, CHECKPOINT_FILE)
            save_vonn_weights(model, MODEL_PATH)

    save_vonn_weights(model, MODEL_PATH)

if __name__ == "__main__":
    train()