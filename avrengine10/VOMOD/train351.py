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
# 160 маг + 160 текстура + 160 фаза + 6 контекст/метрики
# ============================================================================

IN_NUM_MATRIX = 128  # Управляющий параметр количества полос
OUT_NUM_MATRIX = 196 
    
HIDDEN_SIZE = 512
NUM_LAYERS = 5
MODEL_PATH = "test372.vonn"

SEQ_LEN = 16 
MIN_READ_HZ = 2000.0

WINDOW_SIZE = 4096
HOP_SIZE = 1024
CUTOFF_MIN = 8000        
CUTOFF_MAX = 8000       
TARGET_MAX_FREQ = 16000 
EPOCHS = 800          
BATCH_SIZE = 64

CUTOFF_AUGMENTATIONS_PER_FILE = 1  

# ============================================================================
# НАСТРОЙКИ РЕЖИМА МИКШИРОВАНИЯ
# ============================================================================
MODE_FLATNESS_MULT = 1.15         
MODE_HFC_MULT = 0.85              
MODE_HIGH_CUTOFF_THRESH = 6000.0  
MODE_HIGH_CUTOFF_BOOST = 0.333    

AUDIO_DIRECTORY = "data2"
CHECKPOINT_DIR = "checkpoints372"
SAVE_EVERY_N_EPOCHS = 5  
RESUME_TRAINING = True
CACHE_PATH = os.path.join(CHECKPOINT_DIR, f"dataset_fast_seq{SEQ_LEN}.pt")

NUM_CONTEXT_PARAMS = 6 # не регулируется
NUM_BANDS = IN_NUM_MATRIX
NUM_MACRO_BANDS = OUT_NUM_MATRIX

# ВХОД: 4 побендовые группы (Mag, Texture, Phase_Coherence, Phase_GroupDelay) + Context
SINGLE_INPUT_SIZE = NUM_BANDS * 4 + NUM_CONTEXT_PARAMS       
# ВЫХОД: 5 побендовых групп (Gain, DCT_freq, DCT_time, Phase_Coherence, Phase_Rotation) + Mode
SINGLE_OUTPUT_SIZE = NUM_MACRO_BANDS * 5 + 1

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = channels // reduction
        self.fc1 = nn.Conv1d(channels, reduced_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(reduced_channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s

class ResConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(16, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, channels)
        self.se = SqueezeExcitation(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + residual)

class VOMOD30Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Conv1d(SINGLE_INPUT_SIZE, HIDDEN_SIZE, kernel_size=3, padding=1)
        self.act = nn.GELU()
        
        self.res_blocks = nn.ModuleList([
            ResConvBlock(HIDDEN_SIZE) for _ in range((NUM_LAYERS - 2) // 2)
        ])
        
        self.out_proj = nn.Conv1d(HIDDEN_SIZE, SINGLE_OUTPUT_SIZE, kernel_size=1)

    def forward(self, x):
        x = self.act(self.in_proj(x))
        for block in self.res_blocks:
            x = block(x)
        return self.out_proj(x)

def load_wav_scipy(filepath):
    sr, data = wavfile.read(filepath)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return sr, torch.FloatTensor(data)

def extract_dataset_from_audio(waveform, sr, cutoff_hz):
    hann_win = torch.hann_window(WINDOW_SIZE)
    stft = torch.stft(waveform, n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, window=hann_win, return_complex=True)
    mags = torch.abs(stft)
    
    bin_min_read = int(MIN_READ_HZ * WINDOW_SIZE / sr)
    bin_cutoff = int(cutoff_hz * WINDOW_SIZE / sr)
    
    if bin_cutoff <= bin_min_read + 4:
        bin_min_read = max(4, bin_cutoff - 8)

    half_window = WINDOW_SIZE // 2
    max_bin = min(int(TARGET_MAX_FREQ * WINDOW_SIZE / sr), half_window)
    
    donor_stft = stft[bin_min_read : bin_cutoff + 1, :]
    donor_mags = mags[bin_min_read : bin_cutoff + 1, :]
    donor_bins = donor_mags.shape[0]

    sum_mag = torch.sum(mags[1:bin_cutoff+1, :], dim=0)
    mask = sum_mag >= 1e-5
    
    if not torch.any(mask) or donor_bins < 4:
        return torch.empty(0, SINGLE_INPUT_SIZE, SEQ_LEN), torch.empty(0, SINGLE_OUTPUT_SIZE, SEQ_LEN)

    stft = stft[:, mask]
    mags = mags[:, mask]
    donor_stft = donor_stft[:, mask]
    donor_mags = donor_mags[:, mask]
    num_frames = mags.shape[1]

    # 1. Донорский интервал
    norm_pos = torch.linspace(0, 1, NUM_BANDS, device=mags.device)
    bin_pos = norm_pos * (donor_bins - 1)
    idx_low = torch.floor(bin_pos).long()
    idx_high = torch.clamp(idx_low + 1, max=donor_bins - 1)
    weight = (bin_pos - idx_low).unsqueeze(1)

    log_donor_mags = torch.log1p(donor_mags * 10.0)
    interp_mag = (1.0 - weight) * log_donor_mags[idx_low, :] + weight * log_donor_mags[idx_high, :]
    features_mag = torch.clamp(interp_mag / 5.0, 0.0, 2.0)

    # Текстура
    kernel_size = 9
    padding = kernel_size // 2
    donor_t = donor_mags.t().unsqueeze(1)
    kernel = torch.ones(1, 1, kernel_size, device=mags.device) / kernel_size
    smooth_donor = F.conv1d(F.pad(donor_t, (padding, padding), mode='replicate'), kernel).squeeze(1).t() + 1e-7
    interp_smooth = (1.0 - weight) * smooth_donor[idx_low, :] + weight * smooth_donor[idx_high, :]
    features_texture = torch.clamp((interp_mag + 1e-7) / (torch.log1p(interp_smooth * 10.0) + 1e-7), 0.0, 3.0) / 3.0

    # Резкость фазы
    phase_window = max(2, donor_bins // NUM_BANDS)
    pad_len = phase_window // 2
    padded_stft = F.pad(donor_stft.unsqueeze(0), (0, 0, pad_len, pad_len), mode='replicate').squeeze(0)
    complex_sum = torch.zeros_like(donor_stft)
    abs_sum = torch.zeros_like(donor_mags)
    for w in range(-pad_len, pad_len + 1):
        shifted_stft = padded_stft[pad_len + w : pad_len + w + donor_bins, :]
        complex_sum += shifted_stft
        abs_sum += torch.abs(shifted_stft)
    raw_phase = torch.abs(complex_sum) / (abs_sum + 1e-7)
    features_phase = (1.0 - weight) * raw_phase[idx_low, :] + weight * raw_phase[idx_high, :]

    # 2. НЧ-контекст и метрики
    lf_mags = mags[1:bin_min_read, :]
    lf_energy = torch.mean(lf_mags, dim=0, keepdim=True)
    lf_energy_feat = torch.clamp(torch.log1p(lf_energy * 10.0) / 5.0, 0.0, 2.0)

    log_lf_mean = torch.mean(torch.log(lf_mags + 1e-7), dim=0, keepdim=True)
    lf_mean_mag = torch.mean(lf_mags, dim=0, keepdim=True) + 1e-7
    lf_flatness = torch.clamp(torch.exp(log_lf_mean) / lf_mean_mag, 0.0, 1.0)

    donor_energy = torch.mean(donor_mags, dim=0, keepdim=True) + 1e-7
    tilt_ratio = torch.clamp((lf_energy / donor_energy) / 50.0, 0.0, 1.0)

    hf_diff = torch.zeros_like(donor_energy)
    hf_diff[:, 1:] = torch.clamp(donor_energy[:, 1:] - donor_energy[:, :-1], min=0.0)
    hf_onset = torch.clamp(torch.log1p(hf_diff * 20.0), 0.0, 2.0)

    log_donor_mean = torch.mean(torch.log(donor_mags + 1e-7), dim=0, keepdim=True)
    donor_flatness = torch.clamp(torch.exp(log_donor_mean) / donor_energy, 0.0, 1.0)
    cutoff_norm = torch.full_like(donor_flatness, cutoff_hz / (sr / 2.0))

    context_features = torch.cat([lf_energy_feat, lf_flatness, tilt_ratio, hf_onset, donor_flatness, cutoff_norm], dim=0)

    inputs = torch.cat([features_mag, features_texture, features_phase, context_features], dim=0).t()

    num_macro_bands = (SINGLE_OUTPUT_SIZE - 1) // 5
    targets = torch.zeros(SINGLE_OUTPUT_SIZE, num_frames, device=mags.device)

    hf_mags = mags[bin_cutoff + 1 : max_bin + 1, :]
    hf_stft = stft[bin_cutoff + 1 : max_bin + 1, :]
    hf_bins_count = hf_mags.shape[0]

    shaped_flatness = torch.pow(donor_flatness.squeeze(0), 1.4)
    mode_target = torch.clamp(shaped_flatness * MODE_FLATNESS_MULT, 0.0, 1.0)
    bins_per_band = hf_bins_count / float(num_macro_bands)

    donor_start = max(bin_min_read, int(bin_cutoff * 0.7))
    donor_high_energy = torch.mean(mags[donor_start:bin_cutoff + 1, :], dim=0) + 1e-6

    # ----------------------------------------------------------------------------
    # 1. Извлечение 2 признаков фазы для ВХОДА (Coherence + Group Delay)
    # ----------------------------------------------------------------------------
    # А. Фазовая когерентность (Phase Sharpness)
    phase_window = max(2, donor_bins // NUM_BANDS)
    pad_len = phase_window // 2
    padded_stft = F.pad(donor_stft.unsqueeze(0), (0, 0, pad_len, pad_len), mode='replicate').squeeze(0)
    complex_sum = torch.zeros_like(donor_stft)
    abs_sum = torch.zeros_like(donor_mags)
    for w in range(-pad_len, pad_len + 1):
        shifted_stft = padded_stft[pad_len + w : pad_len + w + donor_bins, :]
        complex_sum += shifted_stft
        abs_sum += torch.abs(shifted_stft)
    raw_phase_coh = torch.abs(complex_sum) / (abs_sum + 1e-7)
    features_phase_coh = (1.0 - weight) * raw_phase_coh[idx_low, :] + weight * raw_phase_coh[idx_high, :]

    # Б. Групповая задержка фазы по частоте (Phase Group Delay)
    donor_angles = torch.angle(donor_stft)
    gd_raw = torch.zeros_like(donor_angles)
    gd_raw[1:, :] = torch.remainder(donor_angles[1:, :] - donor_angles[:-1, :] + np.pi, 2 * np.pi) - np.pi
    gd_norm = gd_raw / np.pi  # Нормализация в range [-1.0, 1.0]
    features_phase_gd = (1.0 - weight) * gd_norm[idx_low, :] + weight * gd_norm[idx_high, :]

    # Объединение всех 4 входных групп
    inputs = torch.cat([features_mag, features_texture, features_phase_coh, features_phase_gd, context_features], dim=0).t()

    # ----------------------------------------------------------------------------
    # 2. Формирование 2 целевых параметров фазы для ВЫХОДА
    # ----------------------------------------------------------------------------
    target_hf_angles = torch.angle(hf_stft)
    bin_phase_rotation = torch.zeros_like(hf_mags)

    patch_start = max(bin_min_read, int(bin_cutoff * 0.7))
    patch_width = max(4, bin_cutoff - patch_start)

    # Расчет фазовой когерентности для ВЧ-диапазона
    hf_phase_window = max(2, hf_bins_count // num_macro_bands)
    hf_pad_len = hf_phase_window // 2
    padded_hf_stft = F.pad(hf_stft.unsqueeze(0), (0, 0, hf_pad_len, hf_pad_len), mode='replicate').squeeze(0)
    complex_sum_hf = torch.zeros_like(hf_stft)
    abs_sum_hf = torch.zeros_like(hf_mags)
    for w in range(-hf_pad_len, hf_pad_len + 1):
        shifted_stft = padded_hf_stft[hf_pad_len + w : hf_pad_len + w + hf_bins_count, :]
        complex_sum_hf += shifted_stft
        abs_sum_hf += torch.abs(shifted_stft)
    bin_phase_coherence = torch.abs(complex_sum_hf) / (abs_sum_hf + 1e-7)

    # Расчет идеального поворота фазы (target phase - vocoded donor phase)
    for k_idx in range(hf_bins_count):
        k = bin_cutoff + 1 + k_idx
        actual_offset = k_idx % patch_width
        src_k = patch_start + actual_offset
        delta_k = k - src_k

        phase_adv = 2.0 * np.pi * delta_k * HOP_SIZE / WINDOW_SIZE
        frame_indices = torch.arange(num_frames, device=mags.device).unsqueeze(0)
        time_shift = frame_indices * phase_adv

        donor_ang = torch.angle(donor_stft[src_k - bin_min_read, :])
        expected_phase = donor_ang + time_shift

        diff_angle = torch.remainder(target_hf_angles[k_idx, :] - expected_phase + np.pi, 2 * np.pi) - np.pi
        bin_phase_rotation[k_idx, :] = diff_angle / np.pi  # [-1.0, 1.0]

    for b in range(num_macro_bands):
        b_start = int(b * bins_per_band)
        b_end = int((b + 1) * bins_per_band)
        if b_end <= b_start:
            targets[b + num_macro_bands * 3, :] = 1.0
            targets[b + num_macro_bands * 4, :] = 0.0
            continue

        band_hf = hf_mags[b_start:b_end, :]
        mean_orig = torch.mean(band_hf, dim=0)

        # 1. Gain Target в Log-пространстве
        raw_ratio = mean_orig / donor_high_energy
        targets[b, :] = torch.log1p(torch.clamp(raw_ratio, 0.0, 200.0))

        if b_end - b_start > 1:
            K = b_end - b_start
            
            # 2. DCT Intra-Band Texture
            log_band = torch.log1p(band_hf * 10.0)
            log_mean = torch.mean(log_band, dim=0, keepdim=True)
            log_res = log_band - log_mean

            n = torch.arange(K, device=mags.device).unsqueeze(1)
            dct_basis = torch.cos(np.pi * (n + 0.5) / K)
            
            dct_coeff = torch.sum(log_res * dct_basis, dim=0) / np.sqrt(K)
            targets[b + num_macro_bands, :] = torch.clamp(dct_coeff, -2.0, 2.0)

            # 3. Временной профиль
            targets[b + num_macro_bands * 2, :] = torch.clamp(
                (mean_orig - torch.mean(mean_orig)) / (torch.mean(mean_orig) + 1e-5), -1.0, 1.0
            )
            
            # Параметр фазы 1: Локальная когерентность (Sharpness)
            targets[b + num_macro_bands * 3, :] = torch.mean(bin_phase_coherence[b_start:b_end, :], dim=0)

            # Параметр фазы 2: Поворот фазы (Phase Rotation via Vector Averaging)
            rot_rad = bin_phase_rotation[b_start:b_end, :] * np.pi
            mean_cos = torch.mean(torch.cos(rot_rad), dim=0)
            mean_sin = torch.mean(torch.sin(rot_rad), dim=0)
            targets[b + num_macro_bands * 4, :] = torch.atan2(mean_sin, mean_cos) / np.pi

    targets[-1, :] = mode_target

    inputs_single = inputs.t()
    targets_single = targets
    if num_frames < SEQ_LEN:
        return torch.empty(0, SINGLE_INPUT_SIZE, SEQ_LEN), torch.empty(0, SINGLE_OUTPUT_SIZE, SEQ_LEN)

    packed_inputs, packed_targets = [], []
    step = max(1, SEQ_LEN // 2)
    for i in range(0, num_frames - SEQ_LEN + 1, step):
        packed_inputs.append(inputs_single[:, i : i + SEQ_LEN])
        packed_targets.append(targets_single[:, i : i + SEQ_LEN])

    return torch.stack(packed_inputs, dim=0), torch.stack(packed_targets, dim=0)

def process_single_file(filepath):
    try:
        torch.set_num_threads(1)
        sr, waveform = load_wav_scipy(filepath)
        
        X_sub, Y_sub = [], []
        for _ in range(CUTOFF_AUGMENTATIONS_PER_FILE):
            cutoff_hz = random.uniform(CUTOFF_MIN, CUTOFF_MAX) if CUTOFF_MIN != CUTOFF_MAX else CUTOFF_MIN
            x, y = extract_dataset_from_audio(waveform, sr, cutoff_hz)
            if x.shape[0] > 0:
                X_sub.append(x)
                Y_sub.append(y)
                
        if X_sub:
            return torch.cat(X_sub, dim=0), torch.cat(Y_sub, dim=0)
        return None
    except Exception as e:
        print(f"[ОШИБКА] {filepath}: {e}")
        return None

def build_dataset(audio_dir):
    if os.path.exists(CACHE_PATH) and RESUME_TRAINING:
        print(f"[КЭШ] Загрузка подготовленного датасета из {CACHE_PATH}...")
        return torch.load(CACHE_PATH)

    print(f"[ДАТАСЕТ] Многопроцессорный сбор данных из {audio_dir}...")
    files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
    if not files:
        raise FileNotFoundError(f"В папке '{audio_dir}' нет .wav файлов!")

    X_all, Y_all = [], []
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_single_file, files)
        for res in results:
            if res is not None:
                x, y = res
                X_all.append(x)
                Y_all.append(y)

    X_tensor = torch.cat(X_all, dim=0)
    Y_tensor = torch.cat(Y_all, dim=0)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save((X_tensor, Y_tensor), CACHE_PATH)
    print(f"[ДАТАСЕТ] Успешно собрано {X_tensor.shape[0]} блоков данных.")
    return X_tensor, Y_tensor

def save_vonn_weights(model, filepath):
    model.eval()
    with torch.no_grad():
        with open(filepath, "wb") as f:
            magic = 0x4E4E4F56 
            version = 37 
            
            layers_to_save = []
            layers_to_save.append(model.in_proj)
            for block in model.res_blocks:
                layers_to_save.append(block.conv1)
                layers_to_save.append(block.gn1)
                layers_to_save.append(block.conv2)
                layers_to_save.append(block.gn2)
                layers_to_save.append(block.se.fc1)
                layers_to_save.append(block.se.fc2)
            layers_to_save.append(model.out_proj)

            f.write(struct.pack('II', magic, version))
            f.write(struct.pack('IIIIIII', 
                                SEQ_LEN, 
                                SINGLE_INPUT_SIZE, 
                                SINGLE_OUTPUT_SIZE, 
                                HIDDEN_SIZE, 
                                len(layers_to_save), 
                                TARGET_MAX_FREQ,
                                int(MIN_READ_HZ)))
            
            for layer in layers_to_save:
                w = layer.weight.detach().cpu().numpy().astype(np.float64) 
                b = layer.bias.detach().cpu().numpy().astype(np.float64)   
                f.write(w.tobytes())
                f.write(b.tobytes())
            
    print(f"[EXPORT] Conv1D + GN + SE Модель сохранена: {filepath}")

def compute_gain_loss(pred_gain, target_gain):
    diff = target_gain - pred_gain
    asym_weights = torch.where(diff > 0, 2.0, 1.0)
    base_loss = F.smooth_l1_loss(pred_gain, target_gain, reduction='none') * asym_weights

    target_dt = torch.zeros_like(target_gain)
    target_dt[:, :, 1:] = torch.clamp(target_gain[:, :, 1:] - target_gain[:, :, :-1], min=0.0)
    
    transient_weights = 1.0 + 4.0 * torch.tanh(target_dt * 2.0)
    weighted_base_loss = base_loss * transient_weights

    pred_dt = pred_gain[:, :, 1:] - pred_gain[:, :, :-1]
    target_dt_full = target_gain[:, :, 1:] - target_gain[:, :, :-1]
    loss_transient = F.l1_loss(pred_dt, target_dt_full)

    num_bands = pred_gain.shape[1]
    freq_weights = torch.linspace(1.0, 2.0, num_bands, device=pred_gain.device).view(1, num_bands, 1)

    final_gain_loss = torch.mean(weighted_base_loss * freq_weights) + 1.5 * loss_transient
    return final_gain_loss
    
def compute_psnr(pred_gain, target_gain, max_val=5.0):
    mse = F.mse_loss(pred_gain, target_gain)
    if mse < 1e-8:
        return torch.tensor(100.0, device=pred_gain.device)
    psnr = 20.0 * torch.log10(torch.tensor(max_val, device=pred_gain.device)) - 10.0 * torch.log10(mse)
    return psnr

def train():
    X, Y = build_dataset(AUDIO_DIRECTORY) 
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VOMOD30Net()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
    
    criterion_l1 = nn.L1Loss()
    
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "training_state.pt")
    start_epoch = 0
    
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_FILE):
        print(f"[INFO] Загрузка чекпоинта из {CHECKPOINT_FILE}...")
        checkpoint = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    num_macro_bands = (SINGLE_OUTPUT_SIZE - 1) // 5
    print(f"[INFO] Обучение с емкостью (HIDDEN_SIZE={HIDDEN_SIZE}) и Log-Gain Loss...")
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        batches = 0
        total_psnr = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)

            # Распаковка выходов нейросети
            pred_gain = predictions[:, 0:num_macro_bands, :]
            pred_cfreq = predictions[:, num_macro_bands:num_macro_bands*2, :]
            pred_ctime = predictions[:, num_macro_bands*2:num_macro_bands*3, :]
            pred_phase_coh = predictions[:, num_macro_bands*3:num_macro_bands*4, :]
            pred_phase_rot = predictions[:, num_macro_bands*4:num_macro_bands*5, :]
            pred_mode = predictions[:, -1, :]

            target_gain = batch_y[:, 0:num_macro_bands, :]
            target_cfreq = batch_y[:, num_macro_bands:num_macro_bands*2, :]
            target_ctime = batch_y[:, num_macro_bands*2:num_macro_bands*3, :]
            target_phase_coh = batch_y[:, num_macro_bands*3:num_macro_bands*4, :]
            target_phase_rot = batch_y[:, num_macro_bands*4:num_macro_bands*5, :]
            target_mode = batch_y[:, -1, :]

            loss_g = compute_gain_loss(pred_gain, target_gain)
            loss_cf = F.smooth_l1_loss(pred_cfreq, target_cfreq)
            loss_ct = criterion_l1(pred_ctime, target_ctime)

            loss_p_coh = criterion_l1(pred_phase_coh, target_phase_coh)

            # Комплексный векторный лосс фазы
            pred_rad = pred_phase_rot * np.pi
            target_rad = target_phase_rot * np.pi

            real_pred, imag_pred = torch.cos(pred_rad), torch.sin(pred_rad)
            real_target, imag_target = torch.cos(target_rad), torch.sin(target_rad)

            loss_p_rot = F.l1_loss(real_pred, real_target) + F.l1_loss(imag_pred, imag_target)

            # Временная производная фазы для сохранения жесткости транзиентов
            pred_rot_dt = pred_phase_rot[:, :, 1:] - pred_phase_rot[:, :, :-1]
            target_rot_dt = target_phase_rot[:, :, 1:] - target_phase_rot[:, :, :-1]
            loss_p_dt = F.l1_loss(pred_rot_dt, target_rot_dt)

            loss_m = criterion_l1(pred_mode, target_mode)

            # Итоговый лосс
            loss = (2.0 * loss_g + 
                    2.0 * loss_cf + 
                    1.0 * loss_ct + 
                    0.8 * loss_p_coh + 
                    1.5 * loss_p_rot + 
                    1.0 * loss_p_dt + 
                    1.0 * loss_m)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            with torch.no_grad():
                batch_psnr = compute_psnr(pred_gain, target_gain)
                total_psnr += batch_psnr.item()

            batches += 1
            
        avg_loss = epoch_loss / batches
        avg_psnr = total_psnr / batches
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Эпоха [{epoch+1}/{EPOCHS}] | Лосс: {avg_loss:.5f} | PSNR Upscale: {avg_psnr:.2f} dB | LR: {current_lr:.2e}")
        
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