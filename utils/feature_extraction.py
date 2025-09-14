import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Analisi scientifica
from scipy import stats
from scipy.signal import welch, hilbert

FS = 20480  # Hz - Sampling rate
PULSES_PER_REV = 1  # Pulses per revolution
N_TEETH_INPUT = 40  # Denti ingranaggio input  
N_TEETH_OUTPUT = 72  # Denti ingranaggio output

def _safe_array(x):
    """Converte in array numpy sicuro gestendo vari formati."""
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(x, dtype=float).ravel()
    else:
        try:
            arr = np.asarray([x], dtype=float)
        except Exception:
            arr = np.asarray([], dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr

def time_feats(arr):
    """Estrae 14 features nel dominio del tempo."""
    if arr.size == 0:
        return {
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'median': np.nan, 'p10': np.nan, 'p90': np.nan, 'rms': np.nan,
            'skew': np.nan, 'kurt': np.nan, 'ptp': np.nan, 'iqr': np.nan,
            'zcr': np.nan, 'envelope_rms': np.nan
        }
    
    feats = {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
        'p90': float(np.percentile(arr, 90)),
        'rms': float(np.sqrt(np.mean(arr**2))),
        'skew': float(stats.skew(arr, bias=False)) if arr.size > 2 else 0.0,
        'kurt': float(stats.kurtosis(arr, fisher=True, bias=False)) if arr.size > 3 else 0.0,
        'ptp': float(np.ptp(arr)),
        'iqr': float(np.subtract(*np.percentile(arr, [75, 25]))),
        'zcr': float(np.mean(np.abs(np.diff(np.sign(arr))) > 0)) if arr.size > 1 else 0.0,
    }
    
    # Envelope RMS tramite trasformata di Hilbert
    try:
        env = np.abs(hilbert(arr))
        feats['envelope_rms'] = float(np.sqrt(np.mean(env**2)))
    except Exception:
        feats['envelope_rms'] = np.nan
    
    return feats

def psd_band_feats(arr, fs=FS, bands=None):
    """Estrae energia in 6 bande di frequenza specifiche."""
    if bands is None:
        # Bande ottimizzate per diagnosi ingranaggi
        bands = [(0,200), (200,500), (500,1000), (1000,2000), (2000,4000), (4000,8000)]
    
    if arr.size < 8:
        return {f'band_{lo}_{hi}': np.nan for (lo,hi) in bands}
    
    # Welch PSD
    try:
        f, Pxx = welch(arr, fs=fs, nperseg=min(2048, len(arr)))
    except Exception:
        return {f'band_{lo}_{hi}': np.nan for (lo,hi) in bands}
    
    feats = {}
    for (lo, hi) in bands:
        mask = (f >= lo) & (f < hi)
        feats[f'band_{lo}_{hi}'] = float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
    
    # Normalizza per energia totale
    total = float(np.trapz(Pxx, f)) if f.size else 1.0
    for (lo, hi) in bands:
        key = f'band_{lo}_{hi}'
        feats[key] = feats[key] / total if total > 0 else 0.0
    
    return feats

def tach_feats(arr, fs=FS, pulses_per_rev=PULSES_PER_REV, threshold=0.0):
    """Estrae 5 features dal segnale tachimetrico."""
    if arr.size == 0:
        return {'rpm': np.nan, 'pulse_count': 0, 'ipi_mean': np.nan, 
                'ipi_std': np.nan, 'ipi_cv': np.nan}
    
    # Binarizza segnale
    bin_sig = (arr > threshold).astype(np.uint8)
    idx = np.flatnonzero(bin_sig)
    pulse_count = int(idx.size)
    
    # Calcola RPM
    rpm = np.nan
    if fs is not None and arr.size > 0 and pulses_per_rev > 0:
        window_sec = arr.size / float(fs)
        if window_sec > 0:
            revs = pulse_count / float(pulses_per_rev)
            rpm = (revs / window_sec) * 60.0
    
    # Inter-pulse interval statistics
    if idx.size > 1:
        ipi = np.diff(idx) / float(fs) if fs else np.diff(idx).astype(float)
        ipi_mean = float(np.mean(ipi))
        ipi_std = float(np.std(ipi))
        ipi_cv = float(ipi_std / ipi_mean) if ipi_mean > 0 else np.nan
    else:
        ipi_mean = ipi_std = ipi_cv = np.nan
    
    return {
        'rpm': float(rpm), 
        'pulse_count': pulse_count, 
        'ipi_mean': ipi_mean, 
        'ipi_std': ipi_std, 
        'ipi_cv': ipi_cv
    }

def cross_axis_feats(ax, ay, az):
    """Calcola 3 correlazioni tra assi."""
    feats = {}
    
    def safe_corr(a, b):
        if len(a) < 2 or len(b) < 2:
            return np.nan
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = (np.std(a) * np.std(b))
        if denom == 0:
            return 0.0
        return float(np.mean(a*b) / denom)
    
    feats['corr_xy'] = safe_corr(ax, ay)
    feats['corr_xz'] = safe_corr(ax, az)
    feats['corr_yz'] = safe_corr(ay, az)
    
    return feats

def expand_features(df):
    """
    Espande il dataframe raw in 75 features ottimizzate.
    
    Struttura:
    - 60 features vibrazione (3 assi × 20 features)
    - 3 cross-correlazioni
    - 5 tachometer features
    - 2 scalari (velocita, torque)
    - 5 interazioni
    """
    
    print("🔧 Estrazione delle 75 features ottimizzate...")
    rows = []
    
    for i, row in df.iterrows():
        feat_row = {}
        
        # Features scalari
        feat_row['velocita'] = row['velocita']
        feat_row['torque'] = row['torque']
        
        # Estrai segnali
        ax = _safe_array(row['horizontal_acceleration'])
        ay = _safe_array(row['axial_acceleration'])
        az = _safe_array(row['vertical_acceleration'])
        tach = _safe_array(row['tachometer_signal'])
        
        # Features per asse (20 features × 3 assi = 60)
        for sig, name in [(ax,'ax'), (ay,'ay'), (az,'az')]:
            if sig.size > 0:
                # Time features (14)
                tf = time_feats(sig)
                for k, v in tf.items():
                    feat_row[f'{name}_{k}'] = v
                
                # Frequency features (6)
                pf = psd_band_feats(sig, fs=FS)
                for k, v in pf.items():
                    feat_row[f'{name}_{k}'] = v
        
        # Cross-axis features (3)
        if ax.size > 0 and ay.size > 0 and az.size > 0:
            cross_feats = cross_axis_feats(ax, ay, az)
            feat_row.update(cross_feats)
        
        # Tachometer features (5)
        if tach.size > 0:
            tach_features = tach_feats(tach, fs=FS, pulses_per_rev=PULSES_PER_REV)
            feat_row.update(tach_features)
        
        # Interaction features (5)
        v = feat_row['velocita']
        t = feat_row['torque']
        feat_row['v_times_t'] = v * t
        feat_row['v_sq'] = v**2
        feat_row['t_sq'] = t**2
        
        if 'rpm' in feat_row and np.isfinite(feat_row['rpm']):
            rpm = feat_row['rpm']
            feat_row['v_over_rpm'] = v / (rpm + 1e-6)
            feat_row['t_over_rpm'] = t / (rpm + 1e-6)
        else:
            feat_row['v_over_rpm'] = np.nan
            feat_row['t_over_rpm'] = np.nan
        
        rows.append(feat_row)
        
        if (i + 1) % 50 == 0:
            print(f"   Processati {i + 1}/{len(df)} samples...")
    
    X_full = pd.DataFrame(rows)
    
    # Gestione valori mancanti
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_full = X_full.fillna(X_full.median(numeric_only=True))
    
    print(f"   ✓ Estratte {len(X_full.columns)} features da {len(X_full)} samples")
    
    return X_full