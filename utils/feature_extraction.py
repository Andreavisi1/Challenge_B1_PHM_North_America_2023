import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch, hilbert

# Parametri segnale
FS = 20480  # Hz (dal task PHM)
PULSES_PER_REV = 1
TACH_COL_NAME = 'tachometer_signal'

# -----------------------
# 3) Utils
# -----------------------
def _safe_array(x):
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
    try:
        env = np.abs(hilbert(arr))
        feats['envelope_rms'] = float(np.sqrt(np.mean(env**2)))
    except Exception:
        feats['envelope_rms'] = np.nan
    return feats

def tach_feats(arr, fs=FS, pulses_per_rev=PULSES_PER_REV, threshold=0.0):
    if arr.size == 0:
        return {'rpm': np.nan, 'pulse_count': 0, 'ipi_mean': np.nan, 'ipi_std': np.nan, 'ipi_cv': np.nan}
    bin_sig = (arr > threshold).astype(np.uint8)
    idx = np.flatnonzero(bin_sig)
    pulse_count = int(idx.size)
    rpm = np.nan
    if fs is not None and arr.size > 0 and pulses_per_rev > 0:
        window_sec = arr.size / float(fs)
        if window_sec > 0:
            revs = pulse_count / float(pulses_per_rev)
            rpm = (revs / window_sec) * 60.0
    if idx.size > 1:
        ipi = np.diff(idx) / float(fs) if fs else np.diff(idx).astype(float)
        ipi_mean = float(np.mean(ipi))
        ipi_std  = float(np.std(ipi))
        ipi_cv   = float(ipi_std / ipi_mean) if ipi_mean > 0 else np.nan
    else:
        ipi_mean = ipi_std = ipi_cv = np.nan
    return {'rpm': float(rpm), 'pulse_count': pulse_count, 'ipi_mean': ipi_mean, 'ipi_std': ipi_std, 'ipi_cv': ipi_cv}

def psd_band_feats(arr, fs=FS, bands=None):
    if bands is None:
        # bande fino alla Nyquist (10240 Hz), regolabili
        bands = [(0,200),(200,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)]
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
    # normalizza per energia totale per robustezza
    total = float(np.trapz(Pxx, f)) if f.size else 1.0
    for (lo, hi) in bands:
        key = f'band_{lo}_{hi}'
        feats[key] = feats[key] / total if total > 0 else 0.0
    return feats

def cross_axis_feats(ax, ay, az):
    feats = {}
    def safe_corr(a,b):
        if len(a) < 2 or len(b) < 2:
            return np.nan
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = (np.std(a)*np.std(b))
        if denom == 0:
            return 0.0
        return float(np.mean(a*b)/denom)
    feats['corr_xy'] = safe_corr(ax, ay)
    feats['corr_xz'] = safe_corr(ax, az)
    feats['corr_yz'] = safe_corr(ay, az)
    return feats

# -----------------------
# 4) Espansione feature
# -----------------------
def expand_features(df, array_cols, scalar_cols):
    rows = []
    for i, row in df.iterrows():
        feat_row = {}
        # scalari
        for c in scalar_cols:
            feat_row[c] = row[c]
        # array
        ax = _safe_array(row[array_cols[0]]) if 'horizontal_acceleration' in array_cols else np.array([])
        ay = _safe_array(row[array_cols[1]]) if 'axial_acceleration' in array_cols else np.array([])
        az = _safe_array(row[array_cols[2]]) if 'vertical_acceleration' in array_cols else np.array([])
        tach = _safe_array(row[TACH_COL_NAME]) if TACH_COL_NAME in array_cols else np.array([])

        # time feats per-asse
        for sig, name in [(ax,'ax'), (ay,'ay'), (az,'az')]:
            if sig.size > 0:
                tf = time_feats(sig)
                tf = {f'{name}_{k}': v for k,v in tf.items()}
                feat_row.update(tf)

                pf = psd_band_feats(sig, fs=FS)
                pf = {f'{name}_{k}': v for k,v in pf.items()}
                feat_row.update(pf)

        # cross-axis
        if ax.size > 0 and ay.size > 0 and az.size > 0:
            feat_row.update(cross_axis_feats(ax, ay, az))

        # tach
        if tach.size > 0:
            tfeat = tach_feats(tach, fs=FS, pulses_per_rev=PULSES_PER_REV, threshold=0.0)
            feat_row.update(tfeat)

        # interazioni operative
        if 'velocita' in feat_row and 'torque' in feat_row:
            v = feat_row['velocita']
            t = feat_row['torque']
            feat_row['v_times_t'] = v * t
            feat_row['v_sq'] = v**2
            feat_row['t_sq'] = t**2
            if 'rpm' in feat_row and np.isfinite(feat_row['rpm']):
                rpm = feat_row['rpm']
                # normalizzazioni semplici
                feat_row['v_over_rpm'] = v / (rpm + 1e-6)
                feat_row['t_over_rpm'] = t / (rpm + 1e-6)

        rows.append(feat_row)

    X_full = pd.DataFrame(rows)
    # imputazione semplice
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_full = X_full.fillna(X_full.median(numeric_only=True))
    return X_full

def remove_multicollinearity(X, threshold=0.95):
    """
    Rimuove features altamente correlate mantenendo quelle più informative.
    """
    
    print(f"\n📊 Rimozione multi-collinearità (threshold={threshold})...")
    
    # Calcola matrice correlazione
    corr_matrix = X.corr().abs()
    
    # Triangolare superiore
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Trova features da rimuovere
    to_drop = []
    for column in upper_tri.columns:
        if column in to_drop:
            continue
        
        # Features correlate sopra threshold
        correlated = list(upper_tri.index[upper_tri[column] > threshold])
        
        if correlated:
            # Mantieni quella con maggior varianza
            variances = X[[column] + correlated].var()
            keep = variances.idxmax()
            drop = [col for col in [column] + correlated if col != keep]
            to_drop.extend(drop)
    
    to_drop = list(set(to_drop))
    X_reduced = X.drop(columns=to_drop)
    
    print(f"   • Features iniziali: {len(X.columns)}")
    print(f"   • Features rimosse: {len(to_drop)}")
    print(f"   • Features rimanenti: {len(X_reduced.columns)}")
    
    return X_reduced, to_drop