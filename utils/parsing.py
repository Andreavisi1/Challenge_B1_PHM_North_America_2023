import re, os
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm

SR = 20480  # Hz

def _load_rows(files, build_meta):
    rows = []
    for p in tqdm(files, desc="Parsing dataset", unit="file"):
        try:
            data = np.loadtxt(p)
            meta = build_meta(p)
            if not meta: 
                tqdm.write(f"Nome file non riconosciuto: {p.name}")
                continue
            rows.append({
                **meta,
                'file_name': p.name,
                'horizontal_acceleration': data[:, 0],
                'axial_acceleration': data[:, 1],
                'vertical_acceleration': data[:, 2],
                'tachometer_signal': data[:, 3],
                'sampling_rate': SR,
                'duration': len(data)/SR,
                'num_samples': len(data),
            })
        except Exception as e:
            tqdm.write(f"Errore nel leggere il file {p}: {e}")
    return pd.DataFrame(rows)

def parse_vibration_dataset(dataset_path):
    files = list(Path(dataset_path).rglob('*.txt'))
    print(f"Trovati {len(files)} file .txt da processare")

    pat = re.compile(r'V(\d+)_(\d+)N_(\d+)\.txt')

    def build_meta(p: Path):
        m = pat.search(p.name)
        if not m: return None
        folder = p.parent.name
        if folder.startswith('Pitting_degradation_level_'):
            lab = folder.replace('Pitting_degradation_level_', '')
            etichetta, *rest = [s.strip() for s in re.split(r'[()]', lab) if s.strip()]
            descr = rest[0] if rest else None
        else:
            etichetta, descr = folder, None
        return {
            'etichetta': etichetta,
            'health_level': int(etichetta) if etichetta.isdigit() else etichetta,
            'velocita': int(m.group(1)),
            'torque':   int(m.group(2)),
            'rep':      int(m.group(3)),
            'descrizione': descr
        }

    df = _load_rows(files, build_meta)
    if not df.empty:
        print("\nOrdinamento dataset...")
        df = df.sort_values(['health_level','velocita','torque','rep']).reset_index(drop=True)
        print(f"Dataset caricato: {len(df)} file processati")
        print(f"Health levels disponibili: {sorted(df['health_level'].unique())}")
        print(f"Condizioni operative (rpm): {sorted(df['velocita'].unique())}")
        print(f"Condizioni operative (torque): {sorted(df['torque'].unique())}")
    return df

def parse_test_dataset(dataset_path):
    files = list(Path(dataset_path).rglob('*.txt'))
    print(f"Trovati {len(files)} file .txt da processare")

    pat = re.compile(r'^(\d+)_V(\d+)_(\d+)N(?:_(\d+))?\.txt$', re.IGNORECASE)

    def build_meta(p: Path):
        m = pat.search(p.name)
        if not m: return None
        return {
            'id':       int(m.group(1)),
            'velocita': int(m.group(2)),
            'torque':   int(m.group(3)),
        }

    df = _load_rows(files, build_meta)
    if not df.empty:
        print("\nOrdinamento dataset...")
        df = df.sort_values(['velocita','torque']).reset_index(drop=True)
        print(f"Dataset caricato: {len(df)} file processati")
        print(f"Condizioni operative (rpm): {sorted(df['velocita'].unique())}")
        print(f"Condizioni operative (torque): {sorted(df['torque'].unique())}")
    return df
