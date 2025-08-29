import pandas as pd
import numpy as np
from tqdm import tqdm

def downsample(signal, fs=20480, sec=3):
    """
    Filtra e fa il resample di un segnale ad una lunghezza fissa.
    """
    if len(signal) == fs*sec:
        return signal
    return signal[:fs*sec]  # Truncate or pad to the desired length

def downsample_vibration_dataframe(df, sec=3):
    """
    Applica filtraggio e resample a tutti i segnali accelerometrici e tachimetro nel DataFrame.
    Ritorna un nuovo DataFrame con colonne preprocessate.
    """
    processed_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        try:
            record = {
                'file_name': row['file_name'],
                'etichetta': row['etichetta'],
                'health_level': row['health_level'],
                'velocita': row['velocita'],
                'torque': row['torque'],
                'rep': row['rep'],
                'sampling_rate': row['sampling_rate'],
                'descrizione': row['descrizione'],
                'duration': row['duration'],
                'num_samples': row['num_samples']
            }

            # Applica filtro + resample ai 4 segnali
            record['horizontal_acceleration'] = downsample(
                row['horizontal_acceleration'], fs=row['sampling_rate'], sec=sec
            )
            record['axial_acceleration'] = downsample(
                row['axial_acceleration'], fs=row['sampling_rate'], sec=sec
            )
            record['vertical_acceleration'] = downsample(
                row['vertical_acceleration'], fs=row['sampling_rate'], sec=sec
            )
            record['tachometer_signal'] = downsample(
                row['tachometer_signal'], fs=row['sampling_rate'], sec=sec
            )

            processed_records.append(record)

        except Exception as e:
            tqdm.write(f"Errore durante il preprocessing del file {row['file_name']}: {e}")
            continue

    return pd.DataFrame(processed_records)
