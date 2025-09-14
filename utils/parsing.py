import numpy as np
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings('ignore')

def parse_phm_dataset_comprehensive(dataset_path, n_revolutions=10, max_samples_per_condition=5):
    """
    Parser avanzato per dataset PHM 2023
    
    Strategia:
    1. Estrae metadati completi (health level, RPM, torque, repetition)
    2. Usa tachimetro per segmentazione precisa delle rivoluzioni
    3. Estrae features fisicamente motivate
    4. Crea UN SOLO RECORD per file
    
    Args:
        dataset_path: Percorso al dataset
        n_revolutions: Numero di rivoluzioni da estrarre per consistency
        max_samples_per_condition: Limite samples per condizione
    
    Returns:
        DataFrame con 1 riga per file e features estratte
    """
    
    print(f"🔍 Iniziando parsing di: {dataset_path}")
    print(f"📐 Target: {n_revolutions} rivoluzioni per sample")
    print(f"🎯 APPROCCIO: 1 FILE = 1 RIGA con features complete")
    
    data_records = []
    parsing_stats = {
        'total_files': 0,
        'successful_parses': 0,
        'failed_parses': 0,
        'insufficient_revolutions': 0,
        'conditions_found': set()
    }
    
    # Scansione ricorsiva del dataset
    for root, dirs, files in os.walk(dataset_path):
        txt_files = [f for f in files if f.endswith('.txt')]
        
        if not txt_files:
            continue
            
        # Health level dalla cartella
        folder_name = os.path.basename(root)
        health_level = extract_health_level(folder_name)
        
        print(f"📁 Processando cartella: {folder_name} (Health Level: {health_level})")
        
        for file in txt_files:
            parsing_stats['total_files'] += 1
            
            try:
                # Parsing metadati dal filename
                metadata = parse_filename_metadata(file)
                if metadata is None:
                    print(f"❌ File non parsato: {file}")  # <-- aggiungi questa riga
                    parsing_stats['failed_parses'] += 1
                    continue

                
                # Aggiorna statistiche condizioni
                condition_key = (metadata['rpm'], metadata['torque'])
                parsing_stats['conditions_found'].add(condition_key)
                
                # Caricamento dati raw
                file_path = os.path.join(root, file)
                raw_data = load_vibration_data(file_path)
                
                if raw_data is None:
                    parsing_stats['failed_parses'] += 1
                    continue
                
                # Crea record con array raw
                record = {
                    'file_id': file.replace('.txt', ''),
                    'health_level': health_level,
                    'velocita': metadata['rpm'],
                    'torque': metadata['torque'],
                    'repetition': metadata['repetition'],
                    'horizontal_acceleration': raw_data[:, 0],
                    'axial_acceleration': raw_data[:, 1],
                    'vertical_acceleration': raw_data[:, 2],
                    'tachometer_signal': raw_data[:, 3]
                }
                
                data_records.append(record)
                parsing_stats['successful_parses'] += 1
                
                # Progress feedback ogni 50 files
                if parsing_stats['total_files'] % 50 == 0:
                    success_rate = parsing_stats['successful_parses'] / parsing_stats['total_files'] * 100
                    print(f"   ✓ {parsing_stats['total_files']} files processati (Success: {success_rate:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Errore con {file}: {str(e)[:100]}...")
                parsing_stats['failed_parses'] += 1
                continue
    
    # Report finale parsing
    print(f"\n📈 PARSING COMPLETATO!")
    print(f"   • Files totali: {parsing_stats['total_files']}")
    print(f"   • Parsing riusciti: {parsing_stats['successful_parses']}")
    print(f"   • Parsing falliti: {parsing_stats['failed_parses']}")
    print(f"   • Condizioni operative trovate: {len(parsing_stats['conditions_found'])}")
    
    # Conversione a DataFrame
    if data_records:
        df = pd.DataFrame(data_records)
        print(f"   • Records creati: {len(df)}")
        return df
    else:
        print("   ❌ Nessun record valido creato!")
        return pd.DataFrame()

def extract_health_level(folder_name):
    """Estrae health level dal nome cartella con gestione robusta"""
    if 'Pitting_degradation_level_' in folder_name:
        level_str = folder_name.replace('Pitting_degradation_level_', '')
        if '(' in level_str:
            return int(level_str.split('(')[0].strip())
        else:
            return int(level_str.strip())
    
    # Fallback: cerca primo numero nella cartella
    numbers = re.findall(r'\d+', folder_name)
    return int(numbers[0]) if numbers else 0

def parse_filename_metadata(filename):
    """
    Parsing robusto dei metadati dal filename
    Format atteso: V{RPM}_{TORQUE}N_{REP}.txt
    """
    pattern = r'V(\d+)_(\d+)N_(\d+)\.txt'
    match = re.search(pattern, filename)
    
    if not match:
        return None
    
    return {
        'rpm': int(match.group(1)),
        'torque': int(match.group(2)),
        'repetition': int(match.group(3))
    }

def load_vibration_data(file_path):
    """Caricamento robusto dati vibrazione"""
    try:
        data = np.loadtxt(file_path)
        
        # Validazione formato dati
        if data.shape[1] != 4:
            print(f"   ⚠️  Formato unexpected: {data.shape}")
            return None
            
        if len(data) < 1000:  # Troppo pochi samples
            return None
            
        return data
        
    except Exception as e:
        return None