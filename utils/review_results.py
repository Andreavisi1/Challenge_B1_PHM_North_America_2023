# ——————————————————————————————————————————————
# Analisi e visualizzazione submission (semplice)
# ——————————————————————————————————————————————
# Uso consigliato nel notebook (in una cella separata):
# %matplotlib inline    # oppure: %matplotlib widget

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Parametri di base
# -------------------------------------------------
N_CLASSES = 11              # cambia qui se il numero di classi è diverso
PROB_COLS = [f"prob_{i}" for i in range(N_CLASSES)]
CONF_COL = "confidence"     # opzionale: 0/1; se manca, le parti relative vengono saltate

# -------------------------------------------------
# Analisi testuale
# -------------------------------------------------
def analyze_submission(df: pd.DataFrame, prob_cols=PROB_COLS, conf_col=CONF_COL):
    """
    Stampa statistiche di base sulla submission:
    - numero di predizioni
    - normalizzazione delle probabilità (somma ~ 1)
    - distribuzione delle classi previste
    - (opzionale) statistiche sulla 'confidence' se presente
    Ritorna: y_pred (np.ndarray), prob_sums (pd.Series)
    """
    print("=" * 60)
    print("ANALISI SUBMISSION")
    print("=" * 60)

    # Controlli minimi
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano le colonne di probabilità: {missing}")

    # Statistiche base
    print("📊 STATISTICHE GENERALI:")
    print(f"   • Numero predizioni: {len(df)}")

    # Somma per riga ~ 1?
    prob_sums = df[prob_cols].sum(axis=1)
    is_norm = np.allclose(prob_sums.values, 1.0, rtol=1e-4, atol=1e-6)
    print(f"   • Normalizzazione: {'✅ OK' if is_norm else '⚠️ NON normalizzato'}")

    # Predizione finale = argmax
    y_pred = df[prob_cols].values.argmax(axis=1)
    uniq, cnts = np.unique(y_pred, return_counts=True)

    print("\n📈 DISTRIBUZIONE PREDIZIONI:")
    for c, n in zip(uniq, cnts):
        pct = 100.0 * n / len(df)
        print(f"   • Classe {c:>2}: {n:6d} ({pct:5.1f}%)")

    # Confidence (se presente)
    if conf_col in df.columns:
        conf_counts = df[conf_col].value_counts().sort_index()
        n_high = int(conf_counts.get(1, 0))
        n_low  = int(conf_counts.get(0, 0))
        mean_conf = df[conf_col].mean()
        print("\n🔎 CONFIDENCE:")
        print(f"   • High: {n_high}")
        print(f"   • Low : {n_low}")
        print(f"   • Media: {mean_conf:.1%}")

    return y_pred, prob_sums

# -------------------------------------------------
# Visualizzazione
# -------------------------------------------------
def plot_submission(df: pd.DataFrame,
                    prob_cols=PROB_COLS,
                    conf_col=CONF_COL,
                    figsize=(15, 10),
                    save_path: str | None = None):
    """
    Crea fino a 6 grafici utili. Se 'confidence' non è presente,
    i grafici che la usano vengono saltati automaticamente.
    - Distribuzione predizioni
    - Probabilità medie per classe
    - (se presente) Distribuzione confidence
    - Distribuzione della max probability
    - Somma probabilità per un sottoinsieme di classi "critiche" (modificabile)
    - (se presente) Max probability separata per confidence
    """
    # Controlli minimi
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano le colonne di probabilità: {missing}")

    has_conf = conf_col in df.columns

    # Predizioni e grandezze derivate
    y_pred = df[prob_cols].values.argmax(axis=1)
    mean_probs = df[prob_cols].mean()
    max_probs = df[prob_cols].max(axis=1)

    # Decidiamo quante assi servono
    # Base: 4 plot (1,2,4,5 del layout originale)
    # +1 se c'è confidence (distribuzione)
    # +1 se c'è confidence (max prob per confidence)
    n_plots = 4 + (1 if has_conf else 0) + (1 if has_conf else 0)

    # Calcolo righe/colonne semplice (max 3 per riga)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # axes in array piatto per indicizzazione semplice
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    plot_idx = 0

    # 1) Distribuzione predizioni
    uniq, cnts = np.unique(y_pred, return_counts=True)
    ax = axes[plot_idx]; plot_idx += 1
    ax.bar(uniq, cnts)
    ax.set_title('Distribuzione Predizioni')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Conteggi')
    ax.set_xticks(range(len(prob_cols)))
    ax.grid(True, alpha=0.3)

    # 2) Probabilità medie per classe
    ax = axes[plot_idx]; plot_idx += 1
    ax.bar(range(len(prob_cols)), mean_probs.values)
    ax.set_title('Probabilità Medie per Classe')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Probabilità media')
    ax.set_xticks(range(len(prob_cols)))
    ax.grid(True, alpha=0.3)

    # 3) (opzionale) Distribuzione confidence
    if has_conf:
        conf_counts = df[conf_col].value_counts().sort_index()
        x = [0, 1]
        y = [int(conf_counts.get(0, 0)), int(conf_counts.get(1, 0))]
        ax = axes[plot_idx]; plot_idx += 1
        ax.bar(x, y)
        ax.set_title('Distribuzione Confidence')
        ax.set_xlabel('Confidence (0=Low, 1=High)')
        ax.set_ylabel('Conteggi')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Low', 'High'])
        ax.grid(True, alpha=0.3)

    # 4) Distribuzione Max Probability
    ax = axes[plot_idx]; plot_idx += 1
    ax.hist(max_probs, bins=30)
    ax.set_title('Distribuzione Max Probabilità')
    ax.set_xlabel('Max Probability')
    ax.set_ylabel('Frequenza')
    ax.axvline(max_probs.mean(), linestyle='--', linewidth=2, label=f"Media: {max_probs.mean():.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5) Somma Probabilità di classi "critiche" (personalizza qui)
    critical_classes = [5, 7, 9]  # modifica liberamente in base al tuo caso
    critical_classes = [c for c in critical_classes if 0 <= c < len(prob_cols)]
    if critical_classes:
        sums = [df[f"prob_{c}"].sum() for c in critical_classes]
        ax = axes[plot_idx]; plot_idx += 1
        ax.bar(critical_classes, sums)
        ax.set_title('Somma Prob. Classi Critiche')
        ax.set_xlabel('Classe')
        ax.set_ylabel('Somma Probabilità')
        ax.set_xticks(critical_classes)
        ax.grid(True, alpha=0.3)

    # 6) (opzionale) Max prob per confidence
    if has_conf:
        hi = df[df[conf_col] == 1][prob_cols].max(axis=1)
        lo = df[df[conf_col] == 0][prob_cols].max(axis=1)
        ax = axes[plot_idx]; plot_idx += 1
        if len(hi) > 0:
            ax.hist(hi, bins=20, alpha=0.6, label=f'High Conf ({len(hi)})')
        if len(lo) > 0:
            ax.hist(lo, bins=20, alpha=0.6, label=f'Low Conf ({len(lo)})')
        ax.set_title('Max Prob per Confidence')
        ax.set_xlabel('Max Probability')
        ax.set_ylabel('Frequenza')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Nasconde eventuali assi non usati
    for k in range(plot_idx, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout(pad=2.0)

    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"💾 Salvato: {save_path}")
        except Exception as e:
            print(f"⚠️ Errore nel salvataggio: {e}")

    plt.show()
    print("✅ Grafici visualizzati.")