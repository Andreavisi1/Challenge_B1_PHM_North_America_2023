import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Analisi scientifica
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

def plot_correlation_matrix_full(X, title="Correlation Matrix", threshold=0.85, figsize=(14, 12)):
    """
    Matrice di correlazione COMPLETA con entrambi i triangoli visibili.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calcola correlazione
    corr = X.corr().abs()
    
    # NON MASCHERARE - mostra matrice completa
    sns.heatmap(corr, cmap='coolwarm', center=0.5,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=0, vmax=1, ax=ax, annot=False)
    
    # Evidenzia TUTTE le correlazioni sopra threshold (entrambi i triangoli)
    n_high_corr = 0
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if i != j and corr.iloc[i, j] >= threshold:
                # Rettangolo rosso
                rect = plt.Rectangle((j, i), 1, 1, fill=False, 
                                    edgecolor='red', lw=2)
                ax.add_patch(rect)
                if j > i:  # Conta solo una volta
                    n_high_corr += 1
    
    ax.set_title(f'{title}\n({len(X.columns)} features, {n_high_corr} correlazioni ≥ {threshold})', 
                fontsize=14, fontweight='bold')
    
    # Ruota labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.show()

def remove_multicollinearity(X, threshold=0.85):
    """
    Rimuove features altamente correlate mantenendo quelle più informative.
    Usa threshold più permissivo per approccio conservativo.
    """
    
    print(f"\n📊 RIMOZIONE MULTICOLLINEARITÀ (threshold={threshold})")
    print("="*60)
    
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

def conservative_feature_selection(X, y, min_features=40, critical_features=None):
    """
    Selezione conservativa che mantiene features fisicamente motivate.
    Usa multiple metriche per evitare di eliminare indicatori importanti.
    
    Args:
        X: DataFrame con features
        y: Target (health levels)
        min_features: Numero minimo di features da mantenere
        critical_features: Lista di features sempre da mantenere
    
    Returns:
        selected_features: Lista features selezionate
        selection_scores: Dizionario con tutti gli scores
    """
    
    print(f"\n🛡️ SELEZIONE CONSERVATIVA FEATURES (min={min_features})")
    print("="*60)
    
    # 1. Random Forest importance
    print("   📊 Calcolo Random Forest importance...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importance_rf = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. Mutual Information
    print("   📊 Calcolo Mutual Information...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    importance_mi = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # 3. Correlation with target
    print("   📊 Calcolo correlazioni con target...")
    importance_corr = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values('correlation', ascending=False)
    
    # 4. Variance-based importance
    print("   📊 Calcolo variance scores...")
    importance_var = pd.DataFrame({
        'feature': X.columns,
        'variance': X.var()
    }).sort_values('variance', ascending=False)
    
    # === SELEZIONE MULTI-CRITERIO ===
    keep_features = set()
    
    # Prendi top K da ogni metodo
    k_per_method = max(30, min_features // 3)
    
    print(f"\n   Seleziono top {k_per_method} features per metodo:")
    
    # Random Forest top features
    rf_top = importance_rf.head(k_per_method)['feature'].tolist()
    keep_features.update(rf_top)
    print(f"   • Random Forest: {len(rf_top)} features")
    
    # Mutual Information top features
    mi_top = importance_mi.head(k_per_method)['feature'].tolist()
    keep_features.update(mi_top)
    print(f"   • Mutual Information: {len(mi_top)} features")
    
    # Correlation top features
    corr_top = importance_corr.head(k_per_method)['feature'].tolist()
    keep_features.update(corr_top)
    print(f"   • Correlation: {len(corr_top)} features")
    
    # High variance features (top 20)
    var_top = importance_var.head(20)['feature'].tolist()
    keep_features.update(var_top)
    print(f"   • High Variance: {len(var_top)} features")
    
    # === AGGIUNGI FEATURES CRITICHE DOMAIN-SPECIFIC ===
    if critical_features is None:
        critical_features = [
            'rpm', 'ipi_cv', 'ipi_mean', 'ipi_std',  # Tachometer
            'ax_envelope_rms', 'ay_envelope_rms', 'az_envelope_rms',  # Envelope
            'ax_rms', 'ay_rms', 'az_rms',  # RMS base
            'corr_xy', 'corr_xz', 'corr_yz',  # Cross-correlations
            'velocita', 'torque'  # Operating conditions
        ]
    
    # Aggiungi solo quelle presenti nel dataset
    critical_present = [f for f in critical_features if f in X.columns]
    keep_features.update(critical_present)
    print(f"   • Domain-critical: {len(critical_present)} features")
    
    # === VERIFICA COVERAGE FISICA ===
    print("\n   🔬 Verifica coverage domini fisici:")
    
    domains = {
        'Time domain': [f for f in keep_features if any(x in f for x in ['mean', 'std', 'rms', 'skew', 'kurt', 'ptp'])],
        'Frequency domain': [f for f in keep_features if 'band_' in f],
        'Envelope': [f for f in keep_features if 'envelope' in f],
        'Tachometer': [f for f in keep_features if any(x in f for x in ['rpm', 'ipi', 'pulse'])],
        'Cross-axis': [f for f in keep_features if 'corr_' in f],
        'Operating': [f for f in keep_features if f in ['velocita', 'torque', 'v_times_t', 'v_sq', 't_sq']]
    }
    
    for domain, features in domains.items():
        print(f"   • {domain}: {len(features)} features")
        if len(features) == 0:
            print(f"     ⚠️ WARNING: Nessuna feature per {domain}!")
    
    # === CALCOLA SCORE AGGREGATO ===
    selected_features = list(keep_features)
    
    # Crea dataframe con tutti gli scores
    all_scores = pd.DataFrame({'feature': selected_features})
    
    # Aggiungi scores normalizzati
    for feat in selected_features:
        rf_rank = len(X.columns) - importance_rf[importance_rf['feature']==feat].index[0] if feat in importance_rf['feature'].values else 0
        mi_rank = len(X.columns) - importance_mi[importance_mi['feature']==feat].index[0] if feat in importance_mi['feature'].values else 0
        corr_rank = len(X.columns) - importance_corr[importance_corr['feature']==feat].index[0] if feat in importance_corr['feature'].values else 0
        
        all_scores.loc[all_scores['feature']==feat, 'rf_rank'] = rf_rank
        all_scores.loc[all_scores['feature']==feat, 'mi_rank'] = mi_rank
        all_scores.loc[all_scores['feature']==feat, 'corr_rank'] = corr_rank
        all_scores.loc[all_scores['feature']==feat, 'is_critical'] = feat in critical_present
    
    # Score aggregato
    all_scores['aggregate_score'] = (
        all_scores['rf_rank'] + 
        all_scores['mi_rank'] + 
        all_scores['corr_rank'] + 
        all_scores['is_critical'] * len(X.columns)  # Bonus per features critiche
    )
    
    all_scores = all_scores.sort_values('aggregate_score', ascending=False)
    
    # Se abbiamo troppe features, taglia basandoti sullo score aggregato
    if len(selected_features) > min_features * 1.5:
        print(f"\n   ✂️ Riduco da {len(selected_features)} a ~{min_features} features")
        selected_features = all_scores.head(min_features)['feature'].tolist()
    
    print(f"\n✅ SELEZIONE COMPLETATA:")
    print(f"   • Features finali: {len(selected_features)}")
    print(f"   • Riduzione: {len(X.columns)} → {len(selected_features)} ({(1-len(selected_features)/len(X.columns))*100:.1f}%)")
    
    # Prepara output dettagliato
    selection_scores = {
        'selected_features': selected_features,
        'rf_importance': importance_rf,
        'mi_scores': importance_mi,
        'correlations': importance_corr,
        'variance_scores': importance_var,
        'aggregate_scores': all_scores,
        'domain_coverage': domains
    }
    
    return selected_features, selection_scores

def visualize_conservative_selection(X_before, X_after, selection_scores, y, figsize=(20, 12)):
    """
    Visualizza il processo di selezione conservativa multi-metodo.
    """
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('🛡️ Conservative Feature Selection: Multi-Method Approach', 
                 fontsize=16, fontweight='bold')
    
    # === ROW 1: Metodi di selezione ===
    
    # 1.1 Random Forest importance
    ax1 = fig.add_subplot(gs[0, 0])
    rf_top = selection_scores['rf_importance'].head(15)
    ax1.barh(range(len(rf_top)), rf_top['importance'].values, color='green', alpha=0.7)
    ax1.set_yticks(range(len(rf_top)))
    ax1.set_yticklabels([f[:15] for f in rf_top['feature']], fontsize=8)
    ax1.set_xlabel('Importance')
    ax1.set_title('Random Forest\nImportance')
    ax1.grid(axis='x', alpha=0.3)
    
    # 1.2 Mutual Information
    ax2 = fig.add_subplot(gs[0, 1])
    mi_top = selection_scores['mi_scores'].head(15)
    ax2.barh(range(len(mi_top)), mi_top['mi_score'].values, color='blue', alpha=0.7)
    ax2.set_yticks(range(len(mi_top)))
    ax2.set_yticklabels([f[:15] for f in mi_top['feature']], fontsize=8)
    ax2.set_xlabel('MI Score')
    ax2.set_title('Mutual\nInformation')
    ax2.grid(axis='x', alpha=0.3)
    
    # 1.3 Correlation with target
    ax3 = fig.add_subplot(gs[0, 2])
    corr_top = selection_scores['correlations'].head(15)
    ax3.barh(range(len(corr_top)), corr_top['correlation'].values, color='red', alpha=0.7)
    ax3.set_yticks(range(len(corr_top)))
    ax3.set_yticklabels([f[:15] for f in corr_top['feature']], fontsize=8)
    ax3.set_xlabel('|Correlation|')
    ax3.set_title('Target\nCorrelation')
    ax3.grid(axis='x', alpha=0.3)
    
    # 1.4 Consenso tra metodi
    ax4 = fig.add_subplot(gs[0, 3])
    
    # Conta in quanti metodi appare ogni feature
    rf_set = set(selection_scores['rf_importance'].head(30)['feature'])
    mi_set = set(selection_scores['mi_scores'].head(30)['feature'])
    corr_set = set(selection_scores['correlations'].head(30)['feature'])
    
    all_methods = len(rf_set & mi_set & corr_set)
    two_methods = len((rf_set & mi_set) | (rf_set & corr_set) | (mi_set & corr_set)) - all_methods
    one_method = len(rf_set | mi_set | corr_set) - all_methods - two_methods
    
    ax4.pie([all_methods, two_methods, one_method], 
           labels=['3 metodi', '2 metodi', '1 metodo'],
           colors=['#FFD700', '#C0C0C0', '#CD7F32'],  # Codici hex per gold, silver, bronze
           autopct='%1.0f%%')
    ax4.set_title('Consenso\ntra Metodi')
    
    # === ROW 2: Domain coverage e statistiche ===
    
    # 2.1 Coverage per dominio
    ax5 = fig.add_subplot(gs[1, :2])
    domains = selection_scores['domain_coverage']
    domain_names = list(domains.keys())
    domain_counts = [len(v) for v in domains.values()]
    
    colors_domain = plt.cm.Set3(np.linspace(0, 1, len(domain_names)))
    bars = ax5.bar(domain_names, domain_counts, color=colors_domain, alpha=0.7)
    ax5.set_ylabel('Number of Features')
    ax5.set_title('Domain Coverage Analysis')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, domain_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    # 2.2 Aggregate scores
    ax6 = fig.add_subplot(gs[1, 2:])
    agg_scores = selection_scores['aggregate_scores'].head(20)
    ax6.barh(range(len(agg_scores)), agg_scores['aggregate_score'].values, 
            color='purple', alpha=0.7)
    ax6.set_yticks(range(len(agg_scores)))
    ax6.set_yticklabels([f[:20] for f in agg_scores['feature']], fontsize=8)
    ax6.set_xlabel('Aggregate Score')
    ax6.set_title('Top 20 Features by Aggregate Score')
    ax6.grid(axis='x', alpha=0.3)
    
    # === ROW 3: Validazione e confronto ===
    
    # 3.1 Riduzione features
    ax7 = fig.add_subplot(gs[2, 0])
    stages = ['Initial', 'After\nMulticoll.', 'Final\nSelection']
    counts = [75, len(X_before.columns), len(X_after.columns)]
    colors_stages = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax7.bar(stages, counts, color=colors_stages, alpha=0.7)
    ax7.set_ylabel('Number of Features')
    ax7.set_title('Feature Reduction')
    
    for bar, count in zip(bars, counts):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
str(count), ha='center', va='bottom')
    
    # 3.2 Feature stability
    ax8 = fig.add_subplot(gs[2, 1])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stability_scores = []
    
    for train_idx, val_idx in kf.split(X_before):
        X_train = X_before.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        # Ricalcola importance su fold
        rf_temp = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train[X_after.columns], y_train)
        
        # Conta features con importance > threshold
        important = sum(rf_temp.feature_importances_ > 0.01)
        stability_scores.append(important)
    
    ax8.bar(range(1, 6), stability_scores, color='green', alpha=0.7)
    ax8.set_xlabel('CV Fold')
    ax8.set_ylabel('Important Features')
    ax8.set_title('Feature Stability')
    ax8.axhline(np.mean(stability_scores), color='red', linestyle='--',
               label=f'Mean: {np.mean(stability_scores):.0f}')
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)
    
    # 3.3 Final correlation matrix
    ax9 = fig.add_subplot(gs[2, 2:])
    if len(X_after.columns) <= 30:
        corr_final = X_after.corr().abs()
        im = ax9.imshow(corr_final, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
        ax9.set_title(f'Final Correlation Matrix ({len(X_after.columns)} features)')
        plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
    else:
        # Istogramma correlazioni
        corr_values = X_after.corr().abs().values
        upper_tri = corr_values[np.triu_indices_from(corr_values, k=1)]
        ax9.hist(upper_tri, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax9.axvline(0.85, color='red', linestyle='--', label='Threshold')
        ax9.set_xlabel('|Correlation|')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Final Correlation Distribution')
        ax9.legend()
        ax9.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Report
    print("\n📊 CONSERVATIVE SELECTION SUMMARY:")
    print(f"   • Metodi utilizzati: Random Forest, Mutual Information, Correlation, Variance")
    print(f"   • Features in tutti e 3 i metodi principali: {all_methods}")
    print(f"   • Features protette (domain-critical): {sum(selection_scores['aggregate_scores']['is_critical'])}")
    print(f"   • Coverage domini: {len([d for d in domains.values() if len(d) > 0])}/6")