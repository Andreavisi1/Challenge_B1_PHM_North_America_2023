import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Analisi scientifica
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

def comprehensive_dataset_analysis(df, X_features, y):
    """
    Analisi esplorativa completa del dataset PHM
    
    Obiettivi:
    1. Comprensione struttura dati e distribuzione samples
    2. Identificazione pattern legati a degradation
    3. Analisi robustezza rispetto a condizioni operative  
    4. Detection di outliers e anomalie
    5. Feature correlation analysis
    """
    
    print("🔍 INIZIANDO ANALISI ESPLORATIVA COMPLETA")
    print("="*60)
    
    # === ANALISI STRUTTURA DATASET ===
    print("\n📊 STRUTTURA DATASET:")
    print(f"   • Samples totali: {len(df)}")
    print(f"   • Features totali: {len(X_features.columns)}")
    print(f"   • Health levels: {sorted(y.unique())}")
    print(f"   • RPM range: {df['velocita'].min()} - {df['velocita'].max()}")
    print(f"   • Torque range: {df['torque'].min()} - {df['torque'].max()}")
    print(f"   • Repetitions per condition: {df['repetition'].value_counts().describe()}")
    
    # === ANALISI DISTRIBUZIONE HEALTH LEVELS ===
    plot_health_level_distribution(df, y)
    
    # === ANALISI CONDIZIONI OPERATIVE ===
    plot_operating_conditions_analysis(df)
    
    # === ANALISI FEATURES vs HEALTH LEVELS ===
    plot_features_vs_health_analysis(X_features, y)
    
    # === CORRELATION ANALYSIS ===
    plot_correlation_analysis(X_features, y)
    
    # === OUTLIER DETECTION ===
    outlier_analysis(X_features, y)
    
    # === PHYSICS VALIDATION ===
    physics_based_validation(df, X_features, y)
    
    print("\n✅ ANALISI ESPLORATIVA COMPLETATA!")

def plot_health_level_distribution(df, y):
    """Analizza distribuzione health levels nel dataset"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analisi Distribuzione Health Levels', fontsize=16, fontweight='bold')
    
    # 1. Distribuzione generale health levels
    health_counts = y.value_counts().sort_index()
    axes[0,0].bar(health_counts.index, health_counts.values, alpha=0.7, color='skyblue')
    axes[0,0].set_xlabel('Health Level')
    axes[0,0].set_ylabel('Numero Samples')
    axes[0,0].set_title('Distribuzione Health Levels nel Dataset')
    axes[0,0].grid(True, alpha=0.3)
    
    # Evidenzia livelli mancanti
    all_levels = set(range(11))
    present_levels = set(health_counts.index)
    missing_levels = all_levels - present_levels
    
    if missing_levels:
        axes[0,0].text(0.02, 0.98, f'Livelli MANCANTI: {sorted(missing_levels)}', 
                      transform=axes[0,0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
    
    # 2. Health levels vs RPM
    pivot_rpm = df.pivot_table(values='file_name', index='health_level', 
                              columns='velocita', aggfunc='count', fill_value=0)
    
    sns.heatmap(pivot_rpm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('Health Levels vs RPM (Sample Counts)')
    axes[0,1].set_xlabel('RPM')
    axes[0,1].set_ylabel('Health Level')
    
    # 3. Health levels vs Torque  
    pivot_torque = df.pivot_table(values='file_name', index='health_level', 
                                 columns='torque', aggfunc='count', fill_value=0)
    
    sns.heatmap(pivot_torque, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1,0])
    axes[1,0].set_title('Health Levels vs Torque (Sample Counts)')
    axes[1,0].set_xlabel('Torque (Nm)')
    axes[1,0].set_ylabel('Health Level')
    
    # 4. Consistency check repetitions
    rep_stats = df.groupby(['health_level', 'velocita', 'torque'])['repetition'].count()
    axes[1,1].hist(rep_stats.values, bins=10, alpha=0.7, color='lightcoral')
    axes[1,1].set_xlabel('Repetitions per Condition')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Consistenza Repetitions per Condizione')
    axes[1,1].grid(True, alpha=0.3)
    
    # Stats summary
    axes[1,1].text(0.02, 0.98, 
                  f'Media rep: {rep_stats.mean():.1f}\nStd rep: {rep_stats.std():.1f}', 
                  transform=axes[1,1].transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print(f"\n🎯 INSIGHTS HEALTH LEVELS:")
    print(f"   • Livelli presenti: {sorted(present_levels)}")
    print(f"   • Livelli mancanti: {sorted(missing_levels)} (by design per challenge)")
    print(f"   • Samples per level: min={health_counts.min()}, max={health_counts.max()}")
    print(f"   • Repetitions: media={rep_stats.mean():.1f}, std={rep_stats.std():.1f}")

def plot_operating_conditions_analysis(df):
    """Analisi approfondita delle condizioni operative"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('⚙️ Analisi Condizioni Operative', fontsize=16, fontweight='bold')
    
    # 1. Operating conditions coverage map
    operating_matrix = df.pivot_table(values='file_name', index='torque', 
                                     columns='velocita', aggfunc='count', fill_value=0)
    
    sns.heatmap(operating_matrix, annot=True, fmt='d', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Coverage Map: RPM × Torque')
    axes[0,0].set_xlabel('RPM')
    axes[0,0].set_ylabel('Torque (Nm)')
    
    # 2. Distribuzione RPM
    rpm_counts = df['velocita'].value_counts().sort_index()
    axes[0,1].bar(rpm_counts.index, rpm_counts.values, alpha=0.7, color='orange')
    axes[0,1].set_xlabel('RPM')
    axes[0,1].set_ylabel('Numero Samples')
    axes[0,1].set_title('Distribuzione RPM')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Distribuzione Torque
    torque_counts = df['torque'].value_counts().sort_index()
    axes[1,0].bar(torque_counts.index, torque_counts.values, alpha=0.7, color='green')
    axes[1,0].set_xlabel('Torque (Nm)')
    axes[1,0].set_ylabel('Numero Samples')
    axes[1,0].set_title('Distribuzione Torque')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Health level distribution per operating region
    df_temp = df.copy()
    df_temp['rpm_category'] = pd.cut(df_temp['velocita'], bins=3, labels=['Low', 'Medium', 'High'])
    df_temp['torque_category'] = pd.cut(df_temp['torque'], bins=3, labels=['Low', 'Medium', 'High'])
    
    health_by_condition = df_temp.groupby(['rpm_category', 'torque_category'])['health_level'].nunique()
    
    # Visualization della diversity
    condition_diversity = {}
    for (rpm_cat, torque_cat), unique_levels in health_by_condition.items():
        condition_diversity[f'{rpm_cat}-{torque_cat}'] = unique_levels
    
    categories = list(condition_diversity.keys())
    diversity_values = list(condition_diversity.values())
    
    axes[1,1].bar(categories, diversity_values, alpha=0.7, color='purple')
    axes[1,1].set_xlabel('Operating Region')
    axes[1,1].set_ylabel('Unique Health Levels')
    axes[1,1].set_title('Health Level Diversity per Region')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_features_vs_health_analysis(X_features, y):
    """Analisi relazione features vs health levels"""
    
    # Seleziona features chiave per analysis
    feature_groups = {
        'RMS': [col for col in X_features.columns if 'rms' in col],
        'Peak-to-Peak': [col for col in X_features.columns if 'ptp' in col],
        'Frequency Bands': [col for col in X_features.columns if 'band_' in col],
        'Cross-Correlation': [col for col in X_features.columns if 'corr_' in col]
    }
    
    for group_name, features in feature_groups.items():
        if not features:
            continue
            
        # Prendi le prime 4 features più significative del gruppo
        significant_features = features[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Features {group_name} vs Health Levels', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(significant_features):
            if idx >= 4:
                break
                
            row, col = idx // 2, idx % 2
            
            # Box plot per health level
            health_levels = sorted(y.unique())
            feature_data = [X_features[y == hl][feature].values for hl in health_levels]
            
            axes[row, col].boxplot(feature_data, labels=health_levels)
            axes[row, col].set_xlabel('Health Level')
            axes[row, col].set_ylabel('Feature Value')
            axes[row, col].set_title(f'{feature}')
            axes[row, col].grid(True, alpha=0.3)
            
            # Calcola correlation con health level
            corr_coef = X_features[feature].corr(y)
            axes[row, col].text(0.02, 0.98, f'Corr: {corr_coef:.3f}', 
                              transform=axes[row, col].transAxes, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    # Feature importance analysis basata su correlation
    feature_correlations = {}
    for feature in X_features.columns:
        if feature not in ['health_level', 'velocita', 'torque', 'repetition']:
            corr = abs(X_features[feature].corr(y))
            if not np.isnan(corr):
                feature_correlations[feature] = corr
    
    # Top features by correlation
    top_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Plot top features
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    features_names = [f[0] for f in top_features]
    correlations = [f[1] for f in top_features]
    
    bars = ax.barh(range(len(features_names)), correlations, alpha=0.7, color='teal')
    ax.set_yticks(range(len(features_names)))
    ax.set_yticklabels([f.replace('_', ' ').title()[:30] for f in features_names])
    ax.set_xlabel('Absolute Correlation with Health Level')
    ax.set_title('Top 15 Features per Correlation con Health Level')
    ax.grid(True, alpha=0.3)
    
    # Aggiungi valori sulle barre
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_analysis(X_features, y):
    """Analisi correlazioni tra features"""
    
    # Seleziona subset di features per visualizzazione
    feature_cols = X_features.columns.tolist()
    if len(feature_cols) > 30:
        # Prendi top 30 by variance
        variances = X_features.var().sort_values(ascending=False)
        feature_cols = variances.head(30).index.tolist()
    
    # Calcola correlation matrix
    corr_matrix = X_features[feature_cols].corr()
    
    # Plot correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 1. Full correlation matrix
    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, 
                square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('🔗 Correlation Matrix Features (Top 30)')
    axes[0].tick_params(axis='both', labelsize=8)
    
    # 2. High correlation pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_val
                ))
    
    # Scatter plot delle top correlazioni
    if high_corr_pairs:
        top_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:6]
        
        gs = fig.add_gridspec(2, 3, left=0.55, right=0.95, top=0.9, bottom=0.1)
        subplot_axes = []
        for r in range(2):
            for c in range(3):
                subplot_axes.append(fig.add_subplot(gs[r, c]))
        
        for idx, (feat1, feat2, corr_val) in enumerate(top_pairs):
            if idx >= 6:
                break
            
            ax = subplot_axes[idx]
            ax.scatter(X_features[feat1], X_features[feat2], alpha=0.6, c=y, 
                      cmap='viridis', s=20)
            ax.set_xlabel(feat1.replace('_', ' ')[:15])
            ax.set_ylabel(feat2.replace('_', ' ')[:15])
            ax.set_title(f'Corr: {corr_val:.3f}', fontsize=10)
            ax.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, '✅ Nessuna correlazione alta (>0.8)\nTrovata tra features', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        axes[1].set_title('High Correlation Analysis')
    
    plt.tight_layout()
    plt.show()

def outlier_analysis(X_features, y):
    """Detection e analisi outliers"""
    
    print(f"\nANALISI OUTLIERS:")
    
    # 1. IQR-based outlier detection
    outliers_by_feature = {}
    
    for feature in X_features.columns[:10]:
        Q1 = X_features[feature].quantile(0.25)
        Q3 = X_features[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = X_features[(X_features[feature] < lower_bound) | 
                             (X_features[feature] > upper_bound)]
        outliers_by_feature[feature] = len(outliers)
    
    # 2. Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(X_features.fillna(X_features.median()))
    
    n_outliers = len(np.where(outlier_labels == -1)[0])
    outlier_indices = np.where(outlier_labels == -1)[0]
    
    print(f"   • Outliers multivariati: {n_outliers} ({n_outliers/len(X_features)*100:.1f}%)")
    
    # 3. Visualizzazione outliers
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analisi Outliers', fontsize=16, fontweight='bold')
    
    # Plot 1: Outliers per feature
    features_with_outliers = list(outliers_by_feature.keys())[:10]
    outlier_counts = [outliers_by_feature[f] for f in features_with_outliers]
    
    axes[0,0].bar(range(len(features_with_outliers)), outlier_counts, alpha=0.7, color='red')
    axes[0,0].set_xticks(range(len(features_with_outliers)))
    axes[0,0].set_xticklabels([f.replace('_', '\n')[:15] for f in features_with_outliers], rotation=45)
    axes[0,0].set_ylabel('Numero Outliers')
    axes[0,0].set_title('Outliers per Feature (IQR Method)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Box plot feature con più outliers
    if features_with_outliers:
        worst_feature = max(outliers_by_feature, key=outliers_by_feature.get)
        health_levels = sorted(y.unique())
        data_by_health = [X_features[y == hl][worst_feature].values for hl in health_levels]
        axes[0,1].boxplot(data_by_health, labels=health_levels)
        axes[0,1].set_title(f'Outliers in {worst_feature}')
        axes[0,1].set_xlabel('Health Level')
        axes[0,1].set_ylabel('Feature Value')
    
    # Plot 3: Health level distribution outliers vs normal
    if len(outlier_indices) > 0:
        outlier_health = y.iloc[outlier_indices]
        normal_health = y.iloc[~np.isin(range(len(y)), outlier_indices)]
        
        axes[1,0].hist([normal_health, outlier_health], label=['Normal', 'Outliers'], 
                      alpha=0.7, color=['blue', 'red'])
        axes[1,0].set_xlabel('Health Level')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Health Level Distribution: Normal vs Outliers')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: PCA visualization
    if len(X_features.columns) > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_features.fillna(X_features.median()))
        
        axes[1,1].scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.5, label='Normal')
        if len(outlier_indices) > 0:
            axes[1,1].scatter(X_pca[outlier_indices, 0], X_pca[outlier_indices, 1], 
                            c='red', marker='x', s=50, label='Outliers')
        axes[1,1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1,1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1,1].set_title('PCA: Outliers Visualization')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def physics_based_validation(df, X_features, y):
    """Validazione delle features basata su principi fisici"""
    
    print(f"\n🔬 VALIDAZIONE PHYSICS-BASED:")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Physics-Based Validation', fontsize=16, fontweight='bold')
    
    # 1. RMS vs RPM (dovrebbe esserci correlazione positiva)
    rms_features = [col for col in X_features.columns if 'rms' in col]
    if rms_features and 'velocita' in df.columns:
        rms_feature = rms_features[0]
        axes[0,0].scatter(df['velocita'], X_features[rms_feature], 
                         alpha=0.6, c=y, cmap='viridis')
        axes[0,0].set_xlabel('RPM')
        axes[0,0].set_ylabel(rms_feature)
        corr = X_features[rms_feature].corr(df['velocita'])
        axes[0,0].set_title(f'RMS vs RPM (Corr: {corr:.3f})')
        axes[0,0].grid(True, alpha=0.3)
    
    # 2. Energy distribution across frequency bands
    band_features = [col for col in X_features.columns if 'band_' in col and 'ax_' in col]
    if band_features:
        band_means = [X_features[f].mean() for f in band_features[:6]]
        band_labels = [f.split('_')[-2] + '-' + f.split('_')[-1] for f in band_features[:6]]
        axes[0,1].bar(range(len(band_means)), band_means, alpha=0.7, color='coral')
        axes[0,1].set_xticks(range(len(band_labels)))
        axes[0,1].set_xticklabels(band_labels, rotation=45)
        axes[0,1].set_ylabel('Mean Energy')
        axes[0,1].set_title('Energy Distribution Across Frequency Bands')
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. Cross-correlation distribution
    corr_features = [col for col in X_features.columns if 'corr_' in col]
    if corr_features:
        corr_values = X_features[corr_features].values.flatten()
        axes[1,0].hist(corr_values, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1,0].set_xlabel('Correlation Value')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Cross-Axis Correlations')
        axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Feature evolution with health level
    # Trova feature più monotonica
    monotonic_scores = {}
    for feature in X_features.columns[:20]:
        health_means = X_features.groupby(y)[feature].mean().sort_index()
        diffs = health_means.diff().dropna()
        if len(diffs) > 0:
            monotonic_score = abs(sum(diffs > 0) - sum(diffs < 0)) / len(diffs)
            monotonic_scores[feature] = monotonic_score
    
    if monotonic_scores:
        best_monotonic = max(monotonic_scores, key=monotonic_scores.get)
        health_levels = sorted(y.unique())
        health_means = X_features.groupby(y)[best_monotonic].mean().sort_index()
        health_stds = X_features.groupby(y)[best_monotonic].std().sort_index()
        
        axes[1,1].errorbar(health_levels, health_means.values, 
                          yerr=health_stds.values, fmt='o-', capsize=5, 
                          alpha=0.7, color='purple')
        axes[1,1].set_xlabel('Health Level')
        axes[1,1].set_ylabel(best_monotonic[:30])
        axes[1,1].set_title(f'Most Monotonic Feature (Score: {monotonic_scores[best_monotonic]:.2f})')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()