# ============================================
# MODEL SELECTION - CLASSIFICAZIONE
# ============================================

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pprint import pprint
from contextlib import contextmanager
from tqdm.auto import tqdm

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    RandomizedSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Modelli di classificazione
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.linear_model import RidgeClassifier, LogisticRegression

# Boosting esterni
from xgboost import XGBClassifier
import lightgbm as lgb


class ModelSelectorClassification:
    """
    Classe per la selezione automatica di modelli di classificazione.
    
    Features:
    - Tuning automatico di 8 modelli diversi con RandomizedSearchCV
    - Valutazione completa con metriche multiple
    - Analisi di stabilità e overfitting
    - Permutation importance
    - Salvataggio automatico del miglior modello
    - Possibilità di creare ensemble
    """
    
    def __init__(self, scoring="accuracy", cv_folds=3, random_state=42):
        """
        Inizializza il selettore di modelli.
        
        Args:
            scoring (str): Metrica principale ('accuracy', 'f1_macro', 'balanced_accuracy')
            cv_folds (int): Numero di fold per la cross-validation
            random_state (int): Seed per la riproducibilità
        """
        self.scoring = scoring
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.random_state = random_state
        self.le = LabelEncoder()
        self.searches = {}
        self.val_results = []
        self.best_model = None
        self.best_model_name = None
        
        # Progress bar helper
        self._setup_tqdm_joblib()
        
        # Setup modelli e parametri
        self._setup_models()
    
    @contextmanager
    def _tqdm_joblib(self, tqdm_object):
        """Helper per mostrare progress bar con joblib."""
        try:
            from joblib import parallel
            class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    tqdm_object.update(n=self.batch_size)
                    return super().__call__(*args, **kwargs)
            old_callback = parallel.BatchCompletionCallBack
            parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
            yield
        finally:
            parallel.BatchCompletionCallBack = old_callback
            tqdm_object.close()
    
    def _setup_tqdm_joblib(self):
        """Setup del context manager per tqdm."""
        self.tqdm_joblib = self._tqdm_joblib
    
    def _setup_models(self):
        """Setup di tutti i modelli e i loro parametri."""
        
        # Random Forest
        self.rf = Pipeline([
            ("model", RandomForestClassifier(random_state=self.random_state, n_jobs=-1))
        ])
        self.rf_param_dist = {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [6, 8, 10, 12, None],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
            "model__class_weight": [None, "balanced"]
        }

        # Histogram Gradient Boosting
        self.hgb = Pipeline([
            ("model", HistGradientBoostingClassifier(random_state=self.random_state))
        ])
        self.hgb_param_dist = {
            "model__max_depth": [None, 4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [200, 400, 800],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__min_samples_leaf": [10, 20, 30, 50]
        }

        # XGBoost
        self.xgb = Pipeline([
            ("model", XGBClassifier(
                random_state=self.random_state,
                n_estimators=200,
                n_jobs=-1,
                verbosity=0,
                objective="multi:softprob",
                eval_metric="mlogloss"
            ))
        ])
        self.xgb_param_dist = {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [3, 4, 6, 8],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.15],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
            "model__reg_alpha": [0, 0.1, 1],
            "model__reg_lambda": [1, 1.5, 2],
        }

        # LightGBM
        self.lgbm = Pipeline([
            ("model", lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1))
        ])
        self.lgbm_param_dist = {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [3, 4, 6, 8, -1],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.15],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
            "model__reg_alpha": [0, 0.1, 1],
            "model__reg_lambda": [0, 0.1, 1],
            "model__num_leaves": [31, 50, 100, 200],
            "model__class_weight": [None, "balanced"]
        }

        # Gradient Boosting
        self.gb = Pipeline([
            ("model", GradientBoostingClassifier(random_state=self.random_state))
        ])
        self.gb_param_dist = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [3, 4, 6, 8],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.15],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__max_features": ["sqrt", "log2", 0.3, 0.5]
        }

        # Extra Trees
        self.et = Pipeline([
            ("model", ExtraTreesClassifier(random_state=self.random_state, n_jobs=-1))
        ])
        self.et_param_dist = {
            "model__n_estimators": [200, 400, 600, 800],
            "model__max_depth": [6, 8, 10, 12, None],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__min_samples_split": [2, 5, 10],
            "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
            "model__class_weight": [None, "balanced"]
        }

        # Ridge Classifier
        self.ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RidgeClassifier())
        ])
        self.ridge_param_dist = {
            "model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "model__class_weight": [None, "balanced"]
        }

        # Logistic Regression con Elastic Net
        self.elastic = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=5000, 
                n_jobs=-1, random_state=self.random_state
            ))
        ])
        self.elastic_param_dist = {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.1, 0.5, 0.7, 0.9],
            "model__class_weight": [None, "balanced"]
        }
    
    def _fit_search(self, name, pipe, param_dist, X_tr, y_tr_enc, n_iter=25):
        """Esegue hyperparameter tuning per un modello."""
        print(f"\n[*] Tuning {name}...")
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv,
            verbose=0,
            random_state=self.random_state,
            n_jobs=-1,
            error_score="raise"
        )
        
        total_steps = n_iter * self.cv.get_n_splits(X_tr, y_tr_enc)
        with self.tqdm_joblib(tqdm(total=total_steps, desc=f"Tuning {name}", leave=True)):
            search.fit(X_tr, y_tr_enc)
        
        print(f"[+] Best {self.scoring.upper()} (cv) {name}: {search.best_score_:.4f}")
        print(f"[+] Best params {name}:")
        pprint(search.best_params_)
        return search
    
    def _align_proba_to_labels(self, y_proba_enc, target_labels):
        """Allinea le probabilità alle etichette target."""
        target_labels = np.asarray(target_labels)
        idx = [np.where(self.le.classes_ == c)[0][0] for c in target_labels]
        return y_proba_enc[:, idx]
    
    def fit(self, X_train, y_train, feature_names=None, all_classes=None):
        """
        Esegue il training e tuning di tutti i modelli.
        
        Args:
            X_train: Feature di training
            y_train: Target di training
            feature_names: Nomi delle feature (opzionale, per importance)
            all_classes: Array di TUTTE le classi possibili del problema
                        Es. per PHM: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                        IMPORTANTE: deve includere anche classi non presenti nel training
        """
        print("=" * 60)
        print("TRAINING MODELLI (Classificazione)")
        print("=" * 60)
        
        # CONFIGURAZIONE LABEL ENCODER CON TUTTE LE CLASSI
        if all_classes is not None:
            # Assicurati che sia un array numpy ordinato
            self.all_classes = np.array(sorted(set(all_classes)))
            
            # Configura il LabelEncoder per conoscere TUTTE le classi possibili
            self.le.fit(self.all_classes)
            
            print(f"📊 CONFIGURAZIONE CLASSI:")
            print(f"   - Totale classi nel problema: {len(self.all_classes)}")
            print(f"   - Classi possibili: {list(self.all_classes)}")
            
            # Verifica quali classi sono effettivamente nel training
            unique_train_classes = sorted(np.unique(y_train))
            print(f"   - Classi nel training set: {unique_train_classes}")
            
            # Identifica classi mancanti nel training
            missing_in_training = set(self.all_classes) - set(unique_train_classes)
            if missing_in_training:
                print(f"   ⚠️  CLASSI ASSENTI NEL TRAINING: {sorted(missing_in_training)}")
                print(f"      (I modelli non potranno predire queste classi direttamente)")
            
            # Trasforma i target del training
            y_tr_enc = self.le.transform(y_train)
            
        else:
            # Modalità standard: usa solo le classi presenti nel training
            print("⚠️  ATTENZIONE: all_classes non specificato.")
            print("   I modelli conosceranno solo le classi presenti nel training.")
            
            y_tr_enc = self.le.fit_transform(y_train)
            self.all_classes = self.le.classes_
            print(f"   Classi trovate: {list(self.le.classes_)}")
        
        # Numero totale di classi per configurare i modelli
        n_classes = len(self.le.classes_)
        print(f"\n📈 Configurazione modelli per {n_classes} classi totali")
        
        # IMPORTANTE: Configura XGBoost con il numero corretto di classi
        self.xgb.named_steps["model"].num_class = n_classes
        
        # Per alcuni modelli potrebbe essere necessario configurare parametri aggiuntivi
        # per gestire classi sbilanciate o mancanti
        
        # Salva dati per uso successivo
        self.X_train = X_train
        self.y_train = y_train
        self.y_tr_enc = y_tr_enc
        self.feature_names = feature_names
        
        # Configurazione modelli da addestrare
        models_config = [
            ("RandomForest", self.rf, self.rf_param_dist, 25),
            ("HistGradientBoosting", self.hgb, self.hgb_param_dist, 25),
            ("XGBoost", self.xgb, self.xgb_param_dist, 30),
            ("LightGBM", self.lgbm, self.lgbm_param_dist, 30),
            ("GradientBoosting", self.gb, self.gb_param_dist, 25),
            ("ExtraTrees", self.et, self.et_param_dist, 25),
            ("RidgeClassifier", self.ridge, self.ridge_param_dist, 10),
            ("LogReg_ElasticNet", self.elastic, self.elastic_param_dist, 15),
        ]
        
        print(f"\n🚀 Avvio training di {len(models_config)} modelli...")
        print("-" * 60)
        
        # Training di tutti i modelli
        for name, model, param_dist, n_iter in models_config:
            try:
                self.searches[name] = self._fit_search(
                    name, model, param_dist, X_train, y_tr_enc, n_iter
                )
            except Exception as e:
                print(f"❌ Errore nel training di {name}: {str(e)}")
                continue
        
        # Stampa risultati finali della cross-validation
        self._print_cv_results()
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETATO")
        print("=" * 60)
    
    def _print_cv_results(self):
        """Stampa i risultati della cross-validation."""
        print("\n" + "=" * 60)
        print("RISULTATI FINALI - CONFRONTO MODELLI (Cross-Validation)")
        print("=" * 60)
        print(f"{'Modello':<20} {'Best CV ' + self.scoring.upper():<18} {'Std':<10}")
        print("-" * 55)

        results = []
        for name, search in self.searches.items():
            cv_scores = cross_val_score(
                search.best_estimator_, self.X_train, self.y_tr_enc,
                cv=self.cv, scoring=self.scoring, n_jobs=-1
            )
            score_mean = cv_scores.mean()
            score_std = cv_scores.std()
            results.append((name, score_mean, score_std, search.best_estimator_))
            print(f"{name:<20} {score_mean:<18.4f} {score_std:<10.4f}")

        # Ordina per performance
        results.sort(key=lambda x: -x[1])
        print(f"\n🏆 Miglior modello (CV): {results[0][0]} ({self.scoring.upper()}: {results[0][1]:.4f} ± {results[0][2]:.4f})")
        
        # Salva il miglior modello da CV
        self.best_model = results[0][3]
        self.cv_results = results
    
    def evaluate(self, model, X_val, y_val, model_name, print_cm=True, print_report=True):
        """Valuta un modello su validation set."""
        # Predizione
        y_pred_enc = model.predict(X_val)
        y_pred = self.le.inverse_transform(np.asarray(y_pred_enc).ravel())

        # Etichette reali presenti in validation
        labels_val = np.array(sorted(np.unique(y_val)))

        # Probabilità allineate
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba_enc = model.predict_proba(X_val)
                y_proba = self._align_proba_to_labels(y_proba_enc, labels_val)
            except Exception:
                y_proba = None

        # Metriche
        acc = accuracy_score(y_val, y_pred)
        bacc = balanced_accuracy_score(y_val, y_pred)
        f1m = f1_score(y_val, y_pred, average="macro")
        f1w = f1_score(y_val, y_pred, average="weighted")

        auc = None
        if y_proba is not None and len(labels_val) > 1:
            try:
                auc = roc_auc_score(y_val, y_proba, multi_class="ovr", labels=labels_val)
            except Exception:
                auc = None

        # Confusion matrix
        print(np.unique(y_val, return_counts=True))
        print(np.unique(y_pred, return_counts=True))
        print(np.unique(labels_val, return_counts=True))
        
        cm = confusion_matrix(y_val, y_pred, labels=labels_val)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels_val],
                           columns=[f"pred_{l}" for l in labels_val])

        print(f"\n=== {model_name} on validation ===")
        print(f"Accuracy:            {acc:.4f}")
        print(f"Balanced Accuracy:   {bacc:.4f}")
        print(f"F1 Macro:            {f1m:.4f}")
        print(f"F1 Weighted:         {f1w:.4f}")
        if auc is not None:
            print(f"ROC AUC:             {auc:.4f}")

        if print_report:
            print("\nClassification report:")
            print(classification_report(y_val, y_pred, digits=4, labels=labels_val))

        if print_cm:
            print("\nConfusion matrix:")
            print(cm_df)

        return {
            "name": model_name,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1_macro": f1m,
            "f1_weighted": f1w,
            "auc": auc,
            "y_pred": y_pred,
            "cm_df": cm_df
        }
    
    def evaluate_all(self, X_val, y_val):
        """
        Valuta tutti i modelli su validation set e mostra le confusion matrix.
        """
        self._last_X_val = X_val
        self._last_y_val = y_val

        print("\n" + "="*60)
        print("VALUTAZIONE SU VALIDATION - TUTTI I MODELLI")
        print("="*60)

        self.val_results = []
        self.conf_matrices = {}
        
        # Prima di tutto, capiamo quali classi ci sono
        val_classes = sorted(np.unique(y_val))
        print(f"\n📊 Classi nel validation set: {val_classes}")
        print(f"   Numero totale campioni: {len(y_val)}")
        
        # Per ogni modello
        for name, search in self.searches.items():
            print(f"\n{'='*40}")
            print(f"🔍 {name}")
            print(f"{'='*40}")
            
            est = search.best_estimator_
            
            if hasattr(X_val, 'columns'):
                X_val_array = X_val.values
            else:
                X_val_array = X_val
            
            # Valutazione metriche standard
            res = self.evaluate(est, X_val_array, y_val, name, print_cm=False, print_report=False)
            self.val_results.append(res)
            
            # CORREZIONE: usa y_pred dal risultato di evaluate
            y_pred = res['y_pred']  # Usa questo invece di ripredire
            
            # GESTIONE INTELLIGENTE DELLE CLASSI
            pred_classes = sorted(np.unique(y_pred))
            
            # Verifica quali classi il modello conosce
            if hasattr(est, 'classes_'):
                model_classes = list(est.classes_)
                print(f"   Classi che il modello conosce: {model_classes}")
            else:
                model_classes = pred_classes
                
            print(f"   Classi effettivamente predette: {pred_classes}")
            
            # Identifica problemi
            missing_in_pred = set(val_classes) - set(pred_classes)
            if missing_in_pred:
                print(f"   ⚠️  Classi MAI predette: {sorted(missing_in_pred)}")
            
            # CORREZIONE PRINCIPALE: usa solo le classi del validation
            # Non includere classi spurie che potrebbero apparire nelle predizioni
            labels_for_matrix = val_classes  # USA SOLO LE CLASSI DEL VALIDATION
            
            # Calcola confusion matrix con le classi del validation
            from sklearn.metrics import confusion_matrix
            cm_raw = confusion_matrix(y_val, y_pred, labels=labels_for_matrix)
            
            # Normalizzazione per riga (percentuali per classe vera)
            with np.errstate(invalid='ignore', divide='ignore'):
                cm_norm = cm_raw.astype(float).copy()
                row_sums = cm_norm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cm_norm = cm_norm / row_sums
            
            # Salva per uso futuro
            self.conf_matrices[name] = {
                "labels": labels_for_matrix,  # Usa labels_for_matrix invece di all_classes
                "raw": cm_raw,
                "normalized": cm_norm
            }
            
            # ===== STAMPA BELLA CONFUSION MATRIX =====
            
            # 1. Matrice RAW (conteggi)
            print(f"\n   📋 Confusion Matrix (conteggi)")
            print(f"   " + "-"*50)
            
            # Header
            header = "   TRUE\\PRED"
            for cls in labels_for_matrix:
                header += f"{cls:>6}"
            print(header)
            print(f"   " + "-"*50)
            
            # Righe
            for i, true_cls in enumerate(labels_for_matrix):
                row = f"   {true_cls:>9}"
                for j, pred_cls in enumerate(labels_for_matrix):
                    count = cm_raw[i, j]
                    if count == 0:
                        row += f"{'·':>6}"
                    else:
                        row += f"{count:>6}"
                print(row)
            
            # 2. Matrice NORMALIZZATA (percentuali)
            print(f"\n   📊 Confusion Matrix (percentuali per riga)")
            print(f"   " + "-"*50)
            
            # Header
            header = "   TRUE\\PRED"
            for cls in labels_for_matrix:
                header += f"{cls:>7}"
            print(header)
            print(f"   " + "-"*50)
            
            # Righe
            for i, true_cls in enumerate(labels_for_matrix):
                row = f"   {true_cls:>9}"
                for j, pred_cls in enumerate(labels_for_matrix):
                    val = cm_norm[i, j]
                    if val == 0:
                        row += f"{'·':>7}"
                    else:
                        row += f"{val:>6.1%} "
                print(row)
            
            # 3. Analisi degli errori principali
            print(f"\n   🎯 Analisi errori principali:")
            errors = []
            for i, true_cls in enumerate(labels_for_matrix):
                for j, pred_cls in enumerate(labels_for_matrix):
                    if i != j and cm_raw[i, j] > 0:
                        errors.append((true_cls, pred_cls, cm_raw[i, j], cm_norm[i, j]))
            
            # Ordina per numero di errori
            errors.sort(key=lambda x: x[2], reverse=True)
            
            # Mostra top 5 errori
            for k, (true_cls, pred_cls, count, perc) in enumerate(errors[:5]):
                print(f"      {k+1}. Classe {true_cls} → {pred_cls}: {count} errori ({perc:.1%})")
            
            # 4. Performance per classe
            print(f"\n   📈 Performance per classe:")
            for i, cls in enumerate(labels_for_matrix):
                total = cm_raw[i].sum()
                if total > 0:
                    correct = cm_raw[i, i]
                    accuracy = correct / total
                    
                    # Calcola quanti sono stati predetti come questa classe
                    predicted_as_this = cm_raw[:, i].sum()
                    
                    # Emoji in base alla performance
                    if accuracy >= 0.95:
                        emoji = "✅"
                    elif accuracy >= 0.8:
                        emoji = "🟡"
                    else:
                        emoji = "❌"
                        
                    print(f"      {emoji} Classe {cls}: {correct}/{total} corretti ({accuracy:.1%}) - {predicted_as_this} predetti come {cls}")
                else:
                    print(f"      ⚪ Classe {cls}: nessun campione nel validation")
            
        # ===== ANALISI COMPARATIVA FINALE =====
        print(f"\n{'='*60}")
        print("📊 ANALISI COMPARATIVA")
        print(f"{'='*60}")
        
        self._analyze_results()
        
        # Trova il miglior modello  
        best_idx = np.argmax([r['accuracy'] for r in self.val_results])
        best_model = self.val_results[best_idx]['name']  # CORREZIONE: usa 'name' non 'model'
        best_acc = self.val_results[best_idx]['accuracy']
        
        print(f"\n🏆 MIGLIOR MODELLO: {best_model}")
        print(f"   Accuracy: {best_acc:.4f}")
        
        # Mostra ranking
        print(f"\n📊 RANKING MODELLI:")
        sorted_results = sorted(self.val_results, key=lambda x: x['accuracy'], reverse=True)
        for i, res in enumerate(sorted_results, 1):
            if i == 1:
                medal = "🥇"
            elif i == 2:
                medal = "🥈"
            elif i == 3:
                medal = "🥉"
            else:
                medal = f"{i}."
                
            print(f"   {medal} {res['name']:<25} Acc: {res['accuracy']:.4f}  F1: {res['f1_macro']:.4f}")
        
        return self.val_results
    
    def _analyze_results(self):
        """Analizza e confronta i risultati di CV e validation."""
        # Crea DataFrame per confronto
        cv_data = {
            'Model': [name for name, _, _, _ in self.cv_results],
            'CV_Score': [score for _, score, _, _ in self.cv_results],
            'CV_Std': [std for _, _, std, _ in self.cv_results]
        }
        
        val_data = {
            'Model': [r['name'] for r in self.val_results],
            'Val_Accuracy': [r['accuracy'] for r in self.val_results],
            'Val_F1_Macro': [r['f1_macro'] for r in self.val_results],
            'Val_AUC': [r['auc'] if r['auc'] is not None else np.nan for r in self.val_results]
        }

        cv_df = pd.DataFrame(cv_data)
        val_df = pd.DataFrame(val_data)
        combined = pd.merge(cv_df, val_df, on='Model', how='inner')

        # Ranking
        combined['CV_Rank'] = combined['CV_Score'].rank(ascending=False)
        combined['Val_Rank'] = combined['Val_Accuracy'].rank(ascending=False)
        combined['Avg_Rank'] = (combined['CV_Rank'] + combined['Val_Rank']) / 2

        combined = combined.sort_values('Avg_Rank').reset_index(drop=True)
        self.combined_results = combined

        print("\nANALISI COMPARATIVA CV vs VALIDATION")
        print("=" * 60)
        print(combined[['Model', 'CV_Score', 'CV_Std', 'Val_Accuracy', 'Val_F1_Macro', 
                       'CV_Rank', 'Val_Rank', 'Avg_Rank']].round(4))

        # Selezione miglior modello
        best_model_row = combined.iloc[0]
        self.best_model_name = best_model_row['Model']
        self.best_model = self.searches[self.best_model_name].best_estimator_
        
        self._print_recommendations()
    
    def _print_recommendations(self):
        """Stampa raccomandazioni e analisi finale."""
        print("\n" + "=" * 60)
        print("RACCOMANDAZIONI")
        print("=" * 60)

        # Top 3 modelli
        top_models = self.combined_results.head(3)
        print("🏆 TOP 3 MODELLI (ranking medio CV + Validation):")
        for idx, row in top_models.iterrows():
            print(f"{idx+1}. {row['Model']:<20} | CV: {row['CV_Score']:.4f}±{row['CV_Std']:.4f} | Val: {row['Val_Accuracy']:.4f}")

        print(f"\n🎯 MODELLO RACCOMANDATO: {self.best_model_name}")

        # Analisi stabilità
        print(f"\n📊 ANALISI STABILITÀ:")
        stable_models = self.combined_results[self.combined_results['CV_Std'] < 0.012]
        print("Modelli con bassa variabilità in CV (std < 0.012):")
        for _, row in stable_models.iterrows():
            consistency = abs(row['CV_Score'] - row['Val_Accuracy'])
            print(f"  • {row['Model']:<20} | Consistency: {consistency:.4f}")

        # Check overfitting
        print(f"\n🔍 CHECK OVERFITTING:")
        for _, row in self.combined_results.iterrows():
            diff = row['Val_Accuracy'] - row['CV_Score']
            if diff > 0.02:
                status = "⚠️  Possibile validation set favorevole"
            elif diff < -0.02:
                status = "❌ Possibile overfitting"
            else:
                status = "✅ Consistente"
            print(f"  • {row['Model']:<20} | Diff: {diff:+.4f} | {status}")
    
    def feature_importance(self, X_val, y_val, n_repeats=20, top_k=25):
        """Calcola permutation importance."""
        if self.best_model is None:
            raise ValueError("Esegui prima fit() e evaluate_all()")
        
        print(f"\n[*] Calcolo permutation importance sul modello {self.best_model_name}...")
        
        y_val_enc = self.le.transform(y_val)
        
        perm = permutation_importance(
            self.best_model, X_val, y_val_enc,
            scoring=self.scoring, n_repeats=n_repeats,
            random_state=self.random_state, n_jobs=-1
        )

        if self.feature_names is not None:
            importances = pd.Series(perm.importances_mean, index=self.feature_names).sort_values(ascending=False)
        else:
            importances = pd.Series(perm.importances_mean).sort_values(ascending=False)

        print(f"\nTop {top_k} feature (permutation importance - {self.scoring}):")
        print(importances.head(top_k))
        
        self.feature_importances = importances
        return importances
    
    def create_ensemble(self, top_n=3, voting='soft'):
        """Crea un ensemble dei migliori modelli."""
        if not self.combined_results.empty:
            top_models_names = self.combined_results.head(top_n)['Model'].tolist()
            
            estimators = [(name.lower(), self.searches[name].best_estimator_) 
                         for name in top_models_names]
            
            ensemble = VotingClassifier(estimators=estimators, voting=voting)
            
            print(f"\n[*] Creato ensemble con {voting} voting dei top {top_n} modelli:")
            for name in top_models_names:
                print(f"  • {name}")
            
            return ensemble
        else:
            raise ValueError("Esegui prima evaluate_all()")

    def save_model(self, model=None, model_name=None):
        """Salva solo il modello con joblib."""
        # 1) Sorgente del modello e del nome
        if model is None:
            model = getattr(self, "best_model", None)
            model_name = getattr(self, "best_model_name", None)

        if model is None:
            raise ValueError("Nessun modello da salvare. Esegui prima fit()")

        if model_name is None:
            model_name = type(model).__name__

        # 2) Cartella e timestamp
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 3) Salvataggio modello
        model_filename = f"models/best_model_classification_{model_name}_{timestamp}.joblib"
        joblib.dump(model, model_filename)
        print(f"Modello salvato con joblib in: {model_filename}")

        return model_filename

    def load_model(self, model_path, metadata_path=None):
        """Carica un modello salvato."""
        model = joblib.load(model_path)
        metadata = None
        
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Ripristina il label encoder
            if 'classes' in metadata:
                self.le.classes_ = np.array(metadata['classes'])
        
        print(f"Modello caricato da: {model_path}")
        return model, metadata
    
    def predict(self, X, return_proba=False):
        """Esegue predizioni con il modello migliore."""
        if self.best_model is None:
            raise ValueError("Nessun modello disponibile. Esegui prima fit() e evaluate_all()")
        
        y_pred_enc = self.best_model.predict(X)
        y_pred = self.le.inverse_transform(y_pred_enc)
        
        if return_proba and hasattr(self.best_model, "predict_proba"):
            y_proba_enc = self.best_model.predict_proba(X)
            return y_pred, y_proba_enc
        
        return y_pred
    
    def get_summary(self):
        """Restituisce un summary dei risultati."""
        if not self.val_results:
            raise ValueError("Esegui prima evaluate_all()")
        
        return {
            'best_model_name': self.best_model_name,
            'best_model': self.best_model,
            'cv_results': self.cv_results,
            'validation_results': self.val_results,
            'combined_results': self.combined_results,
            'label_encoder': self.le
        }
    # Funzione helper per visualizzare le metriche dell'ensemble
    def evaluate_ensemble(ensemble, X_va, y_va, plot=True):
        """
        Valuta l'ensemble e visualizza le metriche, gestendo anche classi non viste nel training
        """
        # Predizioni
        y_pred = ensemble.predict(X_va)
        
        # Report testuale
        print("\n" + "="*60)
        print("PERFORMANCE ENSEMBLE - TOP 3 MODELLI")
        print("="*60)
        
        # Ottieni tutte le classi uniche (sia true che predette)
        all_classes = sorted(np.unique(np.concatenate([y_va, y_pred])))
        classes_in_true = sorted(np.unique(y_va))
        classes_in_pred = sorted(np.unique(y_pred))
        
        # Classi predette ma non nel validation set
        novel_predicted = set(classes_in_pred) - set(classes_in_true)
        # Classi nel validation ma non predette
        missed_classes = set(classes_in_true) - set(classes_in_pred)
        
        print(f"\nClassi nel validation set: {classes_in_true}")
        print(f"Classi predette dal modello: {classes_in_pred}")
        
        if novel_predicted:
            print(f"⚠️  CLASSI NUOVE PREDETTE (non nel validation): {sorted(novel_predicted)}")
        if missed_classes:
            print(f"⚠️  CLASSI MAI PREDETTE: {sorted(missed_classes)}")
        
        # Classification report con labels specificati
        try:
            report = classification_report(y_va, y_pred, 
                                        labels=all_classes, 
                                        output_dict=True, 
                                        digits=3,
                                        zero_division=0)
            
            # Stampa il report standard
            print("\n" + classification_report(y_va, y_pred, 
                                            labels=all_classes,
                                            digits=3,
                                            zero_division=0))
        except:
            # Fallback se ci sono problemi
            report = classification_report(y_va, y_pred, 
                                        output_dict=True, 
                                        digits=3,
                                        zero_division=0)
            print("\n" + classification_report(y_va, y_pred, digits=3, zero_division=0))
        
        # Estrai metriche aggregate
        accuracy = accuracy_score(y_va, y_pred)
        weighted_avg = report.get('weighted avg', {})
        macro_avg = report.get('macro avg', {})
        
        print("\n" + "="*60)
        print("SUMMARY METRICHE:")
        print("="*60)
        print(f"Accuracy:              {accuracy:.3f}")
        if weighted_avg:
            print(f"Weighted Precision:    {weighted_avg.get('precision', 0):.3f}")
            print(f"Weighted Recall:       {weighted_avg.get('recall', 0):.3f}")
            print(f"Weighted F1-Score:     {weighted_avg.get('f1-score', 0):.3f}")
        if macro_avg:
            print(f"Macro F1-Score:        {macro_avg.get('f1-score', 0):.3f}")
        
        if plot:
            # Crea figura con subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # 1. Confusion Matrix
            cm = confusion_matrix(y_va, y_pred, labels=all_classes)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=all_classes, yticklabels=all_classes)
            axes[0].set_title('Confusion Matrix - Ensemble')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('True')
            
            # Evidenzia classi anomale
            if novel_predicted:
                for cls in novel_predicted:
                    if cls in all_classes:
                        idx = all_classes.index(cls)
                        axes[0].axvline(x=idx+0.5, color='red', linestyle='--', alpha=0.5)
                        axes[0].text(idx+0.5, -0.5, 'NEW', ha='center', color='red', fontweight='bold')
            
            # 2. Bar plot delle metriche per classe
            precisions = []
            recalls = []
            f1_scores = []
            bar_labels = []
            bar_colors = []
            
            for c in all_classes:
                class_key = str(c)
                if class_key in report and isinstance(report[class_key], dict):
                    precisions.append(report[class_key].get('precision', 0))
                    recalls.append(report[class_key].get('recall', 0))
                    f1_scores.append(report[class_key].get('f1-score', 0))
                    
                    # Colora diversamente le classi speciali
                    if c in novel_predicted:
                        bar_colors.append('red')
                        bar_labels.append(f'{c}*')
                    elif c in missed_classes:
                        bar_colors.append('orange')
                        bar_labels.append(f'{c}!')
                    else:
                        bar_colors.append('blue')
                        bar_labels.append(str(c))
                else:
                    # Classe non nel report (probabilmente novel)
                    precisions.append(0)
                    recalls.append(0)
                    f1_scores.append(0)
                    bar_colors.append('gray')
                    bar_labels.append(f'{c}?')
            
            x = np.arange(len(all_classes))
            width = 0.25
            
            bars1 = axes[1].bar(x - width, precisions, width, label='Precision', alpha=0.8)
            bars2 = axes[1].bar(x, recalls, width, label='Recall', alpha=0.8)
            bars3 = axes[1].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
            
            axes[1].set_xlabel('Classes')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Metriche per Classe')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(bar_labels)
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].set_ylim([0, 1.1])
            
            # 3. Distribuzione delle predizioni
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            true_counts = pd.Series(y_va).value_counts().sort_index()
            
            # Assicurati che tutte le classi siano rappresentate
            for c in all_classes:
                if c not in pred_counts.index:
                    pred_counts[c] = 0
                if c not in true_counts.index:
                    true_counts[c] = 0
            
            pred_counts = pred_counts.sort_index()
            true_counts = true_counts.sort_index()
            
            x_dist = np.arange(len(all_classes))
            width_dist = 0.35
            
            axes[2].bar(x_dist - width_dist/2, true_counts.values, width_dist, 
                    label='True Distribution', alpha=0.7, color='green')
            axes[2].bar(x_dist + width_dist/2, pred_counts.values, width_dist, 
                    label='Predicted Distribution', alpha=0.7, color='orange')
            
            axes[2].set_xlabel('Classes')
            axes[2].set_ylabel('Count')
            axes[2].set_title('Distribuzione Classi: True vs Predicted')
            axes[2].set_xticks(x_dist)
            axes[2].set_xticklabels(all_classes)
            axes[2].legend()
            axes[2].grid(axis='y', alpha=0.3)
            
            # Aggiungi annotazioni per classi anomale
            for i, c in enumerate(all_classes):
                if c in novel_predicted:
                    axes[2].text(i, pred_counts.iloc[i] + 1, 'NEW', 
                            ha='center', color='red', fontweight='bold', fontsize=8)
            
            plt.tight_layout()
            plt.show()
        
        return report

    def plot_confusion_matrices(self, figsize=(20, 12), cmap='Blues', normalize=False):
        """
        Visualizza graficamente tutte le confusion matrix dei modelli.
        """
        if not hasattr(self, 'conf_matrices'):
            raise ValueError("Esegui prima evaluate_all() per generare le confusion matrix")
        
        # Calcola layout ottimale
        n_models = len(self.conf_matrices)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # Crea figura
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Assicurati che axes sia sempre un array
        if n_models == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot per ogni modello
        for idx, (model_name, cm_data) in enumerate(self.conf_matrices.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Estrai dati
            labels = cm_data['labels']
            cm_to_plot = cm_data['normalized'] if normalize else cm_data['raw']
            
            # Crea heatmap
            sns.heatmap(
                cm_to_plot,
                annot=True,
                fmt='.1%' if normalize else 'd',
                cmap=cmap,
                square=True,
                cbar=False,
                vmin=0,
                vmax=1 if normalize else None,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
            
            # CORREZIONE: usa 'name' invece di 'model'
            model_accuracy = next(
                (r['accuracy'] for r in self.val_results if r['name'] == model_name),  # <-- USA 'name'
                0
            )
            
            # Titolo con nome modello e accuracy
            ax.set_title(
                f'{model_name}\nAccuracy: {model_accuracy:.3f}',
                fontsize=11,
                fontweight='bold',
                pad=10
            )
            
            # Label degli assi
            ax.set_xlabel('Predicted Class', fontsize=9)
            ax.set_ylabel('True Class', fontsize=9)
            
            # Ruota le label per leggibilità
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            
            # Aggiungi griglia per separare meglio le celle
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Rimuovi subplot vuoti
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            fig.delaxes(ax)
        
        # Aggiungi una colorbar comune
        if n_models > 0:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            
            norm = Normalize(vmin=0, vmax=1 if normalize else cm_data['raw'].max())
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            
            cbar = fig.colorbar(
                sm,
                ax=axes.ravel().tolist(),
                orientation='vertical',
                fraction=0.02,
                pad=0.02
            )
            cbar.set_label(
                'Accuracy per Class' if normalize else 'Number of Samples',
                rotation=270,
                labelpad=20,
                fontsize=10
            )
        
        # Titolo generale
        title_text = 'Confusion Matrices - All Models'
        if normalize:
            title_text += ' (Normalized)'
        
        plt.suptitle(
            title_text,
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        
        plt.tight_layout(rect=[0, 0, 0.96, 0.96])
        plt.show()

    def plot_final_comparison(self, figsize=(16, 10)):
        """
        Crea un plot comparativo completo dei risultati finali.
        """
        if not hasattr(self, 'combined_results'):
            raise ValueError("Esegui prima evaluate_all()")
        
        fig = plt.figure(figsize=figsize)
        
        # Layout: 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Comparison CV vs Validation Accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        models = self.combined_results['Model'].values
        cv_scores = self.combined_results['CV_Score'].values
        val_scores = self.combined_results['Val_Accuracy'].values
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cv_scores, width, label='CV Score', 
                        color='skyblue', edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, val_scores, width, label='Validation Score',
                        color='lightcoral', edgecolor='black', linewidth=1)
        
        # Aggiungi valori sopra le barre
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Models', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Cross-Validation vs Validation Accuracy', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # 2. F1 Scores Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        f1_macro = self.combined_results['Val_F1_Macro'].values
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax2.barh(models, f1_macro, color=colors, edgecolor='black', linewidth=1)
        
        # Aggiungi valori
        for i, (bar, score) in enumerate(zip(bars, f1_macro)):
            ax2.text(score + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=9)
        
        ax2.set_xlabel('F1 Macro Score', fontweight='bold')
        ax2.set_title('F1 Macro Score by Model', fontweight='bold', fontsize=12)
        ax2.set_xlim([0, 1.05])
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Ranking Plot
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Crea scatter plot per i ranking
        cv_rank = self.combined_results['CV_Rank'].values
        val_rank = self.combined_results['Val_Rank'].values
        
        # Colora in base al ranking medio
        avg_rank = self.combined_results['Avg_Rank'].values
        scatter = ax3.scatter(cv_rank, val_rank, s=200, c=avg_rank, 
                            cmap='RdYlGn_r', edgecolor='black', linewidth=2,
                            alpha=0.7)
        
        # Aggiungi etichette
        for i, model in enumerate(models):
            ax3.annotate(model, (cv_rank[i], val_rank[i]), 
                        fontsize=8, ha='center', va='center')
        
        # Aggiungi linea diagonale
        ax3.plot([0, 9], [0, 9], 'k--', alpha=0.3, label='Perfect Agreement')
        
        ax3.set_xlabel('CV Rank', fontweight='bold')
        ax3.set_ylabel('Validation Rank', fontweight='bold')
        ax3.set_title('Model Rankings: CV vs Validation', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Average Rank', rotation=270, labelpad=15)
        
        # 4. Performance Stability
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calcola differenza tra CV e Validation
        diff = val_scores - cv_scores
        
        # Colora in base alla differenza
        colors_diff = ['green' if abs(d) < 0.02 else 'orange' if d > 0 else 'red' 
                    for d in diff]
        
        bars = ax4.bar(models, diff, color=colors_diff, edgecolor='black', linewidth=1)
        
        # Aggiungi valori
        for bar, d in zip(bars, diff):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{d:+.3f}', ha='center', va='bottom' if d > 0 else 'top',
                    fontsize=8)
        
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Models', fontweight='bold')
        ax4.set_ylabel('Difference (Val - CV)', fontweight='bold')
        ax4.set_title('Model Stability (Validation - CV Score)', fontweight='bold', fontsize=12)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        # Aggiungi zone colorate
        ax4.axhspan(-0.02, 0.02, alpha=0.2, color='green', label='Stable (±0.02)')
        ax4.axhspan(0.02, 0.1, alpha=0.2, color='orange', label='Possible Overfit')
        ax4.axhspan(-0.1, -0.02, alpha=0.2, color='red', label='Underfit')
        ax4.legend(loc='upper right', fontsize=8)
        
        # Titolo generale
        fig.suptitle('Model Performance Comparison - Complete Analysis', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
        
        # Stampa best model
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        best_model = self.combined_results.iloc[0]
        print(f"🏆 Best Model: {best_model['Model']}")
        print(f"   - CV Score: {best_model['CV_Score']:.4f} ± {best_model['CV_Std']:.4f}")
        print(f"   - Validation Accuracy: {best_model['Val_Accuracy']:.4f}")
        print(f"   - F1 Macro: {best_model['Val_F1_Macro']:.4f}")