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
    
    def fit(self, X_train, y_train, feature_names=None):
        """
        Esegue il training e tuning di tutti i modelli.
        
        Args:
            X_train: Feature di training
            y_train: Target di training
            feature_names: Nomi delle feature (per importance)
        """
        print("=" * 60)
        print("TRAINING MODELLI (Classificazione)")
        print("=" * 60)
        
        # Encoding delle etichette
        y_tr_enc = self.le.fit_transform(y_train)
        n_classes = len(self.le.classes_)
        
        # Aggiorna XGB con num_class corretto
        self.xgb.named_steps["model"].num_class = n_classes
        
        # Salva per uso successivo
        self.X_train = X_train
        self.y_train = y_train
        self.y_tr_enc = y_tr_enc
        self.feature_names = feature_names
        
        # Training di tutti i modelli
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
        
        for name, model, param_dist, n_iter in models_config:
            self.searches[name] = self._fit_search(name, model, param_dist, X_train, y_tr_enc, n_iter)
        
        # Confronto finale cross-validation
        self._print_cv_results()
    
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
        """Valuta tutti i modelli su validation set e mostra le confusion matrix."""
        self._last_X_val = X_val
        self._last_y_val = y_val

        print("\n" + "=" * 60)
        print("VALUTAZIONE SU VALIDATION - TUTTI I MODELLI")
        print("=" * 60)

        self.val_results = []
        self.conf_matrices = {}  

        # Ordine/etichette di classe coerenti
        if hasattr(self, "classes_") and self.classes_ is not None:
            classes = list(self.classes_)
        else:
            # Se non definito altrove, inferisco dalle y di validation
            classes = list(np.unique(y_val))

        for name, search in self.searches.items():
            est = search.best_estimator_

            
            res = self.evaluate(est, X_val, y_val, name, print_cm=False, print_report=False)
            self.val_results.append(res)

            try:
                y_pred = est.predict(X_val)
            except Exception as e:
                print(f"[{name}] Impossibile calcolare predizioni per CM: {e}")
                continue

            cm_raw = confusion_matrix(y_val, y_pred, labels=classes)
            # Normalizzazione 'true': per riga (somma riga = 1); gestisco righe nulle
            with np.errstate(invalid='ignore'):
                cm_norm = cm_raw.astype(float)
                row_sums = cm_norm.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm, dtype=float), where=row_sums!=0)

            self.conf_matrices[name] = {
                "labels": classes,
                "raw": cm_raw,
                "normalized": cm_norm
            }

            # Stampa tabellare compatta
            print(f"\n— Matrice di confusione (raw) – {name}")
            df_raw = pd.DataFrame(cm_raw, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
            print(df_raw)

            print(f"\n— Matrice di confusione (normalizzata per riga) – {name}")
            df_norm = pd.DataFrame(cm_norm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
            # 3 decimali per leggibilità
            print(df_norm.round(3))

            '''try:
                import matplotlib.pyplot as plt
                fig1 = ConfusionMatrixDisplay(cm_raw, display_labels=classes)
                fig1.plot(values_format=".0f", xticks_rotation=45, colorbar=False)
                plt.title(f"Confusion Matrix (raw) – {name}")
                plt.tight_layout()
                plt.show()

                fig2 = ConfusionMatrixDisplay(cm_norm, display_labels=classes)
                fig2.plot(values_format=".2f", xticks_rotation=45, colorbar=False)
                plt.title(f"Confusion Matrix (norm) – {name}")
                plt.tight_layout()
                plt.show()
            except Exception as _:
                pass'''

        self._analyze_results()
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
        """Salva il modello (anche ensemble) con metadati robusti."""
        # 1) Sorgente del modello e del nome
        if model is None:
            model = self.best_model
            model_name = self.best_model_name

        if model is None:
            raise ValueError("Nessun modello da salvare. Esegui prima fit()")

        # Se non è stato passato model_name, prova a costruirlo
        if model_name is None:
            try:
                # es. Pipeline(...) o VotingClassifier(...)
                model_name = type(model).__name__
            except Exception:
                model_name = "UnknownModel"

        # Riconosco se è un ensemble (VotingClassifier)
        is_ensemble = isinstance(model, VotingClassifier)

        # 2) Cartella e timestamp
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 3) Salvataggio del modello (joblib gestisce anche VotingClassifier)
        model_filename = f'models/best_model_classification_{model_name}_{timestamp}.joblib'
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

    
