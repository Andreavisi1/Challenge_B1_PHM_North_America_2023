from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Sequence
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from numpy.typing import NDArray


class SubmissionGenerator:
    """Generatore di submission semplificato per competizioni ML."""
    
    def __init__(self, rng_seed: int = 42):
        self.rng_seed = rng_seed
        self.detectors = {}
        self.missing_map = {4: 5, 6: 7, 8: (9, 10)}
        
    @staticmethod
    def get_model_classes(model: Any) -> NDArray[np.int_]:
        """Estrae model.classes_ in modo robusto."""
        if hasattr(model, "classes_"):
            return np.asarray(model.classes_)
        if hasattr(model, "named_steps"):
            step = model.named_steps.get("model", list(model.named_steps.values())[-1])
            return np.asarray(step.classes_)
        if hasattr(model, "steps"):
            return np.asarray(model.steps[-1][1].classes_)
        raise AttributeError("Impossibile recuperare classes_")
    
    def expand_to_0_10(self, proba_seen: NDArray, col_labels: NDArray) -> NDArray:
        """Espande probabilità a 11 classi (0-10)."""
        out = np.zeros((proba_seen.shape[0], 11))
        out[:, col_labels.astype(int)] = proba_seen
        return out
    
    def confidence_score(self, probs, anomaly=None, w=(0.5, 0.3, 0.2)):
        # (1) entropia normalizzata
        eps = 1e-12
        H = -(probs * np.log(probs + eps)).sum(1)
        H /= np.log(probs.shape[1])
        H = 1 - H  # high=confident
        # (2) varianza ordinale
        ord_conf = self.ordinal_confidence(probs)
        # (3) anomalia (invertita)
        if anomaly is None:
            anomaly_comp = np.ones_like(H)
        else:
            a = np.clip(anomaly, 0, 1)
            anomaly_comp = 1 - a
        s = w[0]*H + w[1]*ord_conf + w[2]*anomaly_comp
        return s

    @staticmethod
    def ordinal_confidence(probs: NDArray) -> NDArray:
        """Confidence basata su varianza ordinale."""
        class_values = np.arange(probs.shape[1])
        expected = (probs * class_values).sum(axis=1)
        variance = (probs * (class_values - expected[:, None])**2).sum(axis=1)
        max_var = ((class_values - class_values.mean())**2).mean()
        return 1 - np.sqrt(np.clip(variance / max(max_var, 1e-12), 0, 1))
    
    def needs_model_scaling(self, model: Any) -> bool:
        """Verifica se il modello richiede scaling."""
        scaling_models = ["LogisticRegression", "RidgeClassifier", "SVC"]
        
        # Controlla se è una pipeline
        if hasattr(model, "named_steps"):
            has_scaler = "scaler" in model.named_steps or any("scaler" in name.lower() for name in model.named_steps.keys())
            if has_scaler:
                return False  # Ha già scaler
            model_step = model.named_steps.get("model", list(model.named_steps.values())[-1])
            model_name = model_step.__class__.__name__
        elif hasattr(model, "steps"):
            has_scaler = any("scaler" in name.lower() for name, _ in model.steps[:-1])
            if has_scaler:
                return False
            model_name = model.steps[-1][1].__class__.__name__
        else:
            model_name = model.__class__.__name__
        
        return any(req in model_name for req in scaling_models)
    
    def build_anomaly_detectors(self, X_tr: NDArray, y_tr: NDArray, contamination: float = 0.05) -> None:
        """Costruisce detector per classi 4, 6, 8."""
        scaler = StandardScaler().fit(X_tr)
        X_scaled = scaler.transform(X_tr)
        self.scaler = scaler
        
        for cls in [4, 6, 8]:
            mask = (y_tr == cls)
            if not np.any(mask):
                continue
                
            X_cls = X_scaled[mask]
            if len(X_cls) < 10:  # troppo pochi campioni
                continue
                
            detector = IsolationForest(
                contamination=contamination,
                random_state=self.rng_seed,
                n_jobs=-1
            ).fit(X_cls)
            
            self.detectors[cls] = detector
    
    def apply_transfers(self, probs, X_test, alpha: float = 0.3) -> Dict:
        if not hasattr(self, 'scaler') or not self.detectors:
            return {}
        Xs = self.scaler.transform(X_test)
        transfers = {}
        for cls, det in self.detectors.items():
            # anomalia morbida in [0,1]
            s = det.decision_function(Xs)
            a = 1 / (1 + np.exp((s - np.median(s)) / (np.std(s) + 1e-6)))  # sigmoid centrata
            a = a[:, None]  # broadcast
            if cls in (4, 6):
                dst = self.missing_map[cls]
                moved = alpha * a.squeeze() * probs[:, cls]
                probs[:, cls] -= moved
                probs[:, dst] += moved
                transfers[f"{cls}_to_{dst}"] = int((moved > 0).sum())
            elif cls == 8:
                alpha_eff = (alpha + 0.1) if alpha < 0.9 else alpha
                moved = alpha_eff * a.squeeze() * probs[:, 8]
                probs[:, 8] -= moved
                probs[:, 9] += 0.7 * moved
                probs[:, 10] += 0.3 * moved
                transfers["8_to_9_10"] = int((moved > 0).sum())
        # clamp & renorm di sicurezza
        probs[:] = np.maximum(probs, 0)
        probs[:] = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return transfers
    
    def generate_submission(
        self,
        X_tr: NDArray,
        y_tr: NDArray,
        X_test: NDArray,
        test_ids: Sequence,
        model: Any,
        conf_thresh: float = 0.7,
        contamination: float = 0.05,
        alpha: float = 0.3
    ) -> Tuple[pd.DataFrame, Dict]:
        """Genera submission con gestione anomalie."""
        
        # 1. Prepara dati per il modello
        if self.needs_model_scaling(model):
            model_scaler = StandardScaler().fit(X_tr)
            X_model_input = model_scaler.transform(X_test)
        else:
            X_model_input = X_test
        
        # 2. Predizioni base
        proba_seen = model.predict_proba(X_model_input)
        y_pred = model.predict(X_model_input)
        
        # 3. Mapping a classi reali
        enc_classes = self.get_model_classes(model)
        real_classes = np.sort(np.unique(y_tr))
        
        if set(enc_classes) == set(real_classes):
            col_labels = enc_classes
            y_pred_real = y_pred
        else:
            col_labels = real_classes[enc_classes]
            y_pred_real = real_classes[y_pred]
        
        # 4. Espandi a 11 classi
        probs_11 = self.expand_to_0_10(proba_seen, col_labels)
        
        # 5. Costruisci detector e applica trasferimenti
        self.build_anomaly_detectors(X_tr, y_tr, contamination)
        transfers = self.apply_transfers(probs_11, X_test, alpha)
        
        probs_final = probs_11
        
        # 7. Confidence finale
        confidence = self.confidence_score(probs_final)
        
        # 8. Crea DataFrame
        cols = [f"prob_{i}" for i in range(11)]
        df = pd.DataFrame(probs_final, columns=cols)
        df.insert(0, "id", test_ids)
        df["confidence"] = (confidence > conf_thresh).astype(int)
        
        # 9. Diagnostics
        diagnostics = {
            "detectors_built": list(self.detectors.keys()),
            "transfers": transfers,
            "confidence_stats": {
                "mean": float(confidence.mean()),
                "high_conf_pct": float((confidence > conf_thresh).mean())
            }
        }
        
        return df, diagnostics
    
    def create_submission_simple(
        self,
        X_tr: NDArray,
        y_tr: NDArray, 
        X_test: NDArray,
        test_df: pd.DataFrame,
        model: Any,
        conf_thresh: float = 0.7,
        contamination: float = 0.05,
        alpha: float = 0.3
    ) -> Tuple[pd.DataFrame, Dict]:
        """Wrapper semplificato."""
        return self.generate_submission(
            X_tr, y_tr, X_test, test_df["id"].values, model, conf_thresh, contamination, alpha
        )
