from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class SubmissionGenerator:
    """
    Generatore di submission semplificato per competizioni ML.

    - Espansione delle probabilità alle 11 classi (0..10).
    - Trasferimenti di probabilità per classi 4/6/8 tramite detector (IsolationForest).
    - Confidence SEMPLICE: 1 se max(prob) >= conf_thresh, altrimenti 0.
    """

    def __init__(self, rng_seed: int = 42):
        self.rng_seed = rng_seed
        self.detectors: Dict[int, IsolationForest] = {}
        self.missing_map: Dict[int, int | Tuple[int, int]] = {4: 5, 6: 7, 8: (9, 10)}
        self.scaler: StandardScaler | None = None

    # --------------------------
    # Utility per il modello
    # --------------------------
    @staticmethod
    def get_model_classes(model: Any) -> NDArray[np.int_]:
        """Estrae model.classes_ in modo robusto (anche da pipeline)."""
        if hasattr(model, "classes_"):
            return np.asarray(model.classes_)
        if hasattr(model, "named_steps"):
            # tenta 'model', altrimenti l'ultimo step
            step = model.named_steps.get("model", list(model.named_steps.values())[-1])
            return np.asarray(step.classes_)
        if hasattr(model, "steps"):
            return np.asarray(model.steps[-1][1].classes_)
        raise AttributeError("Impossibile recuperare classes_ dal modello o dalla pipeline.")

    @staticmethod
    def expand_to_0_10(proba_seen: NDArray, col_labels: NDArray) -> NDArray:
        """
        Espande una matrice di probabilità (n_samples x n_classi_viste) a 11 classi (0..10),
        collocando ogni colonna nella posizione coerente con la sua etichetta reale.
        """
        out = np.zeros((proba_seen.shape[0], 11), dtype=float)
        out[:, col_labels.astype(int)] = proba_seen
        # normalizzazione difensiva (eventuali piccoli drift numerici)
        out_sum = out.sum(axis=1, keepdims=True)
        out /= np.maximum(out_sum, 1e-12)
        return out

    def needs_model_scaling(self, model: Any) -> bool:
        """
        Verifica se il modello (o l'ultimo step della pipeline) è tipicamente sensibile allo scaling.
        Se nella pipeline è già presente uno scaler, restituisce False.
        """
        scaling_models = ("LogisticRegression", "RidgeClassifier", "SVC")

        if hasattr(model, "named_steps"):
            # se c'è già uno scaler in pipeline, non serve scalare
            has_scaler = "scaler" in model.named_steps or any(
                "scaler" in name.lower() for name in model.named_steps.keys()
            )
            if has_scaler:
                return False
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

    # --------------------------
    # Anomaly detectors & transfers
    # --------------------------
    def build_anomaly_detectors(
        self,
        X_tr: NDArray,
        y_tr: NDArray,
        contamination: float = 0.05,
    ) -> None:
        """
        Costruisce IsolationForest per le classi 4, 6, 8 (se presenti e con campioni sufficienti).
        """
        scaler = StandardScaler().fit(X_tr)
        X_scaled = scaler.transform(X_tr)
        self.scaler = scaler

        self.detectors.clear()
        for cls in (4, 6, 8):
            mask = (y_tr == cls)
            if not np.any(mask):
                continue
            X_cls = X_scaled[mask]
            if len(X_cls) < 10:  # troppo pochi campioni per un detector affidabile
                continue
            det = IsolationForest(
                contamination=contamination,
                random_state=self.rng_seed,
                n_jobs=-1,
            ).fit(X_cls)
            self.detectors[cls] = det

    def apply_transfers(
        self,
        probs: NDArray,
        X_test: NDArray,
        alpha: float = 0.3,
    ) -> Dict[str, int] | Tuple[Dict[str, int], NDArray]:
        """
        Applica trasferimenti di probabilità basati sui detector.
        - Classi 4 e 6: trasferisce una quota verso 5 e 7 rispettivamente.
        - Classe 8: trasferisce verso 9 e 10 con split DINAMICO basato sull'anomalia:
            a in [0,1] (alto = più anomalo) -> quota_9 = (1 - a), quota_10 = a.
        """
        if (not hasattr(self, "scaler")) or (not self.detectors):
            return {}

        Xs = self.scaler.transform(X_test)
        transfers: Dict[str, int] = {}

        for cls, det in self.detectors.items():
            # decision_function: più alto = più "normale" -> mappiamo a [0,1] con sigmoide invertita
            s = det.decision_function(Xs)
            a = 1.0 / (1.0 + np.exp((s - np.median(s)) / (np.std(s) + 1e-6)))  # alto = più anomalo

            if cls in (4, 6):
                dst = int(self.missing_map[cls])  # 4->5, 6->7
                moved = alpha * a * probs[:, cls]
                probs[:, cls] -= moved
                probs[:, dst] += moved
                transfers[f"{cls}_to_{dst}"] = int((moved > 0).sum())

            elif cls == 8:
                dst9, dst10 = self.missing_map[8]
                # aumento leggero di alpha come prima (facoltativo)
                alpha_eff = (alpha + 0.1) if alpha < 0.9 else alpha
                moved = alpha_eff * a * probs[:, 8]
                probs[:, 8] -= moved

                # --- SPLIT DINAMICO:
                b = 0.9
                k = 6.0  # pendenza
                a_tilt = 1/(1 + np.exp(-k*(a - b)))      # sigmoid(a - b)
                p9 = 1 - a_tilt
                p10 = a_tilt
                     # complementare

                probs[:, dst9] += p9 * moved
                probs[:, dst10] += p10 * moved
                transfers["8_to_9_10"] = int((moved > 0).sum())

        # clamp & renorm
        np.maximum(probs, 0.0, out=probs)
        probs_sum = probs.sum(axis=1, keepdims=True)
        probs /= np.maximum(probs_sum, 1e-12)

        return transfers

    # --------------------------
    # Pipeline principale
    # --------------------------
    def generate_submission(
        self,
        X_tr: NDArray,
        y_tr: NDArray,
        X_test: NDArray,
        test_ids: Sequence,
        model: Any,
        conf_thresh: float = 0.6,
        contamination: float = 0.05,
        alpha: float = 0.3,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Genera il DataFrame di submission e le diagnostics.

        Steps:
          1) (eventuale) scaling per il modello
          2) predict_proba
          3) mapping classi modello -> classi reali
          4) espansione a 11 classi (0..10)
          5) anomaly detectors + transfers
          6) confidence semplice: 1 se max(prob) >= conf_thresh, altrimenti 0
        """
        # 1) Prepara input per il modello
        if self.needs_model_scaling(model):
            model_scaler = StandardScaler().fit(X_tr)
            X_model_input = model_scaler.transform(X_test)
        else:
            X_model_input = X_test

        # 2) Predizioni base
        proba_seen = model.predict_proba(X_model_input)

        # 3) Mapping classi
        enc_classes = self.get_model_classes(model)
        real_classes = np.sort(np.unique(y_tr))

        if set(enc_classes) == set(real_classes):
            col_labels = enc_classes
        else:
            # Caso in cui enc_classes siano indici di real_classes
            col_labels = real_classes[enc_classes]

        # 4) Espansione a 11 classi
        probs_11 = self.expand_to_0_10(proba_seen, col_labels)

        # 5) Detector & transfers
        self.build_anomaly_detectors(X_tr, y_tr, contamination)
        transfers = self.apply_transfers(probs_11, X_test, alpha)

        probs_final = probs_11

        # 6) Confidence semplice
        max_proba = probs_final.max(axis=1)
        confidence_bin = (max_proba >= conf_thresh).astype(int)

        # 7) Costruzione DataFrame
        cols = [f"prob_{i}" for i in range(11)]
        df = pd.DataFrame(probs_final, columns=cols)
        df.insert(0, "id", test_ids)
        df["confidence"] = confidence_bin

        # 8) Diagnostics
        diagnostics = {
            "detectors_built": list(self.detectors.keys()),
            "transfers": transfers,
            "confidence_stats": {
                "threshold": float(conf_thresh),
                "mean_of_max_proba": float(max_proba.mean()),
                "high_conf_pct": float(confidence_bin.mean()),
            },
        }

        return df, diagnostics

    def create_submission_simple(
        self,
        X_tr: NDArray,
        y_tr: NDArray,
        X_test: NDArray,
        test_df: pd.DataFrame,
        model: Any,
        conf_thresh: float = 0.6,
        contamination: float = 0.05,
        alpha: float = 0.3,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Wrapper pratico che prende direttamente test_df con colonna 'id'.
        """
        return self.generate_submission(
            X_tr=X_tr,
            y_tr=y_tr,
            X_test=X_test,
            test_ids=test_df["id"].values,
            model=model,
            conf_thresh=conf_thresh,
            contamination=contamination,
            alpha=alpha,
        )
