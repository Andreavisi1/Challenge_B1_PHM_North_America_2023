from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import NearestNeighbors

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
        self.knn_by_class: Dict[int, Any] = {}
        self.k_for_knn: int = 15  # tunabile
        self.knn_metric: str = "euclidean"


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

    # ------------------------------------------------------------
    # TRAIN: scaler, centroidi, anomaly detectors
    # ------------------------------------------------------------
    def build_anomaly_detectors(
        self,
        X_tr: NDArray,
        y_tr: NDArray,
        contamination: float = 0.05,
    ) -> None:
        # 1) Scala i dati
        scaler = StandardScaler().fit(X_tr)
        X_scaled = scaler.transform(X_tr)
        self.scaler = scaler

        # 2) Centroidi per fallback
        self.centroids_ = {}
        unique_classes = np.unique(y_tr)
        for cls in unique_classes:
            mask = (y_tr == cls)
            if np.any(mask):
                self.centroids_[int(cls)] = np.mean(X_scaled[mask], axis=0)

        # 3) IsolationForest per 4,6,8 (se campioni sufficienti)
        self.detectors: Dict[int, IsolationForest] = {}
        for cls in (4, 6, 8):
            mask = (y_tr == cls)
            if not np.any(mask):
                continue
            X_cls = X_scaled[mask]
            if len(X_cls) < 10:
                continue
            det = IsolationForest(
                contamination=contamination,
                random_state=self.rng_seed,
                n_jobs=-1,
            ).fit(X_cls)
            self.detectors[int(cls)] = det

    def apply_transfers(
        self,
        probs: NDArray,
        X_test: NDArray,
        alpha: float = 0.3,
    ) -> Dict[str, int]:
        """
        Applica trasferimenti guidati da IsolationForest e KNN globale (4,6,8).
        - niente softmax: pesi normalizzati linearmente
        - quantità spostata = alpha  * prob(source) solo se la source è argmax
        Regole:
        6 -> {5,7}  (somiglianza a 4 → 5, somiglianza a 8 → 7)
        8 -> {7,9,10}  (vicino a 6 → 7, lontano da {4,6} → 9)
        4 -> 5
        """
        # prerequisiti minimi
        if not hasattr(self, "scaler") or not self.detectors:
            return {}

        # scala il test come nel train
        Xs = self.scaler.transform(X_test)

        def _accumulate(key: str, val: int, transfers: Dict[str, int]):
            transfers[key] = transfers.get(key, 0) + int(val)

        # pre-calcolo: anomaly in [0,1] per ogni detector (4,6,8)
        anomaly_by_cls: Dict[int, NDArray] = {}
        for cls, det in self.detectors.items():
            s = det.decision_function(Xs)
            s_std = (np.std(s) + 1e-6)
            # più anomalo => valore più vicino a 1
            anomaly_by_cls[int(cls)] = 1.0 / (1.0 + np.exp((s - np.median(s)) / s_std))

        transfers: Dict[str, int] = {}

        # Soglie specifiche per classe
        anomaly_thresholds = {}
        for cls in [4, 6, 8]:
            if cls in anomaly_by_cls:
                anomaly_thresholds[cls] = np.percentile(anomaly_by_cls[cls], 25)  # CORREZIONE: 75 per top 25%

        print(f"Anomaly thresholds: {anomaly_thresholds}")

        # PRE-CALCOLO distanze dai centroidi (serve per tutte le classi)
        all_distances = []
        for centroid_cls in [4, 6, 8]:
            if centroid_cls in self.centroids_:
                dist = np.linalg.norm(Xs - self.centroids_[centroid_cls], axis=1)
                all_distances.append(dist)
        
        mean_dist = None
        if all_distances:
            mean_dist = np.mean(all_distances, axis=0)

        # ==========================================
        # CLASSE 6 -> {5,7}
        # ==========================================
        cls = 6
        if cls in self.detectors and cls in anomaly_thresholds and hasattr(self, 'centroids_'):
            dom = (probs.argmax(axis=1) == cls)
            anomaly_mask = anomaly_by_cls[cls] > anomaly_thresholds[cls]
            combined_mask = dom & anomaly_mask

            # quantità da spostare SOLO per campioni anomali
            total_moved = np.where(combined_mask, alpha * probs[:, cls], 0.0)

            if mean_dist is not None and combined_mask.any():
                # Calcola p50 solo sui campioni che verranno effettivamente trasferiti
                p50 = np.percentile(mean_dist[combined_mask], 50)

                # Maschere di destinazione calcolate SOLO sul subset dei campioni anomali
                to_5_mask = np.zeros_like(combined_mask, dtype=bool)
                to_7_mask = np.zeros_like(combined_mask, dtype=bool)
                
                to_5_mask[combined_mask] = mean_dist[combined_mask] <= p50
                to_7_mask[combined_mask] = mean_dist[combined_mask] > p50

                # Calcola i pesi
                w5 = to_5_mask.astype(float)
                w7 = to_7_mask.astype(float)

                # Applica i trasferimenti
                moved_to_5 = w5 * total_moved
                moved_to_7 = w7 * total_moved

                probs[:, cls] -= (moved_to_5 + moved_to_7)
                probs[:, 5]   += moved_to_5
                probs[:, 7]   += moved_to_7

                _accumulate("6_to_5_distance_gradient", (moved_to_5 > 0).sum(), transfers)
                _accumulate("6_to_7_distance_gradient", (moved_to_7 > 0).sum(), transfers)

        # ==========================================
        # CLASSE 8 -> {7,9,10}
        # ==========================================
        cls = 8
        if cls in self.detectors and cls in anomaly_thresholds and hasattr(self, 'centroids_'):
            dom = (probs.argmax(axis=1) == cls)
            anomaly_mask = anomaly_by_cls[cls] > anomaly_thresholds[cls]
            combined_mask = dom & anomaly_mask

            # quantità da spostare SOLO per campioni anomali
            total_moved = np.where(combined_mask, alpha * probs[:, cls], 0.0)

            if mean_dist is not None and combined_mask.any():
                # Percentili calcolati SOLO sui campioni che verranno trasferiti
                subset_distances = mean_dist[combined_mask]
                p33 = np.percentile(subset_distances, 33)
                p66 = np.percentile(subset_distances, 66)
                
                # Maschere per ogni destinazione
                to_7_mask = np.zeros_like(combined_mask, dtype=bool)
                to_9_mask = np.zeros_like(combined_mask, dtype=bool)
                to_10_mask = np.zeros_like(combined_mask, dtype=bool)
                
                to_7_mask[combined_mask] = mean_dist[combined_mask] <= p33
                to_9_mask[combined_mask] = (mean_dist[combined_mask] > p33) & (mean_dist[combined_mask] <= p66)
                to_10_mask[combined_mask] = mean_dist[combined_mask] > p66
                
                # Calcola i pesi
                w7 = to_7_mask.astype(float)
                w9 = to_9_mask.astype(float)
                w10 = to_10_mask.astype(float)

                # Applica i trasferimenti
                moved_to_7  = w7  * total_moved
                moved_to_9  = w9  * total_moved
                moved_to_10 = w10 * total_moved

                probs[:, cls] -= (moved_to_7 + moved_to_9 + moved_to_10)
                probs[:, 7]   += moved_to_7
                probs[:, 9]   += moved_to_9
                probs[:, 10]  += moved_to_10

                _accumulate("8_to_7_distance_gradient", (moved_to_7  > 0).sum(), transfers)
                _accumulate("8_to_9_distance_gradient", (moved_to_9  > 0).sum(), transfers)
                _accumulate("8_to_10_distance_gradient", (moved_to_10 > 0).sum(), transfers)

        # ==========================================
        # CLASSE 4 -> {5}
        # ==========================================
        cls = 4
        if cls in self.detectors and cls in anomaly_thresholds:
            dom = (probs.argmax(axis=1) == cls)
            anomaly_mask = anomaly_by_cls[cls] > anomaly_thresholds[cls]
            combined_mask = dom & anomaly_mask

            total_moved = np.where(combined_mask, alpha * probs[:, cls], 0.0)

            probs[:, cls] -= total_moved
            probs[:, 5]   += total_moved

            _accumulate("4_to_5", (total_moved > 0).sum(), transfers)

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
