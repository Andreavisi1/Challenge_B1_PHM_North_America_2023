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
    Generatore di submission per competizioni ML.

    Scelte progettuali :
    - K-NN per definire la distanza di un punto da una classe (manifold-aware).
    - Routing per-campione indipendente dal batch:
        * 6 -> {5,7}: TUTTO verso 5 se d_knn(., 4) <= d_knn(., 8), altrimenti TUTTO verso 7.
        * 8 -> {7,9,10}: se vicino a 6 (d_knn(.,6) <= T6) => 7;
                         se lontano da 6 (d_knn(.,6) >= Tfar) => 9;
                         altrimenti => 10.
        * 4 -> 5 (sempre, se anomalo come 4).
    - Soglie T6 e Tfar derivate SOLO dal TRAIN via k-NN.
    - Trasferimento frazionario controllato da `alpha` (es. 0.3).
    - Confidence binaria semplice su max probability.
    """

    def __init__(self, rng_seed: int = 42):
        self.rng_seed = rng_seed

        # Anomaly detectors per alcune classi
        self.detectors: Dict[int, IsolationForest] = {}

        # Mappature legacy (non usate direttamente nel routing ma mantenute per compatibilità)
        self.missing_map: Dict[int, int | Tuple[int, int]] = {4: 5, 6: 7, 8: (9, 10)}

        # Scaling
        self.scaler: Optional[StandardScaler] = None

        # ---- kNN per-classe (routing manifold-aware) ----
        self.knn_by_class: Dict[int, NearestNeighbors] = {}
        self.k_for_knn: int = 15
        self.knn_metric: str = "euclidean"

        # Soglie derivate dal train per il routing 8 -> {7,9,10}
        self._T6_train_: Optional[float] = None   # "vicino a 6"
        self._Tfar_train_: Optional[float] = None # "lontano da 6"

        # Valori usati in test (per diagnostica)
        self._last_T6_used_: Optional[float] = None
        self._last_Tfar_used_: Optional[float] = None

    # --------------------------
    # Utility per il modello
    # --------------------------
    @staticmethod
    def get_model_classes(model: Any) -> NDArray[np.int_]:
        if hasattr(model, "classes_"):
            return np.asarray(model.classes_)
        if hasattr(model, "named_steps"):
            step = model.named_steps.get("model", list(model.named_steps.values())[-1])
            return np.asarray(step.classes_)
        if hasattr(model, "steps"):
            return np.asarray(model.steps[-1][1].classes_)
        raise AttributeError("Impossibile recuperare classes_ dal modello o dalla pipeline.")

    @staticmethod
    def expand_to_0_10(proba_seen: NDArray, col_labels: NDArray) -> NDArray:
        out = np.zeros((proba_seen.shape[0], 11), dtype=float)
        out[:, col_labels.astype(int)] = proba_seen
        out_sum = out.sum(axis=1, keepdims=True)
        out /= np.maximum(out_sum, 1e-12)
        return out

    def needs_model_scaling(self, model: Any) -> bool:
        scaling_models = ("LogisticRegression", "RidgeClassifier", "SVC")

        if hasattr(model, "named_steps"):
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

    # ---------- kNN helpers ----------
    def _fit_knn_per_class(self, X_scaled: NDArray, y_tr: NDArray) -> None:
        self.knn_by_class.clear()
        uniq = np.unique(y_tr)
        for cls in uniq:
            Xc = X_scaled[y_tr == cls]
            if len(Xc) == 0:
                continue
            n_neighbors = min(self.k_for_knn, len(Xc))
            knn = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric=self.knn_metric,
                n_jobs=-1
            ).fit(Xc)
            self.knn_by_class[int(cls)] = knn

    def _knn_distance_to_class(self, X: NDArray, cls_id: int, reducer: str = "mean") -> NDArray:
        """
        Distanza di un punto da una classe come media/mediana delle distanze ai k vicini
        della classe cls_id. Restituisce +inf se la classe non ha KNN addestrato.
        """
        knn = self.knn_by_class.get(cls_id)
        if knn is None:
            return np.full(X.shape[0], np.inf, dtype=float)
        dists, _ = knn.kneighbors(X, return_distance=True)
        if reducer == "median":
            return np.median(dists, axis=1)
        return dists.mean(axis=1)

    # ------------------------------------------------------------
    # TRAIN: scaler, k-NN, soglie dal train, anomaly detectors
    # ------------------------------------------------------------
    def build_anomaly_detectors(
        self,
        X_tr: NDArray,
        y_tr: NDArray,
        contamination: float = 0.05,
    ) -> None:
        # 1) Scala i dati di train
        scaler = StandardScaler().fit(X_tr)
        X_scaled = scaler.transform(X_tr)
        self.scaler = scaler

        # 2) k-NN per-classe (solo questo; niente centroidi)
        self._fit_knn_per_class(X_scaled, y_tr)

        # 3) Soglie FISSE (dal train) per il routing di 8 -> {7,9,10}
        #    T6: "vicino a 6" ; Tfar: "lontano da 6"
        self._T6_train_ = None
        self._Tfar_train_ = None

        if 6 in self.knn_by_class:
            X6 = X_scaled[y_tr == 6]
            if len(X6):
                d6_train = self._knn_distance_to_class(X6, 6)
                self._T6_train_ = float(np.quantile(d6_train, 0.75))

        if 4 in self.knn_by_class and 6 in self.knn_by_class:
            X8 = X_scaled[y_tr == 8]
            d6_8 = self._knn_distance_to_class(X8, 6)
            self._Tfar_train_ = float(np.quantile(d6_8, 0.95))

        # 4) IsolationForest per 4/6/8 (se campioni sufficienti)
        self.detectors = {}
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

    # ------------------------------------------------------------
    # TEST: applica trasferimenti winner-takes-all via k-NN
    # ------------------------------------------------------------
    def apply_transfers(
        self,
        probs: NDArray,
        X_test: NDArray,
        alpha: float = 0.3,
        anomaly_threshold: float = 0.5,
    ) -> Dict[str, int]:
        """
        - Routing per-campione indipendente dal batch:
        * 6 -> {5,7}: TUTTO verso 5 se d_knn(., 4) <= d_knn(., 8), altrimenti TUTTO verso 7.
        * 8 -> {7,9,10}: se vicino a 6 (d_knn(.,6) <= T6) => 7;
                         se NON vicino a 6 e NON lontano da 6 => 9;
                         se NON vicino a 6 e lontano da 6 => 10.
        * 4 -> 5 (sempre, se anomalo come 4).

        """
        if self.scaler is None or not self.detectors or not self.knn_by_class:
            return {}

        Xs = self.scaler.transform(X_test)

        def _accumulate(key: str, val: int, transfers: Dict[str, int]):
            transfers[key] = transfers.get(key, 0) + int(val)

        # anomaly score in [0,1] per ciascun detector (4,6,8)
        anomaly_by_cls: Dict[int, NDArray] = {}
        for cls, det in self.detectors.items():
            s = det.decision_function(Xs)
            s_std = (np.std(s) + 1e-6)
            # squash robusto: maggiore => più anomalo
            anomaly_by_cls[int(cls)] = 1.0 / (1.0 + np.exp((s - np.median(s)) / s_std))

        # Distanze SOLO via k-NN; se una classe non ha k-NN => +inf
        def dist_to(cls_id: int) -> NDArray:
            return self._knn_distance_to_class(Xs, cls_id)

        d4 = dist_to(4)
        d6 = dist_to(6)
        d8 = dist_to(8)

        # Soglie derivate dal TRAIN (via k-NN). Se mancanti, fallback robusti su test (raro).
        T6 = self._T6_train_

        Tfar = self._Tfar_train_

        transfers: Dict[str, int] = {}

        # -------- CLASSE 6 -> {5,7} --------
        cls = 6
        if cls in self.detectors:
            dom = (probs.argmax(axis=1) == cls)
            anomaly_mask = anomaly_by_cls[cls] > anomaly_threshold
            idx = np.where(dom & anomaly_mask)[0]
            if idx.size:
                # winner-takes-all: confronta d4 vs d8
                to5 = d4[idx] <= d8[idx]
                to7 = ~to5
                moved = alpha * probs[idx, cls]
                probs[idx, cls] -= moved
                if np.any(to5):
                    probs[idx[to5], 5] += moved[to5]
                if np.any(to7):
                    probs[idx[to7], 7] += moved[to7]
                _accumulate("6_to_5_independent", int(np.sum(to5)), transfers)
                _accumulate("6_to_7_independent", int(np.sum(to7)), transfers)

        # -------- CLASSE 8 -> {7,9,10} --------
        cls = 8
        if cls in self.detectors:
            dom = (probs.argmax(axis=1) == cls)
            anomaly_mask = anomaly_by_cls[cls] > anomaly_threshold
            idx = np.where(dom & anomaly_mask)[0]
            if idx.size:
                near6 = d6[idx] <= T6
                far6 = d6[idx] >= Tfar
                target7 = near6 
                target9 = (~near6) & (~far6)
                target10 = (~near6) & far6

                moved = alpha * probs[idx, cls]
                probs[idx, cls] -= moved
                if np.any(target7):
                    probs[idx[target7], 7] += moved[target7]
                if np.any(target9):
                    probs[idx[target9], 9] += moved[target9]
                if np.any(target10):
                    probs[idx[target10], 10] += moved[target10]

                _accumulate("8_to_7_independent", int(np.sum(target7)), transfers)
                _accumulate("8_to_9_independent", int(np.sum(target9)), transfers)
                _accumulate("8_to_10_independent", int(np.sum(target10)), transfers)

        # -------- CLASSE 4 -> {5} --------
        cls = 4
        if cls in self.detectors:
            dom = (probs.argmax(axis=1) == cls)
            anomaly_mask = anomaly_by_cls[cls] > anomaly_threshold
            idx = np.where(dom & anomaly_mask)[0]
            if idx.size:
                moved = alpha * probs[idx, cls]
                probs[idx, cls] -= moved
                probs[idx, 5] += moved
                _accumulate("4_to_5_independent", int(idx.size), transfers)

        # Rinormalizza per sicurezza
        row_sums = probs.sum(axis=1, keepdims=True)
        probs /= np.maximum(row_sums, 1e-12)

        # Log soglie effettivamente usate
        self._last_T6_used_ = float(T6)
        self._last_Tfar_used_ = float(Tfar)

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
        anomaly_threshold: float = 0.5,
    ) -> Tuple[pd.DataFrame, Dict]:
        # Input al modello: scala solo se il modello lo richiede
        if self.needs_model_scaling(model):
            model_scaler = StandardScaler().fit(X_tr)
            X_model_input = model_scaler.transform(X_test)
        else:
            X_model_input = X_test

        # Probabilità del modello sulle classi viste
        proba_seen = model.predict_proba(X_model_input)

        # Allineamento colonne
        enc_classes = self.get_model_classes(model)
        real_classes = np.sort(np.unique(y_tr))
        if set(enc_classes) == set(real_classes):
            col_labels = enc_classes
        else:
            col_labels = real_classes[enc_classes]

        # Espansione a 0..10
        probs_11 = self.expand_to_0_10(proba_seen, col_labels)

        # Costruisci scaler/kNN/soglie/detector dal TRAIN
        self.build_anomaly_detectors(X_tr, y_tr, contamination)

        # Applica trasferimenti basati su anomalie + routing k-NN
        transfers = self.apply_transfers(probs_11, X_test, alpha, anomaly_threshold)

        probs_final = probs_11

        # Confidence binaria semplice
        max_proba = probs_final.max(axis=1)
        confidence_bin = (max_proba >= conf_thresh).astype(int)

        cols = [f"prob_{i}" for i in range(11)]
        df = pd.DataFrame(probs_final, columns=cols)
        df.insert(0, "id", test_ids)
        df["confidence"] = confidence_bin

        diagnostics = {
            "detectors_built": list(self.detectors.keys()),
            "k_for_knn": int(self.k_for_knn),
            "knn_metric": str(self.knn_metric),
            "transfers": transfers,
            "anomaly_threshold": float(anomaly_threshold),
            "routing_thresholds_from_train": {
                "T6_near6": self._T6_train_,
                "Tfar_far6": self._Tfar_train_,
                "T6_used_in_test": getattr(self, "_last_T6_used_", None),
                "Tfar_used_in_test": getattr(self, "_last_Tfar_used_", None),
            },
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
        anomaly_threshold: float = 0.5,
    ) -> Tuple[pd.DataFrame, Dict]:
        return self.generate_submission(
            X_tr=X_tr,
            y_tr=y_tr,
            X_test=X_test,
            test_ids=test_df["id"].values,
            model=model,
            conf_thresh=conf_thresh,
            contamination=contamination,
            alpha=alpha,
            anomaly_threshold=anomaly_threshold,
        )
