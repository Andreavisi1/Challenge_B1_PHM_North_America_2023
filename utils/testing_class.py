from __future__ import annotations

import logging
from dataclasses import dataclass, field,replace
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Config:
    """Configurazione per il SubmissionGenerator."""
    base_classes: Tuple[int, ...] = (4, 6, 8)
    missing_map: Mapping[int, Sequence[int] | int] = field(default_factory=lambda: {4: 5, 6: 7, 8: (9, 10)})
    contamination: float = 0.05
    rng_seed: int = 42
    models_requiring_scaling: Tuple[str, ...] = (
        "RidgeClassifier",
        "LogReg_ElasticNet", 
        "LogisticRegression",
    )


class TestingClass:
    """
    Generatore di submission per competizioni di machine learning con gestione
    avanzata di anomalie, confidence ordinale e trasferimenti probabilistici.
    """
    
    def __init__(self, config: Config = None, debug: bool = False):
        """
        Inizializza il generatore.
        
        Args:
            config: Configurazione personalizzata (usa default se None)
            debug: Abilita logging dettagliato
        """
        self.config = config or Config()
        self.logger = self._setup_logger(debug)
        self.detectors: Dict[int, IsolationForest] = {}
        self.thresholds: Dict[int, float] = {}
        self.diagnostics: Dict[str, Any] = {}
    
    def _setup_logger(self, debug: bool) -> logging.Logger:
        """Configura il logger."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if not logger.handlers:
            logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, 
                              format="%(message)s")
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        return logger
    
    @staticmethod
    def get_model_classes(model: Any) -> NDArray[np.int_]:
        """Ricava `model.classes_` in modo robusto (supporta modelli singoli e pipeline)."""
        if hasattr(model, "classes_"):
            return np.asarray(model.classes_)
        if hasattr(model, "named_steps"):
            step = model.named_steps.get("model", list(model.named_steps.values())[-1])
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
        if hasattr(model, "steps"):
            last = model.steps[-1][1]
            if hasattr(last, "classes_"):
                return np.asarray(last.classes_)
        raise AttributeError("Impossibile recuperare `classes_` dal best_model.")
    
    def detect_model_type(self, model: Any) -> Tuple[str, bool]:
        """
        Rileva il tipo di modello per sapere se applicare scaling.
        Returns: (model_name, has_scaler_in_pipeline)
        """
        if hasattr(model, "named_steps"):
            model_step = model.named_steps.get("model")
            if model_step is not None:
                model_name = model_step.__class__.__name__
                has_scaler = "scaler" in model.named_steps
                return model_name, has_scaler

        if hasattr(model, "steps"):
            last_step = model.steps[-1][1]
            model_name = last_step.__class__.__name__
            has_scaler = any("scaler" in name.lower() for name, _ in model.steps[:-1])
            return model_name, has_scaler

        return model.__class__.__name__, False
    
    def needs_scaling(self, model_name: str) -> bool:
        """Determina se il modello richiede scaling basandosi sul nome."""
        return any(req in model_name for req in self.config.models_requiring_scaling)
    
    def enc_to_real_mapper(self, model: Any, y_tr: NDArray[np.int_]) -> Tuple[NDArray[np.int_], NDArray[np.int_], Callable[[NDArray[np.int_]], NDArray[np.int_]]]:
        """
        Crea il mapping da etichette codificate a etichette reali.
        Returns: (enc_classes, col_labels_real, funzione enc->real)
        """
        enc = self.get_model_classes(model).astype(int)
        real_sorted = np.sort(np.unique(y_tr)).astype(int)

        if set(enc.tolist()) == set(real_sorted.tolist()):
            col_labels_real = enc
            enc_to_real = lambda y_enc: np.asarray(y_enc, int).ravel()
        else:
            col_labels_real = real_sorted[enc]
            enc_to_real = lambda y_enc: real_sorted[np.asarray(y_enc, int).ravel()]

        return enc, col_labels_real, enc_to_real
    
    @staticmethod
    def expand_to_0_10(proba_seen: NDArray[np.floating], col_labels_real: NDArray[np.int_]) -> NDArray[np.floating]:
        """Espande le probabilità alle 11 classi (0–10) partendo dalle classi presenti nel modello."""
        out = np.zeros((proba_seen.shape[0], 11), dtype=float)
        out[:, col_labels_real.astype(int)] = proba_seen
        return out
    
    @staticmethod
    def ordinal_confidence(probs: NDArray[np.floating], class_values: NDArray[np.int_] = None) -> NDArray[np.floating]:
        """
        Confidence che considera la natura ordinale: più la distribuzione è concentrata 
        su classi vicine, più alta la confidence (basata su varianza pesata).
        """
        if class_values is None:
            class_values = np.arange(probs.shape[1])
        expected_vals = (probs * class_values).sum(axis=1)
        variance = (probs * (class_values - expected_vals[:, None]) ** 2).sum(axis=1)
        max_var = ((class_values - class_values.mean()) ** 2).mean()
        max_var = float(max(max_var, 1e-12))
        return 1 - np.sqrt(np.clip(variance / max_var, 0.0, 1.0))

    
    @staticmethod
    def apply_ordinal_smoothing(probs: NDArray[np.floating], smooth_factor: float) -> NDArray[np.floating]:
        """Smussa le probabilità tra classi adiacenti rispettando l'ordine."""
        if smooth_factor <= 0:
            return probs

        probs_smooth = probs.copy()

        # classi interne 1–9
        for i in range(1, 10):
            transfer = smooth_factor * probs[:, i] * 0.1
            probs_smooth[:, i] -= 2 * transfer
            probs_smooth[:, i - 1] += transfer
            probs_smooth[:, i + 1] += transfer

        # estremi 0 e 10
        t0 = smooth_factor * probs[:, 0] * 0.1
        probs_smooth[:, 0] -= t0
        probs_smooth[:, 1] += t0

        t10 = smooth_factor * probs[:, 10] * 0.1
        probs_smooth[:, 10] -= t10
        probs_smooth[:, 9] += t10

        # proiezione su simplex
        probs_smooth = np.maximum(probs_smooth, 0.0)
        row_sums = probs_smooth.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return probs_smooth / row_sums
    
    @staticmethod
    def calculate_improved_confidence(probs: NDArray[np.floating], orig_confidence: NDArray[np.floating]) -> NDArray[np.floating]:
        """Confidence composita: ordinale + entropia + max prob + originale."""
        ord_conf = TestingClass.ordinal_confidence(probs)
        eps = 1e-12
        entropy = -(probs * np.log(probs + eps)).sum(axis=1)
        ent_conf = 1 - entropy / np.log(probs.shape[1])
        max_prob_conf = np.max(probs, axis=1)
        final = 0.3 * ord_conf + 0.25 * ent_conf + 0.25 * max_prob_conf + 0.2 * orig_confidence
        return np.clip(final, 0.0, 1.0)
    
    def _build_anomaly_detectors(self, X_tr: NDArray[np.floating], y_tr: NDArray[np.int_], 
                                Xtr_det: NDArray[np.floating]) -> None:
        """Costruisce i detector di anomalie per le classi base."""
        self.detectors = {}
        self.thresholds = {}
        
        for b in self.config.base_classes:
            mask_b = (y_tr == b)
            if not np.any(mask_b):
                continue

            Xb = Xtr_det[mask_b]
            if len(Xb) == 0:
                continue

            cont = self.config.contamination
            n_est = int(max(100, min(500, len(Xb) * 4)))

            det = IsolationForest(
                n_estimators=n_est,
                contamination=cont,
                max_samples=min(256, len(Xb)),
                random_state=self.config.rng_seed,
                n_jobs=-1,
            ).fit(Xb)
            self.detectors[b] = det

            train_scores = det.decision_function(Xb)
            self.thresholds[b] = float(np.percentile(train_scores, 10))
    
    def _get_tail_info_for_class8(self, X_tr: NDArray[np.floating], y_tr: NDArray[np.int_], 
                                 Xtr_det: NDArray[np.floating]) -> Dict[str, Any]:
        """Calcola informazioni per gestione dinamica delle code per classe 8."""
        tail_info = {"strategy": "adaptive"}
        if 8 in self.detectors:
            tr8_mask = (y_tr == 8)
            if np.any(tr8_mask):
                tr8_scores = self.detectors[8].decision_function(Xtr_det[tr8_mask])
                tail_info.update({
                    "q85": float(np.percentile(tr8_scores, 15)),  # → 9
                    "q95": float(np.percentile(tr8_scores, 5)),
                    "q99": float(np.percentile(tr8_scores, 1)),   # → 10
                    "mean": float(tr8_scores.mean()),
                    "std": float(tr8_scores.std()),
                })
        return tail_info
    
    def _apply_transfers(self, probs_final: NDArray[np.floating], y_pred_real: NDArray[np.int_],
                        Xte_det: NDArray[np.floating], orig_confidence: NDArray[np.floating],
                        tail_info: Dict[str, Any], confidence_weight: float, 
                        min_tail_samples: int) -> Dict[str, Any]:
        """Applica i trasferimenti graduali basati sui detector di anomalie."""
        transfer_log = {}
        rng_gen = np.random.RandomState(self.config.rng_seed)

        for b in self.config.base_classes:
            det = self.detectors.get(b)
            if det is None:
                continue

            mask_b_pred = (y_pred_real == b)
            if not np.any(mask_b_pred):
                continue

            scores = det.decision_function(Xte_det[mask_b_pred])
            thr = self.thresholds[b]
            anom_mask = scores < thr
            if not np.any(anom_mask):
                continue

            idx_anom = np.where(mask_b_pred)[0][anom_mask]
            scores_anom = scores[anom_mask]
            orig_conf_anom = orig_confidence[idx_anom]

            if b in (4, 6):
                # Trasferimento semplice 4→5, 6→7
                dst = int(self.config.missing_map[b])
                anomaly_strength = np.clip((thr - scores_anom) / (abs(thr) + 1e-6), 0, 1)
                conf_penalty = np.clip(1 - orig_conf_anom, 0, 1)
                alpha = 0.4 * anomaly_strength + confidence_weight * conf_penalty
                alpha = np.clip(alpha, 0.05, 0.75)

                moved = alpha * probs_final[idx_anom, b]
                probs_final[idx_anom, b] -= moved
                probs_final[idx_anom, dst] += moved

                transfer_log[f"{b}_to_{dst}"] = {
                    "n_transfers": int(len(idx_anom)),
                    "total_moved": float(moved.sum()),
                    "avg_alpha": float(alpha.mean()),
                    "alpha_range": f"{alpha.min():.3f}-{alpha.max():.3f}",
                }

            elif b == 8 and len(tail_info) > 1:
                # Gestione complessa 8→9/10
                self._handle_class8_transfers(
                    probs_final, idx_anom, scores_anom, orig_conf_anom, 
                    tail_info, confidence_weight, min_tail_samples, 
                    rng_gen, transfer_log
                )

        return transfer_log
    
    def _handle_class8_transfers(self, probs_final: NDArray[np.floating], idx_anom: NDArray[np.int_],
                               scores_anom: NDArray[np.floating], orig_conf_anom: NDArray[np.floating],
                               tail_info: Dict[str, Any], confidence_weight: float, 
                               min_tail_samples: int, rng_gen: np.random.RandomState,
                               transfer_log: Dict[str, Any]) -> None:
        """Gestisce i trasferimenti complessi per la classe 8."""
        very_extreme = scores_anom < tail_info["q99"]  # → 10
        extreme = (scores_anom < tail_info["q95"]) & (~very_extreme)
        moderate = (scores_anom < tail_info["q85"]) & (~extreme) & (~very_extreme)

        # → 10 (molto estremi)
        if np.any(very_extreme) and np.sum(very_extreme) >= min_tail_samples:
            idx_10 = idx_anom[very_extreme]
            sc_10 = scores_anom[very_extreme]
            conf_10 = orig_conf_anom[very_extreme]

            anomaly_str = np.clip((tail_info["q99"] - sc_10) / (abs(tail_info["q99"]) + 1e-6), 0, 1)
            conf_penalty = 1 - conf_10
            alpha_10 = np.clip(0.6 * anomaly_str + confidence_weight * conf_penalty, 0.3, 0.9)

            moved_10 = alpha_10 * probs_final[idx_10, 8]
            probs_final[idx_10, 8] -= moved_10
            probs_final[idx_10, 10] += moved_10

            transfer_log["8_to_10"] = {
                "n_transfers": int(len(idx_10)),
                "total_moved": float(moved_10.sum()),
                "avg_alpha": float(alpha_10.mean()),
            }

        # → 9 (moderati + parte degli estremi)
        to_9_mask = moderate.copy()
        if np.any(extreme):
            extreme_idx = np.where(extreme)[0]
            pick_extreme = rng_gen.random(len(extreme_idx)) < 0.6
            chosen = np.zeros_like(extreme, dtype=bool)
            chosen[extreme_idx[pick_extreme]] = True
            to_9_mask |= chosen
            remaining_extreme = extreme & (~chosen)
        else:
            remaining_extreme = np.zeros_like(extreme, dtype=bool)

        if np.any(to_9_mask):
            idx_9 = idx_anom[to_9_mask]
            sc_9 = scores_anom[to_9_mask]
            conf_9 = orig_conf_anom[to_9_mask]

            anomaly_str = np.clip((tail_info["q85"] - sc_9) / (abs(tail_info["q85"]) + 1e-6), 0, 1)
            conf_penalty = 1 - conf_9
            alpha_9 = np.clip(0.4 * anomaly_str + confidence_weight * conf_penalty, 0.1, 0.7)

            moved_9 = alpha_9 * probs_final[idx_9, 8]
            probs_final[idx_9, 8] -= moved_9
            probs_final[idx_9, 9] += moved_9

            transfer_log["8_to_9"] = {
                "n_transfers": int(len(idx_9)),
                "total_moved": float(moved_9.sum()),
                "avg_alpha": float(alpha_9.mean()),
            }

        # Gestisci estremi rimanenti → 10
        if np.any(remaining_extreme):
            idx_10_rem = idx_anom[remaining_extreme]
            conf_10_rem = orig_conf_anom[remaining_extreme]
            alpha_10_rem = np.clip(0.5 + 0.3 * (1 - conf_10_rem), 0.2, 0.8)
            moved_10_rem = alpha_10_rem * probs_final[idx_10_rem, 8]
            probs_final[idx_10_rem, 8] -= moved_10_rem
            probs_final[idx_10_rem, 10] += moved_10_rem

            if "8_to_10" in transfer_log:
                transfer_log["8_to_10"]["n_transfers"] += int(len(idx_10_rem))
                transfer_log["8_to_10"]["total_moved"] += float(moved_10_rem.sum())
            else:
                transfer_log["8_to_10"] = {
                    "n_transfers": int(len(idx_10_rem)),
                    "total_moved": float(moved_10_rem.sum()),
                    "avg_alpha": float(alpha_10_rem.mean()),
                }
    
    def generate_submission(
        self,
        X_tr: NDArray[np.floating],
        y_tr: NDArray[np.int_],
        X_test: NDArray[np.floating],
        test_ids: Sequence[Any],
        best_model: Any,
        conf_thresh: float = 0.7,
        ordinal_smooth: float = 0.05,
        confidence_weight: float = 0.3,
        min_tail_samples: int = 3,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Genera il DataFrame di submission con gestione avanzata delle anomalie.
        
        Args:
            X_tr: Features del training set
            y_tr: Target del training set
            X_test: Features del test set
            test_ids: IDs dei campioni di test
            best_model: Modello addestrato
            conf_thresh: Soglia per la confidence binaria
            ordinal_smooth: Fattore di smoothing ordinale
            confidence_weight: Peso della confidence nei trasferimenti
            min_tail_samples: Numero minimo di campioni per trasferimenti estremi
            
        Returns:
            Tupla (DataFrame submission, diagnostics)
        """
        # 1) Rileva tipo di modello e strategia di scaling
        model_name, has_scaler_in_pipeline = self.detect_model_type(best_model)
        model_requires_scaling = self.needs_scaling(model_name)

        self.logger.debug("Model detected: %s", model_name)
        self.logger.debug("Has scaler in pipeline: %s", has_scaler_in_pipeline)
        self.logger.debug("Requires scaling: %s", model_requires_scaling)

        # 2) Prepara i dati per le predizioni del modello
        if has_scaler_in_pipeline or not model_requires_scaling:
            X_model_input = X_test
            self.logger.debug("Using RAW data for model predictions")
        else:
            model_scaler = StandardScaler().fit(X_tr)
            X_model_input = model_scaler.transform(X_test)
            self.logger.debug("Using SCALED data for model predictions")

        # 3) Dati per detector di anomalie (sempre scalati)
        detector_scaler = StandardScaler().fit(X_tr)
        Xtr_det = detector_scaler.transform(X_tr)
        Xte_det = detector_scaler.transform(X_test)

        # 4) Predizioni del modello
        _, col_labels_real, enc_to_real = self.enc_to_real_mapper(best_model, y_tr)
        proba_seen = best_model.predict_proba(X_model_input)
        y_pred_real = enc_to_real(best_model.predict(X_model_input))
        probs_final = self.expand_to_0_10(proba_seen, col_labels_real)
        orig_confidence = self.ordinal_confidence(probs_final)

        # 5) Costruisci detector di anomalie
        self._build_anomaly_detectors(X_tr, y_tr, Xtr_det)

        # 6) Informazioni per gestione code classe 8
        tail_info = self._get_tail_info_for_class8(X_tr, y_tr, Xtr_det)

        # 7) Applica trasferimenti
        transfer_log = self._apply_transfers(
            probs_final, y_pred_real, Xte_det, orig_confidence,
            tail_info, confidence_weight, min_tail_samples
        )

        # 8) Smoothing ordinale
        if ordinal_smooth > 0:
            probs_final = self.apply_ordinal_smoothing(probs_final, ordinal_smooth)

        # 9) Confidence finale
        final_confidence = self.calculate_improved_confidence(probs_final, orig_confidence)

        # 10) DataFrame risultato
        cols = [f"prob_{c}" for c in range(11)]
        df = pd.DataFrame(probs_final, columns=cols)
        df.insert(0, "id", np.asarray(test_ids))
        df["confidence"] = (final_confidence > conf_thresh).astype(int)

        # 11) Diagnostics
        self.diagnostics = {
            "model_info": {
                "name": model_name,
                "has_scaler": has_scaler_in_pipeline,
                "requires_scaling": model_requires_scaling,
                "input_scaled": not (has_scaler_in_pipeline or not model_requires_scaling),
            },
            "detectors_info": {"thresholds": self.thresholds, "n_detectors": len(self.detectors)},
            "tail_info": tail_info,
            "transfers": transfer_log,
            "final_distribution": {f"prob_{i}": float(df[f"prob_{i}"].sum()) for i in range(11)},
            "confidence_stats": {
                "mean": float(final_confidence.mean()),
                "std": float(final_confidence.std()),
                "high_conf_pct": float((final_confidence > conf_thresh).mean()),
            },
        }

        if self.logger.level == logging.DEBUG:
            self.logger.debug("\n=== DIAGNOSTICS ===")
            for section, data in self.diagnostics.items():
                self.logger.debug("\n%s:", section)
                self.logger.debug("%s", data)

        return df, self.diagnostics
    
    def create_submission_simple(
        self,
        X_tr: NDArray[np.floating],
        y_tr: NDArray[np.int_],
        X_test: NDArray[np.floating],
        test_df: pd.DataFrame,
        best_model: Any,
        conf_thresh: float = 0.25,
        ordinal_smooth: float = 0.03,
        contamination: float = 0.05
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Wrapper semplificato per uso comune."""
        self.config = replace(self.config, contamination=contamination)
        return self.generate_submission(
            X_tr=X_tr,
            y_tr=y_tr,
            X_test=X_test,
            test_ids=test_df["id"].values,
            best_model=best_model,
            conf_thresh=conf_thresh,
            ordinal_smooth=ordinal_smooth,
        )

'''Riassuntino operativo (in ordine)

Il modello produce probabilità sulle classi viste ⇒ si espandono a 10.

Si calcola la confidence ordinale iniziale.

Si addestrano IsolationForest per 4/6/8 (sul training, scalato).

Sul test, per i casi predetti 4/6/8 e giudicati anomali, si sposta una quota di probabilità a 5/7/9/10 in modo guidato.

Si applica lo smoothing ordinale (ammorbidisce verso i vicini).

Si ricalcola la confidence finale (mix di ordinale, entropia, max prob, originale) e si binarizza con conf_thresh.
'''