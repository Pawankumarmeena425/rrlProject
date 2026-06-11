from __future__ import annotations

import numpy as np
import pandas as pd

from .objectives import (annualized_return, annualized_volatility, calmar_ratio,
                         expected_max_drawdown, max_drawdown, sharpe_ratio,
                         sterling_ratio)


def _logsig(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def _softmax(z: np.ndarray) -> np.ndarray:
    z_clip = z - np.max(z)
    exp_z = np.exp(np.clip(z_clip, -50.0, 50.0))
    return exp_z / np.sum(exp_z)


class RRLModel:
    def __init__(
        self,
        asset_names: list[str],
        M: int = 104,
        mu: float = 100.0,
        delta: float = 0.001,
        learning_rate: float = 1e-4,
        objective: str = "sharpe",
        variable_weight: bool = False,
        seed: int | None = 42,
    ) -> None:
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.M = M
        self.mu = mu
        self.delta = delta
        self.learning_rate = learning_rate
        self.objective = objective.lower()
        self.variable_weight = variable_weight
        self.random_state = np.random.RandomState(seed)
        self.theta = self.random_state.normal(scale=0.01, size=(self.n_assets, self.M + 2))
        self.loss_history: list[float] = []

    def _build_features(self, returns: pd.DataFrame, t: int, prev_signal: np.ndarray) -> np.ndarray:
        asset_returns = returns.iloc[t - self.M : t].values.T
        X = np.zeros((self.n_assets, self.M + 2), dtype=float)
        X[:, 0] = 1.0
        X[:, 1 : 1 + self.M] = asset_returns[:, ::-1]
        X[:, -1] = prev_signal
        return X

    def _activation(self, z: np.ndarray) -> np.ndarray:
        if self.variable_weight:
            return _logsig(z)
        return np.tanh(z)

    def _activation_derivative(self, z: np.ndarray, activated: np.ndarray) -> np.ndarray:
        if self.variable_weight:
            return activated * (1.0 - activated)
        return 1.0 - np.square(activated)

    def _compute_positions(self, raw_signals: np.ndarray) -> np.ndarray:
        if not self.variable_weight:
            return raw_signals / float(self.n_assets)
        weights = _softmax(np.abs(raw_signals))
        directional = 2.0 * raw_signals - 1.0
        return weights * directional

    def _portfolio_return(
        self,
        prev_positions: np.ndarray,
        positions: np.ndarray,
        asset_returns: np.ndarray,
    ) -> float:
        return self.mu * (
            np.dot(prev_positions, asset_returns)
            - self.delta * np.sum(np.abs(positions - prev_positions))
        )

    def _objective_gradient_coeffs(self, returns: np.ndarray) -> np.ndarray:
        returns = np.asarray(returns, dtype=float)
        T = len(returns)
        if T == 0:
            return np.zeros(0, dtype=float)
        if self.objective == "sharpe":
            mean = np.mean(returns)
            sigma = np.std(returns, ddof=0)
            if sigma <= 1e-12:
                return np.zeros_like(returns)
            return (1.0 / T) * (1.0 / sigma - mean * (returns - mean) / sigma**3)
        if self.objective == "sterling":
            mdd = max_drawdown(returns)
            if mdd <= 1e-12:
                return np.zeros_like(returns)
            return np.full_like(returns, 1.0 / (T * mdd))
        if self.objective == "calmar":
            mean = np.mean(returns)
            sigma = np.std(returns, ddof=0)
            E = expected_max_drawdown(mean, sigma, T)
            if E <= 1e-12:
                return np.zeros_like(returns)
            if mean > 0 and sigma > 0:
                dE_dmean = sigma / mean
                dE_dsigma = 0.63519 + 0.5 * np.log(max(T, 1)) + np.log(max(mean / sigma, 1e-8)) - 1.0
            elif np.isclose(mean, 0.0):
                dE_dmean = 0.0
                dE_dsigma = 1.2533 * np.sqrt(max(T, 1))
            else:
                dE_dmean = -max(T, 1)
                dE_dsigma = 0.0
            if sigma <= 1e-12:
                return np.zeros_like(returns)
            coeff = np.zeros_like(returns)
            for t, r in enumerate(returns):
                dmean_drt = 1.0 / T
                dsigma_drt = (r - mean) / (T * sigma)
                coeff[t] = (1.0 / T) * (1.0 / E) - mean / E**2 * (
                    dE_dmean * dmean_drt + dE_dsigma * dsigma_drt
                )
            return coeff
        return np.zeros_like(returns)

    def train(
        self,
        training_returns: pd.DataFrame,
        epochs: int = 100,
        max_updates: int | None = None,
        verbose: bool = True,
    ) -> None:
        training_returns = training_returns.copy()
        if len(training_returns) <= self.M:
            raise ValueError("Training data must be longer than the lookback window M.")

        max_updates = max_updates or epochs * (len(training_returns) - self.M)
        update_count = 0
        for epoch in range(epochs):
            positions_prev = np.zeros(self.n_assets, dtype=float)
            prev_raw = np.zeros(self.n_assets, dtype=float)
            returns = []
            raw_history = []
            position_history = []
            X_history = []
            gradients = np.zeros_like(self.theta)

            for t in range(self.M, len(training_returns)):
                X = self._build_features(training_returns, t, prev_raw)
                raw = np.einsum("ij,ij->i", X, self.theta)
                signal = self._activation(raw)
                positions = self._compute_positions(signal)
                period_return = self._portfolio_return(
                    positions_prev, positions, training_returns.iloc[t].values.astype(float)
                )
                returns.append(period_return)
                raw_history.append(raw)
                position_history.append(positions)
                X_history.append(X)
                positions_prev = positions
                prev_raw = raw

            returns_array = np.asarray(returns, dtype=float)
            coeffs = self._objective_gradient_coeffs(returns_array)
            for t in range(len(returns_array)):
                positions_cur = position_history[t]
                positions_prev = position_history[t - 1] if t > 0 else np.zeros_like(positions_cur)
                raw = raw_history[t]
                X = X_history[t]
                signal = self._activation(raw)
                dsignal = self._activation_derivative(raw, signal)

                if self.variable_weight:
                    abs_input = np.abs(raw)
                    weights = _softmax(abs_input)
                    sign_input = np.sign(raw)
                    d_abs_d_raw = sign_input * signal * (1.0 - signal)
                    dweights_draw = np.zeros((self.n_assets, self.n_assets), dtype=float)
                    for i in range(self.n_assets):
                        for j in range(self.n_assets):
                            dweights_draw[i, j] = weights[i] * (
                                (1.0 if i == j else 0.0) - weights[j]
                            ) * d_abs_d_raw[j]
                    dpos_draw = np.zeros((self.n_assets, self.n_assets), dtype=float)
                    dsign_draw = 2.0 * signal * (1.0 - signal)
                    for i in range(self.n_assets):
                        for j in range(self.n_assets):
                            dpos_draw[i, j] = dweights_draw[i, j] * (2.0 * raw[i] - 1.0)
                            if i == j:
                                dpos_draw[i, j] += weights[i] * dsign_draw[i]
                else:
                    dpos_draw = np.diag((1.0 / self.n_assets) * dsignal)

                asset_ret = training_returns.iloc[self.M + t].values.astype(float)
                dR_dpos_prev = self.mu * asset_ret
                diff_sign = np.sign(positions_cur - positions_prev)
                dR_dpos_cur = -self.mu * self.delta * diff_sign
                dR_draw = np.zeros(self.n_assets, dtype=float)
                for i in range(self.n_assets):
                    dR_draw[i] = np.dot(dR_dpos_prev, dpos_draw[:, i]) + np.dot(dR_dpos_cur, dpos_draw[:, i])
                gradients += np.outer(coeffs[t] * dR_draw, np.ones(self.M + 2)) * X
                update_count += 1
                if max_updates and update_count >= max_updates:
                    break
            if np.linalg.norm(gradients) > 0:
                self.theta += self.learning_rate * gradients
            self.loss_history.append(-np.mean(returns_array))
            if verbose:
                objective_value = self.evaluate_objective(returns_array)
                print(f"Epoch {epoch + 1}/{epochs}: objective={objective_value:.6f}, updates={update_count}")
            if max_updates and update_count >= max_updates:
                break

    def evaluate_objective(self, returns: np.ndarray) -> float:
        returns = np.asarray(returns, dtype=float)
        if self.objective == "sharpe":
            return sharpe_ratio(returns)
        if self.objective == "sterling":
            return sterling_ratio(returns)
        if self.objective == "calmar":
            return calmar_ratio(returns, T=len(returns))
        return 0.0

    def generate_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        signals = []
        positions_prev = np.zeros(self.n_assets, dtype=float)
        prev_raw = np.zeros(self.n_assets, dtype=float)
        for t in range(self.M, len(returns)):
            X = self._build_features(returns, t, prev_raw)
            raw = np.einsum("ij,ij->i", X, self.theta)
            signal = self._activation(raw)
            positions = self._compute_positions(signal)
            signals.append(positions)
            prev_raw = raw
            positions_prev = positions
        index = returns.index[self.M :]
        return pd.DataFrame(signals, index=index, columns=self.asset_names)

    def simulate(
        self,
        returns: pd.DataFrame,
        start_at: int | None = None,
        stop_loss_n: float | None = None,
    ) -> pd.DataFrame:
        if start_at is None:
            start_at = self.M
        if len(returns) <= start_at:
            raise ValueError(
                "Simulation data must include at least M warmup rows before start_at."
            )
        positions_prev = np.zeros(self.n_assets, dtype=float)
        prev_raw = np.zeros(self.n_assets, dtype=float)
        records = []
        returns_series: list[float] = []
        for t in range(start_at, len(returns)):
            X = self._build_features(returns, t, prev_raw)
            raw = np.einsum("ij,ij->i", X, self.theta)
            signal = self._activation(raw)
            positions = self._compute_positions(signal)
            period_return = self._portfolio_return(
                positions_prev, positions, returns.iloc[t].values.astype(float)
            )
            returns_series.append(period_return)
            records.append(
                {
                    "date": returns.index[t],
                    "positions": positions,
                    "portfolio_return": period_return,
                    "cumulative_return": np.exp(np.sum(returns_series)) - 1.0,
                }
            )
            if stop_loss_n is not None and len(returns_series) > 1:
                cumret = np.sum(returns_series[:-1])
                vol = np.std(np.asarray(returns_series[:-1]), ddof=0)
                if vol > 0 and cumret / vol <= -stop_loss_n:
                    break
            positions_prev = positions
            prev_raw = raw
        rows: list[dict[str, float]] = []
        for record in records:
            row = {
                "date": record["date"],
                "return": record["portfolio_return"],
                "cumulative_return": record["cumulative_return"],
            }
            row.update({f"pos_{name}": float(record["positions"][i]) for i, name in enumerate(self.asset_names)})
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["return", "cumulative_return"] + [f"pos_{n}" for n in self.asset_names])
        return pd.DataFrame(rows).set_index("date")
