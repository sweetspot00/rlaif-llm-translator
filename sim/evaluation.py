"""
Trajectory evaluation utilities for social force simulations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # fallback if plotting backend missing


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _angular_velocity(positions: np.ndarray, dt: float) -> np.ndarray:
    vel = np.diff(positions, axis=0) / dt
    dv = np.diff(vel, axis=0) / dt
    ang = np.cross(vel[:-1], dv, axis=-1)
    return ang


def _histogram_density(positions: np.ndarray, bounds: Tuple[float, float, float, float], bins: int) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    hist, _, _ = np.histogram2d(
        positions[..., 0].ravel(),
        positions[..., 1].ravel(),
        bins=bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


MetricFunc = Callable[[np.ndarray, np.ndarray, dict], float]


@dataclass
class TrajectoryEvaluator:
    collision_distance: float = 0.3
    goal_radius: float = 0.5
    goal_tolerance: float = 0.0
    density_bins: int = 20
    dt: float = 1.0
    metrics: Iterable[str] = field(
        default_factory=lambda: (
            "ADE",
            "FDE",
            "EMD",
            "Kinem",
            "CollisionRate",
            "VD",
            "DDS",
            "GoalRate",
            "Hausdorff",
        )
    )
    result: Dict[str, float] = field(default_factory=dict, init=False)

    # ---- metric implementations ----
    def _metric_ade(self, pred: np.ndarray, gt: np.ndarray, _: dict) -> float:
        return float(np.linalg.norm(pred - gt, axis=-1).mean())

    def _metric_fde(self, pred: np.ndarray, gt: np.ndarray, _: dict) -> float:
        return float(np.linalg.norm(pred[-1] - gt[-1], axis=-1).mean())

    def _metric_emd(self, pred: np.ndarray, gt: np.ndarray, _: dict) -> float:
        cost = _pairwise_distances(pred[-1], gt[-1])
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(cost[row_ind, col_ind].mean())

    def _metric_kinem(self, pred: np.ndarray, gt: np.ndarray, _: dict) -> float:
        speed_pred = np.linalg.norm(np.diff(pred, axis=0) / self.dt, axis=-1)
        speed_gt = np.linalg.norm(np.diff(gt, axis=0) / self.dt, axis=-1)
        return float(np.abs(speed_pred - speed_gt).mean())

    def _metric_collision_rate(self, pred: np.ndarray, _: Optional[np.ndarray], __: dict) -> float:
        T = pred.shape[0]
        collisions = 0
        for t in range(T):
            dists = _pairwise_distances(pred[t], pred[t])
            mask = dists < self.collision_distance
            np.fill_diagonal(mask, False)
            if mask.any():
                collisions += 1
        return collisions / T

    def _metric_goal_rate(self, pred: np.ndarray, _: Optional[np.ndarray], ctx: dict) -> float:
        goals = ctx.get("goals")
        if goals is None:
            return np.nan
        final_dist = np.linalg.norm(pred[-1] - goals, axis=-1)
        tol = self.goal_radius + self.goal_tolerance
        return float(np.mean(final_dist <= tol))

    def _metric_vd(self, pred: np.ndarray, gt: np.ndarray, _: dict) -> float:
        ang_pred = _angular_velocity(pred, self.dt)
        ang_gt = _angular_velocity(gt, self.dt)
        if ang_pred.shape != ang_gt.shape or ang_pred.size == 0:
            return 0.0
        return float(np.abs(ang_pred - ang_gt).mean())

    def _metric_dds(self, pred: np.ndarray, gt: np.ndarray, ctx: dict) -> float:
        bounds = ctx.get("bounds")
        if bounds is None:
            all_pos = np.concatenate([pred.reshape(-1, 2), gt.reshape(-1, 2)], axis=0)
            x_min, y_min = all_pos.min(axis=0)
            x_max, y_max = all_pos.max(axis=0)
            pad_x = (x_max - x_min) * 0.05 + 1e-6
            pad_y = (y_max - y_min) * 0.05 + 1e-6
            bounds = (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)
        hist_pred = _histogram_density(pred, bounds, self.density_bins)
        hist_gt = _histogram_density(gt, bounds, self.density_bins)
        return float(np.minimum(hist_pred, hist_gt).sum())

    def _hausdorff(self, a: np.ndarray, b: np.ndarray) -> float:
        """Symmetric Hausdorff distance between two point sets (M,2) and (K,2)."""
        dists = _pairwise_distances(a, b)
        d_ab = dists.min(axis=1).max() if dists.size else 0.0
        d_ba = dists.min(axis=0).max() if dists.size else 0.0
        return max(d_ab, d_ba)

    def _metric_hausdorff(self, pred: np.ndarray, gt: Optional[np.ndarray], _: dict) -> float:
        if gt is not None:
            a = pred.reshape(-1, 2)
            b = gt.reshape(-1, 2)
            return float(self._hausdorff(a, b))
        # No GT: use pairwise agent trajectories to report the closest pair (safety)
        T, N, _ = pred.shape
        if N < 2:
            return np.nan
        best = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                a = pred[:, i, :]
                b = pred[:, j, :]
                h = self._hausdorff(a, b)
                if h < best:
                    best = h
        return float(best)

    _metric_map = {
        "ADE": _metric_ade,
        "FDE": _metric_fde,
        "EMD": _metric_emd,
        "Kinem": _metric_kinem,
        "CollisionRate": _metric_collision_rate,
        "GoalRate": _metric_goal_rate,
        "VD": _metric_vd,
        "DDS": _metric_dds,
        "Hausdorff": _metric_hausdorff,
    }

    def evaluate(
        self,
        pred: np.ndarray,
        gt: Optional[np.ndarray] = None,
        goals: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, float]:
        if pred.shape[0] < 2:
            raise ValueError("Need at least two timesteps for evaluation.")
        if gt is not None and pred.shape != gt.shape:
            raise ValueError("pred and gt must have the same shape (T, N, 2).")

        ctx = {"goals": goals, "bounds": bounds}
        out: Dict[str, float] = {}

        requires_gt = {
            "ADE": True,
            "FDE": True,
            "EMD": True,
            "Kinem": True,
            "CollisionRate": False,
            "GoalRate": False,  # depends on goals
            "VD": True,
            "DDS": True,
            "Hausdorff": False,  # can run pred-only (pairwise) or pred-vs-gt
        }

        for name in self.metrics:
            func = self._metric_map.get(name)
            if func is None:
                continue
            if requires_gt.get(name, False) and gt is None:
                out[name] = np.nan
                continue
            out[name] = func(self, pred, gt, ctx)

        self.result = out
        return out

    # ---- visualization ----
    def draw_flow_heatmap(
        self,
        traj: np.ndarray,
        *,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        bins: int = 50,
        save_path: Optional[str] = None,
    ):
        """
        Draw a flow heatmap (average velocity magnitude per cell) for a trajectory.

        Args:
            traj: array of shape (T, N, 2)
            bounds: optional (x_min, x_max, y_min, y_max); inferred if None
            bins: number of bins per axis
            save_path: optional path to save the figure
        Returns:
            (fig, ax) if matplotlib is available; otherwise None
        """
        if plt is None:
            raise RuntimeError("matplotlib is not available; cannot draw heatmap.")
        if traj.shape[0] < 2:
            raise ValueError("Need at least two timesteps to compute flow.")

        pos = traj[:-1]  # (T-1, N, 2)
        vel = np.diff(traj, axis=0) / self.dt  # (T-1, N, 2)
        speed = np.linalg.norm(vel, axis=-1)  # (T-1, N)

        if bounds is None:
            all_pos = pos.reshape(-1, 2)
            x_min, y_min = all_pos.min(axis=0)
            x_max, y_max = all_pos.max(axis=0)
            pad_x = (x_max - x_min) * 0.05 + 1e-6
            pad_y = (y_max - y_min) * 0.05 + 1e-6
            bounds = (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)

        x_min, x_max, y_min, y_max = bounds
        # Average speed per bin using histogram weights
        H_count, xedges, yedges = np.histogram2d(
            pos[..., 0].ravel(),
            pos[..., 1].ravel(),
            bins=bins,
            range=[[x_min, x_max], [y_min, y_max]],
        )
        H_speed, _, _ = np.histogram2d(
            pos[..., 0].ravel(),
            pos[..., 1].ravel(),
            bins=bins,
            range=[[x_min, x_max], [y_min, y_max]],
            weights=speed.ravel(),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            H_avg = np.divide(H_speed, H_count, out=np.zeros_like(H_speed), where=H_count > 0)

        fig, ax = plt.subplots()
        im = ax.imshow(
            H_avg.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto",
            cmap="inferno",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Flow heatmap (avg speed)")
        fig.colorbar(im, ax=ax, label="m/s")

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig, ax

    def draw_density_map(
        self,
        traj: np.ndarray,
        *,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        bins: int = 50,
        save_path: Optional[str] = None,
    ):
        """
        Draw a density map (occupancy count) for a trajectory.

        Args:
            traj: array of shape (T, N, 2)
            bounds: optional (x_min, x_max, y_min, y_max); inferred if None
            bins: number of bins per axis
            save_path: optional path to save the figure
        Returns:
            (fig, ax) if matplotlib is available; otherwise None
        """
        if plt is None:
            raise RuntimeError("matplotlib is not available; cannot draw density map.")

        pos = traj.reshape(-1, 2)
        if bounds is None:
            x_min, y_min = pos.min(axis=0)
            x_max, y_max = pos.max(axis=0)
            pad_x = (x_max - x_min) * 0.05 + 1e-6
            pad_y = (y_max - y_min) * 0.05 + 1e-6
            bounds = (x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y)

        x_min, x_max, y_min, y_max = bounds
        hist, xedges, yedges = np.histogram2d(
            pos[:, 0],
            pos[:, 1],
            bins=bins,
            range=[[x_min, x_max], [y_min, y_max]],
        )

        fig, ax = plt.subplots()
        im = ax.imshow(
            hist.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Density map (occupancy)")
        fig.colorbar(im, ax=ax, label="count")

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig, ax
