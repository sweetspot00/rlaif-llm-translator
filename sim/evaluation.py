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
    min_distance: float = 0.35
    collision_distance: float = 0.7
    goal_radius: float = 5
    goal_tolerance: float = 10
    density_bins: int = 20
    dt: float = 0.4 # step_width in seconds, align to pysocialforce default
    group_cohesion_threshold: float = 5
    group_stick_min_percentage: float = 0.5
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
            "goal_achieve_as_long_as_pass_by",
            # "Hausdorff",
            "SocialDistanceViolations",
            "towards_event_achieve_rate",
            "away_event_achieve_rate",
            "group_stick_together",
            "group_goal_achievement",
        )
    )
    result: Dict[str, float] = field(default_factory=dict, init=False)

    def _hungarian_match(
        self, pred_final: np.ndarray, gt_final: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Match GT agents to predicted agents via Hungarian assignment on final positions.

        Returns:
            (pred_idx, gt_idx) arrays of equal length K, where K=min(N_pred, N_gt).
        """
        cost = _pairwise_distances(pred_final, gt_final)
        pred_idx, gt_idx = linear_sum_assignment(cost)
        return pred_idx.astype(int), gt_idx.astype(int)

    def _align_by_hungarian_final(
        self, pred: np.ndarray, gt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reorder/trim (pred, gt) so agent dimension is aligned by Hungarian matching.

        Matching is computed on final positions only, producing a consistent mapping
        across all timesteps.

        Returns:
            (pred_aligned, gt_aligned, perm) where perm maps pred agent index -> gt agent index.
            If N_pred != N_gt, arrays are trimmed to K=min(N_pred, N_gt) and perm is length K.
        """
        if pred.ndim != 3 or gt.ndim != 3 or pred.shape[2] != 2 or gt.shape[2] != 2:
            raise ValueError("pred and gt must have shape (T, N, 2).")
        pred_idx, gt_idx = self._hungarian_match(pred[-1], gt[-1])
        pred_aligned = pred[:, pred_idx, :]
        gt_aligned = gt[:, gt_idx, :]
        perm = gt_idx.copy()
        return pred_aligned, gt_aligned, perm

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

    def _metric_collision_rate(self, pred: np.ndarray, *_): # Fraction of pairs colliding per timestep:
        T, N, _ = pred.shape
        thr = self.collision_distance
        total = 0
        coll = 0

        for t in range(T):
            X = pred[t]  # (N,2)

            # OPTIONAL: drop invalid agents if you have padding/NaNs
            valid = np.isfinite(X).all(axis=1)
            X = X[valid]
            n = X.shape[0]
            if n < 2:
                continue

            d = _pairwise_distances(X, X)
            iu = np.triu_indices(n, k=1)   # unique pairs only
            pair_d = d[iu]

            total += pair_d.size
            coll += np.sum(pair_d < thr)

        return (coll / total) if total > 0 else 0.0

    def _metric_social_distance_violations(self, pred: np.ndarray, _: Optional[np.ndarray], __: dict) -> float:
        T = pred.shape[0]
        violations = 0
        total_pairs = 0
        for t in range(T):
            dists = _pairwise_distances(pred[t], pred[t])
            mask = dists < self.collision_distance + self.min_distance
            np.fill_diagonal(mask, False)
            violations += np.sum(mask)
            total_pairs += (pred.shape[1] * (pred.shape[1] - 1))  # total pairs at this timestep
        if total_pairs == 0:
            return 0.0
        print(f"Social Distance Violations: {violations} out of {total_pairs} pairs.")
        return violations / total_pairs

    def _metric_goal_rate(self, pred: np.ndarray, _: Optional[np.ndarray], ctx: dict) -> float:
        pred_goal = ctx.get("pred_scenario", pred)
        success = self._goal_success_mask(pred_goal, ctx.get("goals"))
        if success is None:
            return np.nan
        groups = ctx.get("groups") or []
        if groups:
            N = pred_goal.shape[1]
            members = sorted({m for grp in groups for m in grp if 0 <= m < N})
            if members:
                return float(np.mean(success[members]))
        return float(np.mean(success))

    def _goal_success_mask(self, pred: np.ndarray, goals: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if goals is None:
            return None
        goals_arr = np.asarray(goals, dtype=float)
        if goals_arr.ndim != 2 or goals_arr.shape[1] != 2:
            return None
        tol = self.goal_radius + self.goal_tolerance
        final_pos = pred[-1]
        if goals_arr.shape[0] == final_pos.shape[0]:
            dists = np.linalg.norm(final_pos - goals_arr, axis=-1)
        else:
            dists = _pairwise_distances(final_pos, goals_arr).min(axis=1)
        return dists <= tol

    def _metric_goal_pass_by(self, pred: np.ndarray, _: Optional[np.ndarray], ctx: dict) -> float:
        pred_goal = ctx.get("pred_scenario", pred)
        goals = ctx.get("goals")
        if goals is None:
            return np.nan
        goals_arr = np.asarray(goals, dtype=float)
        if goals_arr.ndim != 2 or goals_arr.shape[1] != 2:
            return np.nan
        tol = self.goal_radius + self.goal_tolerance
        T, N, _ = pred_goal.shape
        if goals_arr.shape[0] == N:
            # Per-agent goals: min distance along trajectory to each agent's goal
            dists = np.linalg.norm(pred_goal - goals_arr[None, :, :], axis=-1)
            min_dist = dists.min(axis=0)
        else:
            # Shared goals: min distance to any goal over all timesteps
            pred_flat = pred_goal.reshape(T * N, 2)
            dists = _pairwise_distances(pred_flat, goals_arr).reshape(T, N, -1)
            min_dist = dists.min(axis=(0, 2))
        return float(np.mean(min_dist <= tol))

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

    def _metric_event_direction(
        self,
        pred: np.ndarray,
        _: Optional[np.ndarray],
        ctx: dict,
        *,
        towards: bool,
    ) -> float:
        direction = ctx.get("towards_event")
        if direction is None or direction == "random":
            return np.nan
        if towards and direction is not True:
            return np.nan
        if not towards and direction is not False:
            return np.nan

        event_center = ctx.get("event_center")
        if event_center is None:
            return np.nan
        center = np.asarray(event_center, dtype=float).reshape(-1)
        if center.size < 2:
            return np.nan
        center = center[:2]
        start = pred[0]
        end = pred[-1]
        start_dist = np.linalg.norm(start - center, axis=1)
        end_dist = np.linalg.norm(end - center, axis=1)
        if start_dist.size == 0:
            return np.nan
        if towards:
            moving = end_dist <= (start_dist + 1e-6)
        else:
            moving = end_dist >= (start_dist - 1e-6)
        return float(np.mean(moving))

    def _metric_towards_event(self, pred: np.ndarray, gt: Optional[np.ndarray], ctx: dict) -> float:
        return self._metric_event_direction(pred, gt, ctx, towards=True)

    def _metric_away_event(self, pred: np.ndarray, gt: Optional[np.ndarray], ctx: dict) -> float:
        return self._metric_event_direction(pred, gt, ctx, towards=False)

    def _metric_group_stick(self, pred: np.ndarray, _: Optional[np.ndarray], ctx: dict) -> float:
        pred_group = ctx.get("pred_scenario", pred)
        groups = ctx.get("groups") or []
        if not groups:
            return np.nan
        T, N, _ = pred_group.shape
        if N == 0:
            return np.nan
        threshold = self.group_stick_min_percentage
        max_pair_distance = self.group_cohesion_threshold
        group_scores: list[bool] = []
        for grp in groups:
            members = [m for m in grp if 0 <= m < N]
            if len(members) < 2:
                continue
            traj = pred_group[:, members, :]  # (T, G, 2)
            per_timestep_max: list[float] = []
            for t in range(T):
                d = _pairwise_distances(traj[t], traj[t])
                if d.size == 0:
                    continue
                iu = np.triu_indices(len(members), k=1)
                per_timestep_max.append(float(np.max(d[iu])))
            if not per_timestep_max:
                continue
            stick = np.asarray(per_timestep_max) < max_pair_distance
            group_scores.append(bool(np.mean(stick) >= threshold))
        if not group_scores:
            return np.nan
        return float(np.mean(group_scores))

    def _metric_group_goal(self, pred: np.ndarray, _: Optional[np.ndarray], ctx: dict) -> float:
        pred_goal = ctx.get("pred_scenario", pred)
        groups = ctx.get("groups") or []
        goals = ctx.get("goals")
        if goals is None or not groups:
            return np.nan
        success_mask = self._goal_success_mask(pred_goal, goals)
        if success_mask is None:
            return np.nan
        N = pred_goal.shape[1]
        group_hits: list[bool] = []
        for grp in groups:
            members = [m for m in grp if 0 <= m < N]
            if not members:
                continue
            group_hits.append(bool(np.all(success_mask[members])))
        if not group_hits:
            return np.nan
        return float(np.mean(group_hits))

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
        "goal_achieve_as_long_as_pass_by": _metric_goal_pass_by,
        "VD": _metric_vd,
        "DDS": _metric_dds,
        # "Hausdorff": _metric_hausdorff,
        "SocialDistanceViolations": _metric_social_distance_violations,
        "towards_event_achieve_rate": _metric_towards_event,
        "away_event_achieve_rate": _metric_away_event,
        "group_stick_together": _metric_group_stick,
        "group_goal_achievement": _metric_group_goal,
    }

    def _compute_distance_scale(
        self, pred: np.ndarray, gt: Optional[np.ndarray], bounds: Optional[Tuple[float, float, float, float]]
    ) -> float:
        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
        else:
            pts = [pred.reshape(-1, 2)]
            if gt is not None:
                pts.append(gt.reshape(-1, 2))
            all_pos = np.concatenate(pts, axis=0)
            x_min, y_min = np.nanmin(all_pos, axis=0)
            x_max, y_max = np.nanmax(all_pos, axis=0)
        diag = np.linalg.norm([x_max - x_min, y_max - y_min])
        if not np.isfinite(diag) or diag <= 0:
            diag = np.linalg.norm(pred[-1] - pred[0], axis=1).max(initial=1.0)
        return float(max(diag, 1e-6))

    def _scale_metric_value(self, name: str, value: float, ctx: dict) -> float:
        if not np.isfinite(value):
            return value
        distance_scale = ctx["distance_scale"]
        if name in {"ADE", "FDE", "EMD"}:
            return float(np.clip(value / distance_scale, 0.0, 1.0))
        if name == "Kinem":
            speed_scale = distance_scale / self.dt
            return float(np.clip(value / speed_scale, 0.0, 1.0))
        if name == "VD":
            ang_scale = (distance_scale / self.dt) ** 2
            return float(np.clip(value / ang_scale, 0.0, 1.0))
        if name in {
            "GoalRate",
            "goal_achieve_as_long_as_pass_by",
            "CollisionRate",
            "DDS",
            "SocialDistanceViolations",
            "towards_event_achieve_rate",
            "away_event_achieve_rate",
            "group_stick_together",
            "group_goal_achievement",
        }:
            return float(np.clip(value, 0.0, 1.0))
        # Fallback normalization keeps the metric bounded without changing ordering.
        return float(np.clip(value / (value + distance_scale), 0.0, 1.0))

    def evaluate(
        self,
        pred: np.ndarray,
        gt: Optional[np.ndarray] = None,
        goals: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        towards_event: Optional[str | bool] = None,
        groups: Optional[Iterable[Iterable[int]]] = None,
        event_center: Optional[np.ndarray] = None,
        *,
        match_agents: bool = True,
    ) -> Dict[str, float]:
        if pred.shape[0] < 2:
            raise ValueError("Need at least two timesteps for evaluation.")
        if gt is not None and (pred.ndim != 3 or gt.ndim != 3 or pred.shape[2] != 2 or gt.shape[2] != 2):
            raise ValueError("pred and gt must have shape (T, N, 2).")
        if gt is not None and pred.shape[0] != gt.shape[0]:
            raise ValueError("pred and gt must have the same number of timesteps (T).")

        pred_for_goals = pred
        perm: Optional[np.ndarray] = None
        if match_agents and gt is not None:
            pred, gt, perm = self._align_by_hungarian_final(pred, gt)

        distance_scale = self._compute_distance_scale(pred, gt, bounds)

        def _normalize_towards_event(value: object) -> str | bool | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "towards", "toward"}:
                    return True
                if lowered in {"false", "away"}:
                    return False
                if lowered == "random":
                    return "random"
            return None

        ctx = {
            "goals": goals,
            "bounds": bounds,
            "perm": perm,
            "event_center": event_center,
            "towards_event": _normalize_towards_event(towards_event),
            "groups": [list(map(int, g)) for g in groups] if groups is not None else None,
            "distance_scale": distance_scale,
            "pred_scenario": pred_for_goals,
        }
        out: Dict[str, float] = {}

        requires_gt = {
            "ADE": True,
            "FDE": True,
            "EMD": True,
            "Kinem": True,
            "CollisionRate": False,
            "GoalRate": False,  # depends on goals
            "goal_achieve_as_long_as_pass_by": False,
            "VD": True,
            "DDS": True,
            "Hausdorff": False,  # can run pred-only (pairwise) or pred-vs-gt
            "SocialDistanceViolations": False,
            "towards_event_achieve_rate": False,
            "away_event_achieve_rate": False,
            "group_stick_together": False,
            "group_goal_achievement": False,
        }

        for name in self.metrics:
            func = self._metric_map.get(name)
            if func is None:
                continue
            if requires_gt.get(name, False) and gt is None:
                out[name] = np.nan
                continue
            raw_value = func(self, pred, gt, ctx)
            out[name] = self._scale_metric_value(name, raw_value, ctx)

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
