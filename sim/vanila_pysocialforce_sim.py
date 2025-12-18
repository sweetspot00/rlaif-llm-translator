"""Minimal pysocialforce example with a simple obstacle map.

The script builds an occupancy-style grid for reference and then converts the
obstacles, start, and target positions into the format pysocialforce expects.
"""

# TODO
# - dynamic obstacle map
# - interrupt simulation and change the prompts dynamically

from __future__ import annotations

from math import ceil, tau
from pathlib import Path
from typing import Optional, Sequence
import json
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import pysocialforce as psf
import logging
from utils.build_obstacles_map import build_obstacle_map 

# Completely ignore DEBUG logs from every logger in the process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def find_your_target(targets_on_the_map:Sequence[Sequence[int]], 
                     agents_start_points: Sequence[Sequence[int]]) -> Sequence[Sequence[int]]:
    """Assign each agent to the nearest target based on Manhattan distance."""
    assigned_targets: list[Sequence[int]] = []
    for start in agents_start_points:
        min_distance = float("inf")
        closest_target: Optional[Sequence[int]] = None
        for target in targets_on_the_map:
            distance = abs(start[0] - target[0]) + abs(start[1] - target[1])
            if distance < min_distance:
                min_distance = distance
                closest_target = target
        if closest_target is not None:
            assigned_targets.append(closest_target)
    return assigned_targets # same size as agents_start_points


def find_your_target_random_sampling(
    obstacle_map: Sequence[Sequence[int]],
    agents_start_points: Sequence[Sequence[int]],
    rng: Optional[np.random.Generator] = None,
) -> Sequence[Sequence[float]]:
    """Assign each agent a random obstacle-free target in world coordinates."""
    obstacle_map = np.asarray(obstacle_map)
    free_cells = np.argwhere(obstacle_map == 0)
    if free_cells.size == 0:
        raise ValueError("No free cells available for target sampling.")

    generator = rng or np.random.default_rng()
    replace = len(free_cells) < len(agents_start_points)
    chosen = generator.choice(len(free_cells), size=len(agents_start_points), replace=replace)

    targets: list[Sequence[float]] = []
    for idx in chosen:
        row, col = free_cells[idx]
        targets.append((col + 0.5, row + 0.5))
    return targets


def occupancy_to_simulation(
    obstacle_map: np.ndarray,
    *,
    start_cells: Optional[Sequence[Sequence[int]]] = None,
    target_cells: Optional[Sequence[Sequence[int]]] = None,
    start_positions: Optional[Sequence[Sequence[float]]] = None,
    target_positions: Optional[Sequence[Sequence[float]]] = None,
    desired_speed: float = 1.2,
    group_probability: float = 0.3,
    max_group_size: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, list[list[float]], list[list[int]]]:
    """Convert obstacle map data into pysocialforce inputs.

    Args:
        obstacle_map: Occupancy grid; 1 marks obstacles, 2/3 optionally mark default start/goal cells.
        start_cells: Optional list of grid (row, col) indices for starting positions.
        target_cells: Optional list of grid (row, col) indices for target positions.
        start_positions: Optional world-coordinate (x, y) starts; overrides start_cells.
        target_positions: Optional world-coordinate (x, y) goals; overrides target_cells.
        desired_speed: Target walking speed in m/s used to seed initial velocities.

    Returns:
        Tuple (initial_state, obstacles, groups) suitable for pysocialforce.Simulator.
        initial_state has shape (N, 6) indexed as [px, py, vx, vy, gx, gy].
    """
    resolution = 1.0  # meters per cell

    def cell_center(indices: Sequence[int]) -> np.ndarray:
        row, col = indices
        return np.array([col * resolution + resolution / 2, row * resolution + resolution / 2])

    map_start_cells = np.argwhere(obstacle_map == 2)
    map_target_cells = np.argwhere(obstacle_map == 3)

    if start_positions is None:
        source = start_cells if start_cells is not None else map_start_cells
        source = np.asarray(source, dtype=int)
        if source.size == 0:
            raise ValueError("No start cells provided or encoded in the map.")
        start_positions = np.vstack([cell_center(idx) for idx in source])
    else:
        start_positions = np.asarray(start_positions, dtype=float)

    if target_positions is None:
        source = target_cells if target_cells is not None else map_target_cells
        source = np.asarray(source, dtype=int)
        if source.size == 0:
            raise ValueError("No target cells provided or encoded in the map.")
        target_positions = np.vstack([cell_center(idx) for idx in source])
    else:
        target_positions = np.asarray(target_positions, dtype=float)

    if start_positions.ndim != 2 or start_positions.shape[1] != 2:
        raise ValueError("start_positions must be an array of (x, y) pairs.")
    if target_positions.ndim != 2 or target_positions.shape[1] != 2:
        raise ValueError("target_positions must be an array of (x, y) pairs.")

    num_agents = start_positions.shape[0]
    if num_agents == 0:
        raise ValueError("At least one start position is required.")

    goal_pool = target_positions
    if goal_pool.shape[0] < num_agents:
        goal_indices = np.arange(num_agents) % goal_pool.shape[0]
        goals = goal_pool[goal_indices]
    else:
        goals = goal_pool[:num_agents]

    directions = goals - start_positions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms > 0)
    zero_mask = norms.squeeze(-1) == 0
    if np.any(zero_mask):
        directions[zero_mask] = np.array([1.0, 0.0])

    velocities = directions * desired_speed
    initial_state = np.hstack([start_positions, velocities, goals])

    obstacle_layer = obstacle_map.copy()
    obstacle_layer[obstacle_map == 2] = 0
    obstacle_layer[obstacle_map == 3] = 0
    if start_cells is not None:
        for row, col in np.asarray(start_cells, dtype=int):
            obstacle_layer[row, col] = 0
    if target_cells is not None:
        for row, col in np.asarray(target_cells, dtype=int):
            obstacle_layer[row, col] = 0

    obstacles: list[list[float]] = []
    rows, cols = np.where(obstacle_layer == 1)
    for row, col in zip(rows, cols):
        x_min = col * resolution
        x_max = (col + 1) * resolution
        y_min = row * resolution
        y_max = (row + 1) * resolution
        obstacles.append([x_min, x_max, y_min, y_max])

    groups: list[list[int]] = []
    if group_probability > 0 and max_group_size > 1 and num_agents > 1:
        generator = rng or np.random.default_rng()
        indices = np.arange(num_agents)
        generator.shuffle(indices)
        remaining = list(indices)
        while len(remaining) >= 2:
            if generator.random() < group_probability:
                upper = min(max_group_size, len(remaining))
                group_size = int(generator.integers(2, upper + 1))
                members = [remaining.pop() for _ in range(group_size)]
                if len(members) > 1:
                    groups.append(sorted(members))
            else:
                remaining.pop()

    return initial_state, obstacles, groups


def polyline_to_segments(points: Sequence[Sequence[float]], closed: bool = False) -> list[list[float]]:
    """Convert a polyline to pysocialforce segments [x0, x1, y0, y1]."""
    if len(points) < 2:
        raise ValueError("Polyline requires at least two points.")

    segments: list[list[float]] = []
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        segments.append([float(x0), float(x1), float(y0), float(y1)])

    if closed:
        x0, y0 = points[-1]
        x1, y1 = points[0]
        segments.append([float(x0), float(x1), float(y0), float(y1)])

    return segments


def approximate_circle_obstacle(
    center: Sequence[float], radius: float, max_segment_length: float
) -> list[list[float]]:
    """Approximate a circular obstacle by chaining linear segments."""
    if radius <= 0 or max_segment_length <= 0:
        raise ValueError("Radius and segment length must be positive.")

    circumference = tau * radius
    n_points = max(8, ceil(circumference / max_segment_length))
    angles = np.linspace(0.0, tau, num=n_points, endpoint=False)
    cx, cy = center
    points = [(cx + radius * np.cos(theta), cy + radius * np.sin(theta)) for theta in angles]
    return polyline_to_segments(points, closed=True)


def sample_positions_avoiding_obstacles(
    obstacle_map: np.ndarray, *, n_samples: int, min_distance: float
) -> np.ndarray:
    """Sample free-space positions on the occupancy grid with minimum spacing."""
    resolution = 1.0
    free_mask = obstacle_map == 0
    free_indices = np.argwhere(free_mask)
    if free_indices.size == 0:
        raise ValueError("No free cells available for sampling.")

    rng = np.random.default_rng()
    selected: list[np.ndarray] = []
    min_dist_sq = (min_distance / resolution) ** 2

    while len(selected) < n_samples:
        candidate = free_indices[rng.integers(0, len(free_indices))]
        if all(np.sum((candidate - np.asarray(p)) ** 2) >= min_dist_sq for p in selected):
            selected.append(candidate)
        elif len(selected) == 0:
            selected.append(candidate)

    positions = np.vstack(
        [np.array([col + 0.5, row + 0.5]) * resolution for row, col in selected]
    )
    return positions


def export_coordinate_log(sim: psf.Simulator, output_path: Path):
    """Persist per-step (x, y) coordinates for each agent."""
    states, _ = sim.get_states()
    out_path = output_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("step,agent,x,y\n")
        for step_idx in range(states.shape[0]):
            for agent_idx in range(states.shape[1]):
                x, y = states[step_idx, agent_idx, :2]
                f.write(f"{step_idx},{agent_idx},{x:.4f},{y:.4f}\n")


def sim_metrics(sim_data: np.ndarray, goal_positions: np.ndarray, goal_threshold: float = 0.5):
    """Compute basic metrics: collision rate, avg time to goal, avg speed, goal reached rate."""
    steps, agents = sim_data.shape[:2]
    positions = sim_data[:, :, :2]
    velocities = sim_data[:, :, 2:4]
    goals = goal_positions.reshape(agents, 2)

    # Collision detection if agents closer than 2*radius (approx 0.7m)
    collisions = 0
    total_pairs = 0
    for i in range(agents):
        for j in range(i + 1, agents):
            dist = np.linalg.norm(positions[:, i, :] - positions[:, j, :], axis=1)
            collisions += np.sum(dist < 0.7)
            total_pairs += steps
    collision_rate = collisions / total_pairs if total_pairs else 0.0

    # Goal reach metrics
    time_to_goal = []
    reached_mask = []
    for i in range(agents):
        dist_traj = np.linalg.norm(positions[:, i, :] - goals[i], axis=1)
        reached = dist_traj < goal_threshold
        reached_mask.append(np.any(reached))
        if np.any(reached):
            time_to_goal.append(np.argmax(reached))
    avg_time_to_goal = float(np.mean(time_to_goal)) if time_to_goal else float("nan")
    goal_reached_rate = np.mean(reached_mask)

    # Avg speed magnitude
    speed = np.linalg.norm(velocities, axis=2)
    avg_speed = float(np.mean(speed))

    return {
        "collision_rate": collision_rate,
        "avg_time_to_goal": avg_time_to_goal,
        "avg_speed": avg_speed,
        "goal_reached_rate": goal_reached_rate,
    }


def draw_obstacles(ax: plt.Axes, obstacle_map: np.ndarray):
    """Render obstacle cells as filled rectangles."""
    height, width = obstacle_map.shape
    for row in range(height):
        for col in range(width):
            if obstacle_map[row, col] == 1:
                rect = Rectangle(
                    (col, row),
                    1.0,
                    1.0,
                    facecolor="#f6e7a1",
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linestyle=":", linewidth=0.3)


def render_visuals(
    obstacle_map: np.ndarray,
    exits: np.ndarray,
    states: np.ndarray,
    output_prefix: Path,
    groups: Optional[list[list[int]]] = None,
):
    """Create a static figure and an animation from recorded states."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.cm.get_cmap("tab10", states.shape[1])

    group_colors = {}
    if groups:
        for idx, grp in enumerate(groups):
            color = cmap(idx % cmap.N)
            for agent_idx in grp:
                group_colors[agent_idx] = color

    # Static plot
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_obstacles(ax, obstacle_map)
    for agent_idx in range(states.shape[1]):
        color = group_colors.get(agent_idx, cmap(agent_idx))
        ax.plot(
            states[:, agent_idx, 0],
            states[:, agent_idx, 1],
            "-o",
            markersize=2,
            color=color,
        )
    ax.scatter(exits[:, 0], exits[:, 1], marker="x", c="green", s=120, linewidths=2, label="exit")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plot_path = output_prefix.parent / f"{output_prefix.name}_plot.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    # Animation
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_obstacles(ax, obstacle_map)
    colors = [group_colors.get(idx, cmap(idx)) for idx in range(states.shape[1])]
    scatters = [
        ax.scatter([], [], color=colors[idx], s=40, zorder=5)
        for idx in range(states.shape[1])
    ]
    trails = [
        ax.plot([], [], "-", color=colors[idx], linewidth=1.5)[0]
        for idx in range(states.shape[1])
    ]
    ax.scatter(exits[:, 0], exits[:, 1], marker="x", c="green", s=120, linewidths=2, label="exit")
    ax.legend(loc="upper right")
    fig.tight_layout()

    def init():
        for scatter in scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for trail in trails:
            trail.set_data([], [])
        return scatters + trails

    def update(frame: int):
        for idx in range(states.shape[1]):
            trails[idx].set_data(states[: frame + 1, idx, 0], states[: frame + 1, idx, 1])
            scatters[idx].set_offsets([states[frame, idx, 0], states[frame, idx, 1]])
        return scatters + trails

    ani = mpl_animation.FuncAnimation(
        fig, update, frames=states.shape[0], init_func=init, interval=80, blit=True
    )
    ani.save(output_prefix.with_suffix(".gif"), writer="pillow", fps=12)
    plt.close(fig)
    
def calculate_evaluation_metrics(sim_data_path: Path, 
                                 config: Path,
                                 dt: float = 0.4, 
                                 obstacle_map: Optional[np.ndarray] = None,
                                 gate: Optional[tuple[float, float, float, float]] = None):
    from utils.evaluation import evaluate_simulation
    metrics = evaluate_simulation(sim_data_path, 
                                  config=config, 
                                  obstacle_mask=(obstacle_map == 1), 
                                  dt=dt, 
                                  gate=gate)
    return metrics

def run_sim(n_agents: int = 10, 
            min_distance: float = 0.2, 
            desired_speed: float = 1.2, 
            need_random_targets: bool = True,
            config_path: Path = Path("test.toml"),
            save_file_name: str = "vanila_sim",
            use_default_map: bool = True,
            obstacle_map: Optional[np.ndarray] = None,
            target_positions: Optional[np.ndarray] = None):
    
    if use_default_map or obstacle_map is None or target_positions is None:
        obstacle_map, target_positions = build_obstacle_map()

    logger.info("Obstacle map shape: %s", obstacle_map.shape)
    np.random.seed(42)
    start_positions = sample_positions_avoiding_obstacles(obstacle_map, n_samples=n_agents, min_distance=min_distance)
    
    # for non emergency case
    if str(need_random_targets).lower() == "true":
        agents_targets = find_your_target_random_sampling(
            obstacle_map,
            agents_start_points=start_positions
        )
    else:  # for emergency case with exits 
        logger.debug("use predefined exits as targets")
        agents_targets = find_your_target(targets_on_the_map=target_positions, agents_start_points=start_positions)
        logger.debug(f"agents_targets: {np.unique(agents_targets, axis=0)}")
    initial_state, obstacles, groups = occupancy_to_simulation(
        obstacle_map,
        start_positions=start_positions,
        target_positions=agents_targets,
        desired_speed=desired_speed
    )
    sim = psf.Simulator(
        initial_state,
        # groups=groups if groups else None, # TODO: disable group behavior for now
        obstacles=obstacles,
        config_file=str(config_path),
    )
    sim.step(n=150)
    export_coordinate_log(sim, Path(f"sim/{save_file_name}.csv"))
    logger.info("Simulation completed.")
    states, _ = sim.get_states()
    # metrics = sim_metrics(states, initial_state[:, 4:6])
    # logger.info("Simulation metrics: %s", metrics)
    # metrics_path = Path(f"sim/{save_file_name}.json")
    # metrics_path.write_text(json.dumps(metrics, indent=2))
    
    """" use evaluation module"""
    metrics = calculate_evaluation_metrics(
        sim_data_path=Path(f"sim/{save_file_name}.csv"),
        config=config_path,
        obstacle_map=obstacle_map,
        dt=0.4,
        gate=(10,15,0,1)
    )
    logger.info("Evaluation metrics: %s", metrics)
    metrics_path = Path(f"sim/{save_file_name}.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    
    render_visuals(
        obstacle_map,
        target_positions,
        states,
        Path(f"sim/{save_file_name}"),
        groups=groups,
    )
    visual_path = Path(f"sim/{save_file_name}")
    return metrics, states, groups, visual_path
