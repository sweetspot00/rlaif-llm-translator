1. Obstacle map generation \
From complicated obstacle png `downloads/google_maps/obstacles` generate simplified obstacle png `downloads/google_maps/simplified_obstacles` and anchored npz  `preprocess/pysfm_obstacles_meter_close_shape`. Using `experiments.ipynb`

2. Context dataset generation \
Generate `context.jsonl` dataset using simplified obstacle png `downloads/google_maps/simplified_obstacles` using `utils/generate_context.py`

3. Dataset proprocess \
Sample data for each line of the `context.jsonl` -> `preprocess/preprocessed_scene/line_number_scene.json`

4. Simulation
`sim/sim.py`by context line number \
TODO: Multi-thread sim

5. Trajectory plot
Need simulated result `sim/results/simulations` and scene(preprocessed data) `preprocess/preprocessed_scene`


google map 的h是pixel -> world(m)

gt's H is world(m) -> pixel