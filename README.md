# RAS598 Assignment 3: 3D Discrete Bayes Filter (Histogram Filter)

## Overview

A 3D Discrete Bayes Filter (Histogram Filter) for robot localization in a known 16×16 m cave environment. The belief is maintained as a probability distribution over the state space (x, y, θ), represented as an 80×80×72 numpy array. The robot starts at [−7, −7, 90°] and localizes itself using noisy odometry and fiducial landmark observations.

---

## Demo

![Bayes Filter Demo](figure%20and%20gif/ras598-Assignment3-GIF.gif)

---

## 7.1 Technical Implementation (15 Points)

### Coordinate Mapping (2 pts)

**`real_to_grid(x, y, θ)`** maps continuous world coordinates (±8 m) to discrete grid indices. Row index 0 corresponds to y = +8 m (top of the map), matching the `np.flipud` convention used in `_publish_costmap` so the OccupancyGrid displays correctly in RViz:

```
ix  = int((x + 8.0) / resolution)      # 0 → leftmost column
iy  = int((8.0 - y) / resolution)      # 0 → top row (highest y)
ith = int(θ % 360 / theta_res) % theta_dim
```

**`grid_to_real()`** returns fully vectorized (rx, ry, rth) arrays over all grid cells for efficient numpy-based likelihood computation without Python loops.

---

### Motion Model (5 pts)

The Turn-Go-Turn odometry decomposition extracts three motion primitives from consecutive odometry messages:

- **δ_rot1** — initial rotation to face the direction of travel
- **δ_trans** — forward translation
- **δ_rot2** — final rotation to reach the new heading

Each is applied by rolling the belief array along the appropriate axis. For the translation step, each θ-slice is shifted independently based on its heading direction (`cos θ`, `sin θ`). Gaussian diffusion (σ = 0.5) is applied after each prediction to model process noise.

**Belief moving correctly with the robot's motion:**

| Initial Position | After Motion (1–2 Landmarks Seen) |
|---|---|
| ![Initial](figure%20and%20gif/initialisiing_position.png) | ![1-2 Landmarks](figure%20and%20gif/1-2%20landmark.png) |

The belief cloud (light blue in Stage, dark blob in RViz) tracks the robot as it moves through the environment.

---

### Measurement Model (5 pts)

For each fiducial detection, the expected range and bearing from every grid cell to the known landmark position are computed vectorially. A Gaussian likelihood (`σ_r = 0.5 m`, `σ_b = 15°`) is applied:

```
likelihood = exp(-0.5 * ((r_meas - r_exp) / σ_r)²) ×
             exp(-0.5 * ((b_meas - b_exp) / σ_b)²)
```

The belief is multiplied element-wise and renormalized (Bayes update: Bel(x) = η · p(z|x) · Bel̄(x)).

**Landmark sightings collapsing the cloud:**

![After 4 Landmarks](figure%20and%20gif/4%20landmark.png)

After seeing 4 landmarks, the belief collapses into a tight, distinct peak near the robot's true position.

---

### Convergence (2 pts)

After observing 2–3 landmarks, the peak of the belief converges to within **0.6 m** of the Ground Truth. As shown above, the costmap blob (RViz, dark region) aligns closely with the green Ground Truth path after multiple landmark sightings.

---

### Explanation (1 pt)

**Why does the probability distribution take different shapes?**

At initialization, the belief is a tight **circular Gaussian blob** at the known starting pose (−7, −7) — the robot knows exactly where it starts.

After motion without landmark observations, the prediction step shifts each θ-slice in its respective heading direction and applies Gaussian diffusion. The projected x–y belief spreads into an **elongated ellipse** aligned with the direction of travel, since translational uncertainty grows faster along the motion axis than perpendicular to it. Angular uncertainty also broadens over time.

After a single landmark observation, the likelihood field constrains possible positions to a **ring-shaped arc** at the measured range from the landmark — the robot knows how far the landmark is but not from which direction it arrived. The bearing measurement narrows this arc further into a crescent or partial arc.

After 2–3 observations from different landmark positions, these arcs **intersect**, collapsing the distribution into a compact peak (or multi-modal peaks if ambiguity remains) near the true pose.

![Initial Belief](figure%20and%20gif/initialisiing_position.png)
*Initial: tight Gaussian blob at starting pose*

![After 1-2 Landmarks](figure%20and%20gif/1-2%20landmark.png)
*After motion + 1–2 landmarks: elliptical spread beginning to collapse*

![After 4 Landmarks](figure%20and%20gif/4%20landmark.png)
*After 4 landmarks: concentrated peak near ground truth*

---

## Bonus: Kidnapping and Recovery

When the robot is teleported to a new location (kidnapping), the belief remains concentrated at the old estimated position — a fundamental limitation of standard histogram filters, which have no recovery mechanism.

![Kidnapped](figure%20and%20gif/kidnap.png)
*Kidnapping: belief (dark blob) stays at old estimate while robot is moved*

![Recovery](figure%20and%20gif/recovery.png)
*Partial recovery: after driving near landmarks again, the filter begins to relocalize*

Full kidnapping recovery requires particle filters with random particle injection; a standard Bayes filter requires re-initialization.

---

## Running the Filter

### Prerequisites
```bash
pip3 install --user --break-system-packages "scipy>=1.13"
cd ~/ros2_ws
colcon build --packages-select ras598_assignment_3
source install/setup.bash
```

### Launch

**Terminal 1 — Stage + Filter + RViz:**
```bash
ROS_DOMAIN_ID=9 ros2 launch ras598_assignment_3 bayes_launch.py
```

**Terminal 2 — Teleop:**
```bash
ROS_DOMAIN_ID=9 ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

## Package Structure

```
ras598_assignment_3/
├── figure and gif/
│   ├── initialisiing_position.png
│   ├── 1-2 landmark.png
│   ├── 4 landmark.png
│   ├── kidnap.png
│   ├── recovery.png
│   └── ras598-Assignment3-GIF.gif
├── launch/
│   └── bayes_launch.py
├── ras598_assignment_3/
│   ├── __init__.py
│   └── bayes_boilerplate.py
├── bayes_boilerplate.py
├── bayes.rviz
├── package.xml
├── setup.py
└── README.md
```
