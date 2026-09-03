# SafeLIBERO feasibility analysis

This folder contains the complete exploratory SafeLIBERO feasibility analysis
used before the bidirectional-contract experiment. It is copied out of the
scratch workspace so the scripts, measurements, and generated scene images can
be reviewed together.

## Contents

- `render_demo_endpoints.py` renders the initial and final demonstrated scenes.
- `render_demo_obstacle_overview.py` compares demonstrations with Level-I or
  Level-II obstacle scenes.
- `render_level_comparison.py` renders the combined no-obstacle, Level-I, and
  Level-II comparison.
- `render_demo_A_B.py` visualizes the demonstration start, recovery boundary
  state, and pre-close state.
- `analyze_demo_positions.py` records object positions throughout the demos.
- `analyze_B_collision_coverage.py` measures whether the selected recovery
  boundary states overlap the safety obstacles.
- `images/` contains all generated PNG images and CSV measurements from the
  analysis, including the per-task endpoint images.

The scripts use the repository's existing SafeLIBERO environment and write all
new artifacts back into `images/`.
