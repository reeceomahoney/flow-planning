# Obstacle perception

Live obstacle tracking and RGB-D point-cloud estimation for the Piper setup.

## Run

The `rick` setup is already configured, so normal use only requires:

```bash
hardware/obstacle_perception/track_obstacle.sh
```

This loads the saved camera crop, calibrations, and obstacle references; starts both cameras, SAM2 tracking, point-cloud estimation, and robot-frame projection; and serves the dashboard at `http://rick:8080`.

## Optional setup and recalibration

| Command | Purpose | Latest output |
| --- | --- | --- |
| `setup.sh` | Creates the local Python environment, installs dependencies, downloads SAM2, and runs preflight checks. | `.venv/` |
| `select_top_roi.sh` | Selects the overhead-camera crop used for tracking. Run again only if the working view changes. | `calibration/top_roi.json` |
| `calibrate_cameras.sh` | Estimates the wrist-camera pose relative to the overhead camera. Run again if either camera moves. | `calibration/top_from_left.json` |
| `calibrate_robot_frame.sh` | Estimates the overhead-camera pose relative to the Piper base. Run again if the camera-to-robot geometry changes. | `calibration/base_from_top.json` |
| `select_obstacle.sh` | Records the object references used to initialize tracking in both camera views. Run again when changing the tracked object. | `calibration/targets/` |

The latest accepted outputs are stored at the paths above and loaded automatically by `track_obstacle.sh`. On `rick`, they are already populated and ready to use.

## Files

```text
track_obstacle.py             Coordinates cameras, SAM2 tracking, and live output
point_cloud_stream.py         Fuses RGB-D views and estimates the obstacle in robot coordinates
camera_web.py                 Serves the live browser dashboard and shared state
policy_trajectory_stream.py   Projects recorded robot trajectories into the scene

select_top_roi.py             Selects the working crop of the overhead camera
calibrate_cameras.py          Calibrates the wrist camera relative to the overhead camera
calibrate_robot_frame.py      Calibrates the overhead camera relative to the Piper base
preflight.py                  Checks dependencies, camera access, display, and model availability

setup.sh                      Creates the local environment
*.sh                          Launchers for the matching Python tools
calibration/                  Generated calibration and target files; not committed
runs/                         Generated runtime output; not committed
```
