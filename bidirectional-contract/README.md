# Bidirectional contract

This folder is the PR-sized implementation of one idea:

> The controller owns outward obstacle evasion; the learned policy owns return
> to the task. The controller may intervene only through deviations for which
> recovery supervision was constructed.

It contains the core method and one fixed-scene benchmark—not the exploratory
rendering, plotting, and rollout scripts used to reach it.

This is a first feasibility implementation, not the final method. See
[`README_WHATS_MISSING.md`](README_WHATS_MISSING.md) for the current conceptual
and experimental limitations.

## Contract

Let a demonstration provide end-effector pose
\((p_t^\star, R_t^\star)\). A local frame is constructed from the path tangent
\(\tau_t\):

\[
v_t = \frac{e_z-(e_z^\top\tau_t)\tau_t}
           {\|e_z-(e_z^\top\tau_t)\tau_t\|},\qquad
s_t = \frac{\tau_t\times v_t}{\|\tau_t\times v_t\|}.
\]

The shared intervention alphabet is a discrete set of rays in the upper normal
half-plane and a discrete set of radii:

\[
n_t^j=\cos\theta_j\,s_t+\sin\theta_j\,v_t,
\quad \theta_j\in\{0,\tfrac\pi4,\tfrac\pi2,\tfrac{3\pi}4,\pi\},
\quad r\in\{3,6,9,12\}\text{ cm}.
\]

At training time, a recovery window starts at an offset \(r n_a^j\) from an
anchor \(a\), while demonstration phase continues forward. Its Cartesian
target is

\[
\tilde p_{a+k}=p_{a+k}^\star+r\,\rho(k/T)\,n_{a+k}^j,
\qquad
\rho(u)=1-(10u^3-15u^4+6u^5),
\]

with \(\rho=0\) after the chosen recovery duration. Wrist orientation follows
the demonstration. Sequential IK converts these targets to absolute joint
states/actions; invalid pose or joint-step solutions are rejected. When the
gripper carries the task object, its state is translated with the end effector.
The current bank uses one pre-grasp and four carry anchors per demonstration.

The state-only flow policy is fine-tuned on a 50/50 mixture of nominal windows
and validated recovery windows. The 20 direction/radius cells are sampled
uniformly so that the harder, less frequently IK-valid cells are not starved.

At inference, the policy proposes a nominal \(H=50\) chunk and the controller
executes \(s=10\) actions before replanning. If the nominal full-horizon path is
unsafe, the controller considers only cells present in the validated recovery
bank. A candidate ramps from the current deviation to its selected radius in
the next \(s\) actions, then holds that radius while task phase advances:

\[
p_{t+k}^{\mathrm{ctrl}}=p_{t+k}^\star+
\big[r_0+\beta(k/s)(r-r_0)\big]n_{t+k}^{j},
\qquad \beta(u)=10u^3-15u^4+6u^5.
\]

Robot links and a rigidly held object are checked against the obstacle over all
\(H\) steps. The chosen side and last safe plan are latched across replans. As
soon as the policy's own full chunk is safe, control is released to the learned
recovery. If no supported safe continuation exists, the system holds and emits
an infeasibility reason instead of executing an unsupported deviation.

That failure is the useful training signal: it says the safety intervention
required by this scene is **not supported by the current learned recovery
envelope**. The failing anchor/direction/radius search identifies where the
training contract must be expanded. It is not a claim that avoidance is
physically impossible, and collision freedom remains relative to the modeled
robot, held-object, and obstacle geometry.

## Files

- `bidirectional_contract/interventions.py`: shared normal-ray geometry, ramp,
  recovery profile, and sequential IK.
- `bidirectional_contract/recovery_data.py`: creates validated recovery windows.
- `bidirectional_contract/train_recovery.py`: nominal/recovery fine-tuning.
- `bidirectional_contract/controller.py`: bank-constrained predictive controller.
- `bidirectional_contract/aegis.py`: AEGIS comparison adapted from
  [THU-RCSCT/vlsa-aegis](https://github.com/THU-RCSCT/vlsa-aegis), commit
  `57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b`.
- `benchmark_fixed_scene.py`: the only evaluation entry point; supports
  `base`, `aegis`, and `ours` with process-parallel MuJoCo workers.
- `reproduce_fixed_scene.sh`: builds missing recovery artifacts and reruns the
  current fixed-scene evaluations.
- `results/current_results.json`: machine-readable snapshot with provenance.

## Reproduce

Prerequisites are the existing task-2 state dataset and nominal checkpoints:

- `slop/goal/data/spatial_task2_base`
- `slop/goal/outputs/spatial_task2_h50_s10/checkpoints/final` (20k baseline)
- `slop/goal/outputs/spatial_task2_h50_s10_50k/checkpoints/final` (recovery parent)

On Sirius, from the repository root:

```bash
chmod +x bidirectional-contract/reproduce_fixed_scene.sh
WORKERS=8 GPUS=0,1 ./bidirectional-contract/reproduce_fixed_scene.sh
```

The workflow generates the recovery bank if absent, fine-tunes for 30k updates,
uses the selected 10k checkpoint, and evaluates one fixed scene: SafeLIBERO
Spatial task 2, Level I, initialization 0, moka-pot obstacle. Each worker owns a
separate simulator and policy copy. Outputs are written under
`bidirectional-contract/results/reproduction/`.

Individual stages can be run as follows:

```bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}bidirectional-contract"
CUDA_VISIBLE_DEVICES=0 pixi run python -m bidirectional_contract.recovery_data
CUDA_VISIBLE_DEVICES=0 pixi run python -m bidirectional_contract.train_recovery
pixi run python bidirectional-contract/benchmark_fixed_scene.py \
  --method ours --episodes 50 --workers 8 --gpus 0,1 --max-frames 300 \
  --recovery-checkpoint \
    bidirectional-contract/artifacts/recovery_policy/checkpoints/step_010000
```

## Results so far

All obstacle results below use the fixed scene above and policy seeds starting
at zero. Counts are shown because the sample sizes are intentionally different.

| Method | Setting | Episodes | Task success | Collision | Safe success |
|---|---|---:|---:|---:|---:|
| Nominal state-only flow action chunker | 20k checkpoint | 50 | 40/50 | 50/50 | 0/50 |
| AEGIS | released boundary, 0 cm margin | 50 | 50/50 | 26/50 | 24/50 |
| AEGIS | best tested margin, 2 cm | 25 | 24/25 | 0/25 | 24/25 |
| Bidirectional contract | **untuned**, 10k recovery checkpoint | 50 | 34/50 | 0/50 | 34/50 |

The 50-seed contract run used the same scene, seeds, and 300-frame budget as the
50-seed nominal and released-AEGIS runs. Successful episodes completed in 132.82
frames on average. All 16 task failures remained collision-free and timed out
in the explicit hold because the current bank offered no safe supported
continuation. Twenty episodes encountered an infeasible hold in total; four of
those later reacquired a supported continuation and succeeded. The contract
controller remains untuned, and its task policy is the recovery-fine-tuned 50k
parent rather than the nominal 20k checkpoint used by the two baseline rows.

As a task-retention check without obstacles, both the 50k nominal parent and the
selected 10k recovery checkpoint succeeded on 50/50 initializations. In the one
Level-II initialization-0 smoke test, the contract controller remained
collision-free but explicitly held because none of the current 12 cm bank cells
cleared the taller obstacle.
