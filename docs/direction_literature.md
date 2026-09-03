# Recovery-Aware Safety Filter Co-Design

## High-level idea

The original project asked the learned policy to do two jobs:

1. **Branch outward** into many possible obstacle-avoidance routes.
2. **Return inward** and recover the demonstrated task execution.

That is expensive and poorly grounded: the policy must represent and sample enough combinations of spatial detours, gripper rotations, and timings, while still preserving task intent. It is effectively being asked to learn its own safety controller from obstacle-free demonstrations.

The revised project splits the responsibilities cleanly:

* **The safety controller owns the outward detour.** It uses known obstacle geometry to modify the policy’s motion and guarantee or explicitly optimize collision avoidance.
* **The learned policy owns task recovery.** It is trained to continue the task from exactly the displaced and rotated states that the controller is allowed to produce.

The crucial contribution is **bidirectional co-design**:

> The policy is trained on the controller’s intervention distribution, and the controller is constrained to remain inside the policy’s trained recovery distribution.

This directly addresses the central failure identified in SafeLIBERO: AEGIS can successfully avoid an obstacle but push the robot into unfamiliar states from which the VLA no longer knows how to finish the task.

## Why this is much stronger than the original method

The original arc augmentation contained the right intuition—expand task-valid behavior around a demonstration—but assigned the wrong part to the policy. Asking a generative policy to sample the correct path, direction, amplitude, timing, and gripper rotation scales badly and provides no hard safety guarantee.

The revised method preserves the entire foundation:

* training remains obstacle-blind;
* the original demonstrations still define task intent;
* the policy still learns how to reach the demonstrated interaction poses;
* obstacle geometry is introduced only at inference;
* synthetic trajectories still expand the policy around the demonstrated path.

What changes is **who generates the deviation**. Instead of training the policy to produce every outward branch, we let a geometric controller produce a small, explicit family of safe deviations. The policy only learns the much simpler, nearly deterministic mapping:

> “Given that I have been displaced or rotated in this allowed way, continue forward and recover the task.”

This is more sample-efficient, easier to debug, and more defensible scientifically. It also gives a precise capability boundary: the method succeeds whenever a collision-free, task-progressing route exists inside the policy’s trained recovery region.

## Revised method

### 1. Define a shared intervention family

For every free-space phase of a demonstration, define a small set of controller actions around the nominal end-effector path:

* a persistent detour direction, such as left, right, upward, or diagonal;
* a continuous outward displacement rate or magnitude;
* optional slowing or stopping of nominal progress;
* a temporary gripper-yaw correction.

The controller is **not** allowed to arbitrarily modify all robot actions. It operates only through these intervention variables. A persistent direction is retained while passing an obstacle, preventing the controller from creating arbitrary zigzags that were never represented during training.

This reduces the relevant distribution from the robot’s full joint space to a small, known intervention manifold.

### 2. Train recovery from the same intervention family

During training, procedurally sample states that the intervention controller could produce:

* select a demonstration and a free-space progress point;
* select an allowed detour direction;
* sample displacement and yaw within the permitted bounds;
* construct a physically feasible perturbed robot state;
* generate a smooth, forward-progressing trajectory that reconnects to a future state on the demonstration;
* append the original demonstration suffix;
* train the normal chunked diffusion, flow, or VLA policy on these recovery trajectories.

The artificial outward action is not learned. Only the recovery continuation is used as supervision.

The result is a task-conditioned recovery field around the demonstration: the original trajectory remains the preferred execution, while displaced states map back toward a successful future task state.

### 3. Use a recovery-aware safety controller

At inference:

1. The policy predicts its normal action chunk.
2. The controller detects whether that chunk would collide.
3. It searches only over the shared intervention variables—direction, displacement, speed, and yaw.
4. It selects the smallest collision-free intervention.
5. It additionally enforces that the resulting state remains inside the policy’s trained or empirically validated recovery region.
6. Only the next execution prefix is applied, after which the policy and controller replan.

When the obstacle no longer requires intervention, the controller stops pushing outward and the recovery-trained policy naturally returns toward the task execution.

### 4. Feasibility condition

The combined system can solve an obstacle exactly when there exists a route that is simultaneously:

* collision-free;
* expressible by the allowed intervention family;
* contained inside the policy’s recovery region;
* capable of making progress toward the next task interaction.

If no such route exists, the correct result is controller infeasibility or a safe stop—not silently sending the policy into an unsupported state.

## Expected capabilities

The fixed target capability is:

> **Complete manipulation tasks around unseen free-space obstacles when the required position-and-yaw detour lies inside the jointly designed intervention and recovery region.**

Expected experimental behavior:

* **Base policy + safety filter:** much safer, but often fails after being displaced.
* **Recovery-trained policy + ordinary filter:** similar collision avoidance, substantially better task completion.
* **Recovery-trained policy + recovery-aware filter:** best task completion and lowest rate of unsupported interventions.

The method does not initially claim arbitrary global motion planning, alternative grasp poses, full-arm avoidance, or recovery from states outside its calibrated intervention region.

## Closest related work

### Recovery-oriented policy training

**SAFE-GIL** injects adversarial disturbances during expert demonstrations so the imitation policy learns corrective behavior from safety-critical states. It also shows that a safety filter can keep a robot safe while a normal BC policy fails to return to the task, and that recovery-trained policies compose better with filtering.

**MPC-SafeGIL** replaces expensive reachability calculations with sampling-based MPC disturbance generation and explicitly presents design-time recovery training as complementary to deployment-time safety filtering. This is the closest high-level precedent.

**SART** autonomously augments demonstrations inside human-annotated spatial precision regions, producing diverse but task-valid trajectories from very little human data. It is closely related to the synthetic recovery-trajectory generator, but it has no deployment safety-filter co-design.

**RESample** specifically collects policy-induced deviations and recovery trajectories rather than relying on arbitrary random perturbations. This supports targeting the perturbation distribution toward states likely to occur during deployment. It does not constrain an external safety controller to the resulting recovery support.

### Distribution-aware inference and safety

**AEGIS / SafeLIBERO** adds a hard CBF-QP safety layer to a VLA. Its own analysis identifies the exact remaining problem: the controller can avoid an obstacle but move the VLA to rare lateral or vertical states, after which the policy behaves erratically and fails to recover. It proposes richer training coverage as future work.

**PACS** avoids this problem by requiring safety interventions to remain on the policy’s original intended path, primarily through path-consistent braking. It preserves task success but deliberately gives up broad spatial detours. Our method instead expands the task-valid region first, then permits interventions inside that larger region.

**Latent Policy Barrier** predicts future latent states and optimizes policy execution to remain close to the expert distribution. Again, it protects the original narrow distribution rather than constructing a larger controller-compatible recovery region.

**CAPE** expands diffusion-policy modes at inference using collision-aware iterative refinement. It is similar in motivation, but the learned generative model remains responsible for producing the obstacle detour. It does not provide the same separation between an explicit safety controller and a policy trained specifically to recover from that controller’s interventions.

## Novelty assessment

No single paper found in this sweep closes the complete reciprocal loop:

1. define a restricted physical intervention family;
2. train the manipulation policy to recover from exactly that family;
3. calibrate the resulting recovery region;
4. constrain the same safety controller to remain inside that region.

However, simply claiming **“recovery augmentation plus a safety filter” is not novel**—SAFE-GIL and MPC-SafeGIL already establish that these components can be complementary.

The defensible contribution is therefore:

> **A support-aligned co-design of generalist manipulation policies and geometric safety controllers, where each component explicitly defines the valid operating distribution of the other.**

The strongest ablation is:

1. original policy + filter;
2. randomly recovery-augmented policy + filter;
3. controller-aligned recovery policy + filter;
4. controller-aligned recovery policy + recovery-aware filter.

This isolates whether the gain comes specifically from the bidirectional alignment rather than from generic augmentation or generic filtering.

## ICRA case

This is substantially more compelling for ICRA than the original arc-and-rank method because it targets a documented, practical failure of current VLA safety systems and combines learning and control in a modular, testable way.

The project becomes weak if it is only “SART-style recovery bridges added to AEGIS.” It becomes strong if the experiments demonstrate that:

* unrestricted filters generate unsupported states;
* recovery augmentation expands the policy’s valid intervention region;
* restricting the controller to that region further improves task success;
* the shared low-dimensional intervention family achieves this with far less data than generic perturbation coverage.

The paper’s central message should be:

> **A safety controller and a learned manipulation policy cannot be designed independently: the controller must intervene where the policy can recover, and the policy must be trained where the controller will intervene.**
