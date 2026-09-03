# What is still missing

This implementation is a **first feasibility test**, not the final version of
the method. It tests whether controller-owned evasion can be coupled to a policy
trained to recover from the controller's intervention family. The fixed-scene
result shows that this basic contract can work, but the current implementation
is deliberately limited and still leaves substantial method and tuning work.

## The controller is not a CBF yet

The current controller is a discrete predictive selector. It enumerates
direction/radius cells from the recovery bank, converts them to joint-space
plans, rejects candidates that fail its geometric checks, and selects a safe
candidate or holds.

It does **not** yet formulate a control barrier function, enforce a barrier
derivative constraint, solve a CBF-QP, or establish forward invariance of a
safe set. Consequently, the current zero-collision result should be understood
as an empirical result of the selector and modeled geometry—not a formal CBF
safety guarantee.

The intended next controller should use the nominal policy chunk as its task
reference, impose continuous safety constraints through a proper CBF or safety
filter, and constrain that filter to remain inside the learned recovery
envelope.

## Recovery coverage is sparse

Recovery training currently uses only four anchors during the carrying segment,
plus one pre-grasp anchor, for each demonstration. At each anchor, coverage is
further discretized into five translation directions and four radii. Large
parts of the state and task-phase space therefore have no direct recovery
supervision.

The explicit infeasible holds expose this sparsity. They mean that the selector
could not find a collision-free intervention represented by the current bank;
they do not mean that the task is physically impossible. More anchors and
better placement of those anchors are required before judging the underlying
contract.

## The intervention family is minimal

The only current intervention is Cartesian translation using a quintic
ramp-and-hold profile. It is intentionally the simplest possible intervention:
move outward along one normal-plane ray, reach one of four fixed radii, and hold
that deviation until the policy can recover.

The number and placement of directions, radii, ramp duration, persistence, and
branch-selection behavior have not been tuned. The present family is a first
shared parameterization between training and inference, not a claim that
ramp-and-hold is the best intervention family.

## Yaw is not controlled

The selector changes end-effector translation while copying the nominal or
demonstrated wrist orientation. It therefore cannot rotate the gripper or a
held object around an obstacle even when a yaw change would create a much
larger safe passage.

The recovery state and intervention contract need to include selected
orientation dimensions—at minimum yaw—before the controller can exploit this
degree of freedom safely.

## Intervention velocities are too aggressive

Some selected ramps still generate large Cartesian and joint velocities. The
current IK validation has a joint-step cutoff, but this is only a coarse
rejection rule. It does not properly constrain Cartesian velocity, joint
velocity, acceleration, or jerk over the intervention and transition back to
the policy.

The final controller needs explicit dynamic limits and a smoother objective.
The training distribution must then use the same dynamically feasible
interventions so that the recovery contract matches what can actually be
executed.

## Evaluation is still narrow

The present obstacle result covers one SafeLIBERO Spatial task, one fixed Level-I
initialization, one obstacle placement, and 50 stochastic policy seeds. The
34/50 safe-success and 0/50 collision counts are useful debugging numbers for
this testbed, but they are not final benchmark results.

Before treating the method as complete, it still needs tuning and evaluation
across obstacle placements, Level II, additional task phases and tasks, and a
matched comparison after the controller and recovery envelope have stopped
changing.

## Bottom line

The current code establishes the smallest useful loop:

1. define one intervention family shared by training and inference;
2. train the policy to recover from that family;
3. allow the selector to use only supported interventions; and
4. turn missing support into an explicit hold instead of a collision.

What remains is the substantive final method: a true safety filter or CBF,
denser and calibrated recovery coverage, yaw-aware interventions, dynamically
feasible motion, and systematic tuning of the intervention parameterization.
