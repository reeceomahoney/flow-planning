## Mathematical summary

### 1. A shared perturbation family

Let one demonstrated free-space phase be

$$
T^\star(s)=\bigl(p^\star(s),R^\star(s)\bigr),
\qquad s\in[0,1],
$$

where \(s\) is progress from the start of the phase toward the next interaction point.

Instead of allowing arbitrary robot deviations, define a compact family of permitted perturbations:

$$
e=(j,r,\psi,\lambda).
$$

Here:

* \(j\) selects one of a few spatial directions \(d_j\), such as left, right, or upward;
* \(r\) is the displacement in that direction;
* \(\psi\) is temporary gripper yaw;
* \(\lambda\in[0,1]\) controls whether nominal forward progress continues, slows, or stops.

The perturbed end-effector pose is approximately

$$
p=p^\star(s)+r\,d_j,
$$

$$
R=R^\star(s)\operatorname{Exp}\!\left(\psi[\hat a]_\times\right).
$$

Thus, the supported perturbation region is

$$
\mathcal E
=
\bigcup_j
\left\{
(r\,d_j,\psi):
0\le r\le R_{\max},
\quad
|\psi|\le\Psi_{\max}
\right\}.
$$

This is closely related to the original arc method. The arc was already a low-dimensional family of deviations around the demonstration. The important change is:

$$
\boxed{
\text{Previously, the policy had to generate both the outward branch and the return.}
}
$$

$$
\boxed{
\text{Now, the controller generates the outward branch and the policy learns only the return.}
}
$$

That avoids asking the learned policy to sample every possible collision-avoidance trajectory.

---

### 2. Training the policy to recover

Choose a point \(s_0\) on the demonstration and sample a permitted perturbation

$$
e_0=(r\,d_j,\psi)\in\mathcal E.
$$

This produces a perturbed starting pose

$$
p_0=p^\star(s_0)+r\,d_j,
$$

$$
R_0=R^\star(s_0)\operatorname{Exp}\!\left(\psi[\hat a]_\times\right).
$$

Next, choose a future point \(s_1>s_0\) on the same successful demonstration. The synthetic recovery trajectory must move forward from \(s_0\) to \(s_1\), while gradually removing the perturbation.

Let

$$
s(\tau)=s_0+\tau(s_1-s_0),
\qquad
\tau\in[0,1],
$$

and let \(b(\tau)\) smoothly decrease from

$$
b(0)=1
\qquad\text{to}\qquad
b(1)=0.
$$

Then define

$$
\boxed{
p_{\mathrm{rec}}(\tau)
=
p^\star(s(\tau))
+
b(\tau)\,r\,d_j
}
$$

and

$$
\boxed{
R_{\mathrm{rec}}(\tau)
=
R^\star(s(\tau))
\operatorname{Exp}
\left(
b(\tau)\psi[\hat a]_\times
\right).
}
$$

At the start, the robot is fully displaced:

$$
p_{\mathrm{rec}}(0)=p^\star(s_0)+r\,d_j.
$$

At the end, it has rejoined the successful demonstration:

$$
p_{\mathrm{rec}}(1)=p^\star(s_1),
\qquad
R_{\mathrm{rec}}(1)=R^\star(s_1).
$$

The policy is trained on these recovery trajectories together with the original demonstrations:

$$
\pi_\theta
\left(
a_{t:t+H-1}\mid o_t,m_t
\right).
$$

Conceptually, it learns:

$$
\boxed{
\text{From any supported displaced state, continue forward and contract back toward successful execution.}
}
$$

The artificial outward motion is not used as a target. Only the recovery behavior is learned.

---

### 3. Restricting the safety controller to the same family

At inference, the policy first predicts its nominal action chunk

$$
A_t^\pi
=
\pi_\theta(o_t,m_t).
$$

The safety controller does not freely modify every joint or every pose dimension. It searches only over the shared parameters

$$
(j,r,\psi,\lambda)\in\mathcal E.
$$

For each candidate perturbation, it geometrically modifies the nominal chunk and checks the resulting robot trajectory for collision.

The controller chooses

$$
\boxed{
e^\star
=
\arg\min_{e\in\mathcal E}
C_{\mathrm{change}}(e)
}
$$

subject to

$$
\text{modified trajectory is collision-free}
$$

and

$$
\text{modified trajectory remains inside the trained recovery region}.
$$

In words:

> Find the smallest allowed displacement, yaw change, or slowdown that avoids the obstacle without moving the robot somewhere the policy was never trained to handle.

The controller reasons over the full prediction horizon \(H\), but executes only the first \(s\) actions:

$$
H=50,
\qquad
s=10.
$$

It then queries the policy again. If the obstacle still blocks the path, the controller adds another permitted outward correction. Once the obstacle is cleared, the controller stops intervening and the recovery-trained policy naturally returns toward the demonstration.

A persistent direction \(j\) can be retained while passing one obstacle, preventing arbitrary switching between incompatible branches.

---

### 4. The bidirectional guarantee

The training procedure covers the controller’s allowed intervention set:

$$
\boxed{
\text{controller outputs}
\subseteq
\text{policy recovery-training support}.
}
$$

The controller is simultaneously prevented from leaving the policy’s recovery region:

$$
\boxed{
\text{executed intervention}
\in
\text{policy recovery region}.
}
$$

This is the bidirectional design:

$$
\boxed{
\begin{aligned}
&\text{The controller defines which perturbations the policy learns to recover from,}\\
&\text{and the trained policy defines which perturbations the controller may apply.}
\end{aligned}
}
$$

---

### 5. Capability boundary

The combined method succeeds whenever there exists some permitted intervention sequence that is:

1. collision-free;
2. contained inside \(\mathcal E\);
3. recoverable by the policy;
4. capable of making progress toward the next task interaction.

Compactly:

$$
\boxed{
\exists\text{ a forward path inside }
\mathcal S_{\mathrm{collision\mbox{-}free}}
\cap
\mathcal R_{\mathrm{policy}}.
}
$$

The method is therefore not attempting arbitrary global motion planning. It supports exactly the safety detours that lie inside the jointly designed controller–policy envelope.
