# Research Plan: Taylor-Consistency Supervision for SuperMimic

## 1. Goal and research question

**Project:** Explore whether explicit first- and higher-order Taylor-consistency losses improve a SuperMimic motion-tracking policy beyond the existing model-based RL objective and action-smoothness reward.

**Central question:** Does teaching the actor to make accurate local predictions of its own *next action* from its derivatives improve motion-tracking performance, robustness, or temporal consistency?

The proposed method is **not** simply using policy derivatives during backpropagation. SuperMimic already uses derivatives of the actor and world model in backpropagation through time (BPTT). The proposed addition explicitly **penalizes disagreement between a Taylor-predicted next action and a directly evaluated next-policy action**.

### Distinguish the objectives

- **RL / BPTT:** Learn actions that produce high tracking return.
- **Existing action-smoothness reward:** Discourage large changes between consecutive actions.
- **Taylor consistency (proposed):** Encourage the policy's local derivatives to accurately approximate its finite-step changes in action.
- **Expert-derivative supervision (original HOS):** Match policy action derivatives to derivatives from expert action trajectories. This is a different method and would require suitable action-derivative targets.

**Important limitation:** Low Taylor error is not itself evidence of good control. A constant policy can have zero Taylor error and still track badly. Assess tracking performance and stability alongside approximation error.

## 2. Notation and measurement setup

Let:

- \(\pi_\theta\): current actor policy.
- \(x_t\): **complete policy input** at time \(t\), not just the physical state. Include the observation history and motion-reference/goal inputs used by SuperMimic.
- \(a_t = \pi_\theta(x_t)\): action produced by direct evaluation.
- \(\Delta x_t = x_{t+1} - x_t\): observed or imagined transition in the complete policy-input space.
- \(a_{t+1}^{\mathrm{direct}} = \pi_\theta(x_{t+1})\): next action from a direct policy evaluation.

The first-order directional Taylor prediction is

\[
\hat a_{t+1}^{(1)} = \pi_\theta(x_t) + D\pi_\theta(x_t)[\Delta x_t].
\]

The second-order directional prediction is

\[
\hat a_{t+1}^{(2)} = \hat a_{t+1}^{(1)}
+ \frac{1}{2}D^2\pi_\theta(x_t)[\Delta x_t,\Delta x_t].
\]

Higher orders add \(\tfrac{1}{k!}D^k\pi_\theta(x_t)[\Delta x_t,\ldots,\Delta x_t]\). These are **directional Taylor expansions in policy-input space**. They do not require explicitly estimating second-, third-, or fourth-order time derivatives of the robot state.

For each order \(k\), define a per-transition error:

\[
e_{k,t} = \left\|a_{t+1}^{\mathrm{direct}} - \hat a_{t+1}^{(k)}\right\|_2^2,
\qquad E_k = \mathbb{E}_t[e_{k,t}].
\]

**Implementation notes:** Use JVPs / nested directional automatic differentiation instead of constructing dense Jacobian or high-order derivative tensors. Ensure network activations support the derivative order being tested; nonsmooth activations complicate higher-order interpretation. Because SuperMimic's observation includes previous actions and history, build \(x_t\) and \(x_{t+1}\) exactly as the deployed actor does. Treat sampling noise consistently when measuring the change between inputs.

---

## Phase 1 — Diagnose the pretrained policy (no training changes)

**Question:** Is there a meaningful finite-step Taylor prediction error in an existing SuperMimic actor? Does higher order explain the remaining error?

### Procedure

1. Load a trained SuperMimic checkpoint and **freeze the policy**.
2. Collect consecutive actor inputs \((x_t, x_{t+1})\) and corresponding actions on representative rollouts. Initially use the same type of imagined rollouts used for policy training; separately assess real simulator rollouts where feasible.
3. Compute \(a_{t+1}^{\mathrm{direct}}\) and first- and second-order predictions \(\hat a_{t+1}^{(1)}\), \(\hat a_{t+1}^{(2)}\).
4. If practical, also measure third- and fourth-order predictions **as diagnostics**, without adding their losses to training.
5. Group transitions by motion type (walking/running, jumping/flips, recovery), contact events, tracking quality, and perturbation conditions.

### Plots and metrics

- **Main plot:** Approximation error \(E_k\) versus Taylor order \(k=1,2\) (and \(3,4\) if measured).
- Distribution plots (median, tails, e.g. 90th/99th percentiles) of \(e_{k,t}\); do not rely on mean alone.
- Per-motion and per-event error, including contact transitions and recovery.
- Scatter plots of Taylor error versus tracking error, return, and failure events.
- A normalized error metric such as
  \[
  \frac{\|a_{t+1}^{\mathrm{direct}}-\hat a_{t+1}^{(k)}\|_2}
  {\|a_{t+1}^{\mathrm{direct}}-a_t\|_2+\varepsilon}
  \]
  alongside absolute error; interpret ratios cautiously when action changes are near zero.
- Runtime and memory needed to compute each order.

### Deliverables

- Reproducible evaluation script and saved rollout dataset / seed list.
- Plots comparing first-, second-, and optional higher-order errors.
- Short report: **where errors occur, whether higher orders help, and whether errors are associated with actual control difficulties**.

### Decision gate

Proceed to Phase 2 if first-order prediction error is meaningful **or** the error is concentrated in safety-/robustness-relevant transitions. Small absolute error alone is not a reason to stop if difficult transitions have large tails. If error is uniformly tiny and unrelated to performance, reassess the motivation before adding a training loss.

---

## Phase 2 — Train with first-order Taylor consistency

**Question:** Can explicit first-order predictability improve the actor, beyond ordinary RL and action smoothing?

### Proposed auxiliary loss

\[
\mathcal L_1(\theta) =
\mathbb E_t\left[
\left\|
\operatorname{sg}\!\left[\pi_\theta(x_{t+1})\right]
- \left(\pi_\theta(x_t)+D\pi_\theta(x_t)[\Delta x_t]\right)
\right\|_2^2
\right],
\]

where \(\operatorname{sg}\) is stop-gradient. During the auxiliary-loss computation, initially **detach the sampled inputs and \(\Delta x_t\)** as well, so the loss updates the policy mapping rather than manipulating the sampled transitions. Do **not** detach the JVP output on the prediction branch: gradients must flow into actor parameters through it.

Combine with SuperMimic's unchanged RL loss:

\[
\mathcal L_{\mathrm{total}}(\theta)
= -\hat J_{\mathrm{SuperMimic}}(\theta)
+ \lambda_1\mathcal L_1(\theta).
\]

Keep the world model frozen within actor optimization, as in the original training loop. An optional **EMA target actor** is a later target-stability ablation, not an initial requirement. Stop-gradient freezes the target branch **for one backward pass**, not the whole actor across iterations.

### Controlled experiments

| Run | Objective | Purpose |
|---|---|---|
| A | Vanilla SuperMimic | Original baseline |
| B | SuperMimic with stronger original action-smoothness weight | Test whether simple smoothing explains any benefit |
| C | SuperMimic + \(\lambda_1\mathcal L_1\) | Main first-order experiment |
| D | Sweep \(\lambda_1\) (including 0) | Sensitivity / trade-off |
| Optional E | First-order consistency with EMA target | Test whether a slowly updated target helps |

Use matched data, training budgets, evaluation conditions, and multiple random seeds whenever feasible. Report the extra computation introduced by JVP-based training.

### Evaluation criteria

- First-order Taylor error on **held-out evaluation rollouts**.
- Motion-tracking return and tracking errors.
- Success/failure rates and recovery from perturbations.
- Action-difference statistics, jitter, and other stability measures.
- Performance on contact-rich motions and unseen / difficult starts.
- Training time, GPU memory, and inference-time overhead (if any).

### Interpret outcomes

- **Lower Taylor error + better control:** Supports the research hypothesis.
- **Lower Taylor error + unchanged control:** Loss works as consistency regularization, but practical value is unproven.
- **Lower Taylor error + worse control:** Regularization may suppress nonlinear corrections needed for tracking.
- **No lower Taylor error:** Debug gradients, target construction, detached inputs, derivative implementation, and loss weight before increasing order.

### Decision gate

Advance to higher-order training only after verifying that first-order consistency is trainable, measuring its benefit/cost, and understanding whether the key limitation is approximation accuracy or tracking performance.

---

## Phase 3 — Compare first- and second-order supervision

**Question:** Does explicitly supervising second-order Taylor predictability give an advantage beyond first-order consistency?

Define

\[
\mathcal L_2(\theta)=
\mathbb E_t\left[
\left\|\operatorname{sg}[\pi_\theta(x_{t+1})]
-\hat a_{t+1}^{(2)}\right\|_2^2
\right].
\]

### Experimental ablations

| Run | Auxiliary training objective |
|---|---|
| C1 | \(\lambda_1\mathcal L_1\) only |
| C2 | \(\lambda_2\mathcal L_2\) only |
| C3 | \(\lambda_1\mathcal L_1+\lambda_2\mathcal L_2\) |
| C0 | No Taylor loss (baseline) |

Use the same evaluation protocol as Phase 2. Measure errors for **all** Taylor orders after training, regardless of which loss was optimized.

**Interpretation caution:** Accurate second-order approximation does **not** imply accurate first-order approximation. A second-order term can compensate for nonlinear behavior the first-order term misses. Strongly minimizing both losses may overconstrain useful nonlinear control. Test single-order objectives before combined objectives.

### Decision gate

Proceed beyond second order only if higher-order diagnostics predict material gains and actual training improves an important control metric or robustness metric enough to justify additional computation.

---

## Phase 4 — Optional third/fourth order and adaptive weighting

**Question:** Can the method selectively use higher-order information where it helps without oversmoothing the actor or incurring prohibitive cost?

### Option A: Higher-order single-loss ablations

Evaluate \(\mathcal L_3\) and \(\mathcal L_4\) separately before combining them. Quantify both prediction and control improvements versus compute/memory overhead.

### Option B: Multi-order loss

\[
\mathcal L_{\mathrm{multi}}
=\sum_{k=1}^{K}\lambda_k\mathcal L_k.
\]

Treat the coefficients as real hyperparameters rather than assuming equal weighting is optimal. Avoid interpreting a small loss at order \(K\) as proof that lower-order derivatives are accurate.

### Option C: State-/motion-dependent reweighting

\[
\mathcal L_{\mathrm{adaptive}}
=\mathbb E_t\left[\sum_{k=1}^K\lambda_k(x_t)\,e_{k,t}\right].
\]

Potential conditioning variables include motion type, observed prediction error, or contact/recovery indicators. Begin with simple, **detached and bounded** weighting rules so the network cannot trivially reduce training loss by changing the weights. Compare against fixed-weight controls. Do not automatically upweight every large error: rapid legitimate corrections and noisy transitions may require high curvature.

### Decision gate

Keep a more complex variant only if it improves policy outcomes, approximation behavior, and the cost/benefit trade-off compared with simpler losses.

---

## Cross-phase experimental discipline

- **Always evaluate deployed policy behavior.** Consistency losses are auxiliary; one-step prediction error is not the final objective.
- **Separate training vs. evaluation data.** Report held-out transitions and test robustness outside nominal training conditions.
- **Avoid gradient leakage.** For initial consistency experiments, detach target outputs and the sampled transition inputs. Preserve differentiability through student-policy directional derivatives.
- **Keep the next-input definition correct.** SuperMimic is goal-conditioned and history-dependent; full policy inputs must shift consistently with the rollout.
- **Use fair baselines.** Compare against the existing action-smoothness term, tuned smoothness, and vanilla SuperMimic.
- **Track runtime.** High-order differentiation and backpropagation can be expensive even when it is mathematically possible.
- **Check differentiability.** Record activations and numerical behavior around nonsmooth inputs/contact events.
- **Use more than one seed** for any claimed performance improvement.
- **Label diagnostics vs. claims.** Correlation between Taylor error and failure is not proof that reducing the error fixes failure.

## Immediate next actions

1. Locate a pretrained SuperMimic actor checkpoint and a reproducible rollout/evaluation setup.
2. Confirm the actor's exact input structure, preprocessing, observation history, and goal representation.
3. Implement a first-order JVP predictor and verify it against finite differences on small controlled tests.
4. Implement second-order directional differentiation; optionally measure third/fourth orders only after validating correctness and cost.
5. Run Phase 1 plots and summarize **whether the approximation gap is substantial, where it appears, and whether it relates to tracking quality**.
6. Use the Phase 1 findings to decide whether to launch first-order regularized training.

## Success criterion

A compelling outcome is **not** merely smaller Taylor error. It is evidence that an explicit derivative-consistency objective yields improved tracking, robustness, or stability **relative to SuperMimic and stronger action-smoothness baselines**, with acceptable computational overhead. If it only reduces approximation error without improving control, record that as a useful negative or mechanistic result and reconsider the research goal.
