# Mathematical Foundations and Proof Obligations

## 1. Exact replay theorem

Assume:

1. the initial model state \(\theta_0\) is identical,
2. the initial optimizer state \(o_0\) is identical,
3. the complete RNG state is identical,
4. every batch and augmentation is identical,
5. the learning-rate schedule is a deterministic function of step,
6. every executed operation is deterministic on the same software and hardware,
7. floating-point execution order is identical.

Let the optimizer recurrence be:

\[
(\theta_{t+1}, o_{t+1}) = U_t(\theta_t, o_t, B_t).
\]

Then replay produces bitwise-identical \((\theta_t,o_t)\) for every step.

### Proof

At \(t=0\), equality holds by assumptions 1 and 2.

Suppose equality holds at step \(t\). The replay receives the same state, batch, augmentation, RNG outputs, learning rate, and deterministic floating-point operations. Therefore \(U_t\) returns the same next model and optimizer state bit for bit.

By induction, equality holds for every later step.

The repository tests this theorem operationally by reconstructing a stored future anchor from an earlier anchor and comparing SHA-256 hashes plus the maximum absolute parameter difference.

## 2. Why a saved seed is insufficient

A seed alone does not determine a run if any of the following changes:

- batch order,
- worker scheduling coupled to random transforms,
- random-number consumption before a later operation,
- optimizer state,
- learning-rate state,
- nondeterministic kernels,
- library version,
- hardware execution behavior.

This is why the project stores the realized batch plan and full RNG states rather than only the original seed.

## 3. Fixed-denominator example removal

The ordinary batch loss is:

\[
L_B(\theta) = \frac{1}{B}\sum_{i=1}^{B}\ell_i(\theta).
\]

To remove example \(r\), use:

\[
L_B^{(-r)}(\theta) = \frac{1}{B}
\sum_{i=1}^{B} \mathbf{1}[i \ne r]\ell_i(\theta).
\]

Then:

\[
\nabla L_B^{(-r)} = \nabla L_B - \frac{1}{B}\nabla \ell_r.
\]

This is a literal subtraction of the removed sample's gradient contribution.

Renormalizing by \(B-1\) would instead give:

\[
\frac{1}{B-1}\sum_{i\ne r}\nabla \ell_i,
\]

which rescales every remaining gradient. That intervention confounds removal with a learning-rate change for the rest of the batch.

Weight decay, momentum, and the global learning-rate schedule remain unchanged in Feature Genesis.

## 4. GroupNorm and sample independence

For BatchNorm, normalized sample \(x_i\) depends on batch statistics:

\[
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}.
\]

Because \(\mu_B\) and \(\sigma_B\) contain all samples, editing sample \(x_r\) changes \(\hat{x}_i\) for \(i \ne r\).

GroupNorm computes statistics within channels of one sample:

\[
\hat{x}_{i,g} = \frac{x_{i,g} - \mu_{i,g}}
{\sqrt{\sigma_{i,g}^2 + \epsilon}}.
\]

Thus the forward path of sample \(i\) is independent of other batch members, assuming no other cross-sample operation exists.

## 5. Sparse autoencoder geometry

Let normalized input be:

\[
\tilde{x} = \frac{x}{s}.
\]

A standard encoder computes:

\[
p = (\tilde{x} - b_D)W_E + b_E.
\]

After a sparsifier \(\mathcal{S}\):

\[
z = \mathcal{S}(p),
\qquad
\hat{x} = s(Dz + b_D).
\]

Decoder atoms satisfy:

\[
\|d_j\|_2 = 1.
\]

The decoder gradient is projected onto the tangent space of the unit sphere:

\[
g_j^{\perp} = g_j - (g_j^T d_j)d_j.
\]

After the optimizer update, atoms are renormalized.

## 6. Why strict orthogonality cannot replace sparsity

If \(D \in \mathbb{R}^{m \times d}\) has \(m>d\), at most \(d\) decoder rows can be mutually orthogonal. An overcomplete dictionary with \(m>d\) must contain nonzero pairwise inner products.

Moreover, semantic factors can be correlated and hierarchical. Forcing all features into a single orthogonal basis can destroy useful shared geometry. Sparsity answers a different question: which small subset of an overcomplete dictionary explains this activation?

## 7. BatchTopK sparsity identity

For a batch preactivation matrix \(A \in \mathbb{R}_{\ge0}^{N\times m}\), BatchTopK retains the largest \(Nk\) entries globally:

\[
Z = \operatorname{Top}_{Nk}(A).
\]

Ignoring exact zero ties:

\[
\|Z\|_0 = Nk,
\qquad
\frac{1}{N}\sum_{i=1}^{N}\|z_i\|_0 = k.
\]

Unlike per-sample TopK, individual \(\|z_i\|_0\) may vary.

## 8. Sample-independent BatchTopK inference

Global BatchTopK cannot be used on one isolated sample without changing its meaning. After training, let \(A \in \mathbb{R}_{\ge 0}^{N\times m}\) contain positive preactivations on a deterministic calibration sample. Set \(q=Nk\), and let \(\tau\) be the \(q\)-th largest entry of \(A\). Inference uses:

\[
z_j = a_j \mathbf{1}[a_j \ge \tau].
\]

Ignoring ties at the threshold, the calibration sample then has exactly mean L0 \(k\). Ties can produce a small excess because every activation equal to \(\tau\) is retained. The realized inference L0 is measured and stored rather than assumed. A smoke test constructs distinct scores and verifies exact recovery of the target mean L0.

## 9. JumpReLU and direct L0 training

JumpReLU is:

\[
J_{\theta}(a) = a\mathbf{1}[a>\theta].
\]

The loss is:

\[
\mathcal{L} = \|x-\hat{x}\|_2^2 + \lambda
\sum_j \mathbf{1}[a_j>\theta_j].
\]

Because the threshold operation is discontinuous, straight-through estimators approximate gradients in a small band around \(a_j=\theta_j\).

The initialization selects a global activation quantile so the starting threshold approximately satisfies:

\[
\frac{1}{N}\sum_i \|z_i\|_0 \approx k_{\text{target}}.
\]

This makes early comparisons less sensitive to an arbitrary threshold scale.

## 10. Cosine-scored encoder

Let centered input be \(x_c\) and normalized encoder direction be \(\bar{w}_j\). The score is:

\[
p_j = \exp(b_j)\|x_c\|_2^{\alpha_j}
\frac{x_c^T \bar{w}_j}{\|x_c\|_2} + b_{E,j}.
\]

Special cases are:

\[
\alpha_j = 0 \Rightarrow \text{pure cosine dependence},
\]

\[
\alpha_j = 1 \Rightarrow \text{inner-product-like norm dependence}.
\]

The test suite verifies scale invariance when \(\alpha_j=0\) and biases are neutralized.

## 11. Matryoshka objective

Let dictionary prefixes be:

\[
1 \le m_1 < \cdots < m_R=m.
\]

Compute one global sparse code \(z\). Prefix \(r\) reconstructs using only the first \(m_r\) units:

\[
\hat{x}^{(r)} = D_{1:m_r}z_{1:m_r} + b_D.
\]

The loss is:

\[
\mathcal{L}_{\text{mat}} = \sum_{r=1}^{R} w_r
\|x-\hat{x}^{(r)}\|_2^2.
\]

The implementation test proves that the returned full code is exactly the one global BatchTopK code and is not recomputed independently per prefix.

## 12. Matching Pursuit residual monotonicity

Let residual before step \(q\) be \(r_q\), and choose unit direction \(d_q\) with positive coefficient:

\[
a_q = \max(0,r_q^T d_q).
\]

Update:

\[
r_{q+1} = r_q-a_qd_q.
\]

If \(a_q>0\), then:

\[
\|r_{q+1}\|_2^2
= \|r_q\|_2^2 - 2a_qr_q^Td_q + a_q^2
= \|r_q\|_2^2-a_q^2
\le \|r_q\|_2^2.
\]

Also:

\[
r_{q+1}^Td_q = r_q^Td_q-a_q=0.
\]

The test suite verifies both monotonic residual norm and stepwise orthogonality numerically.

## 13. Fraction of variance explained

For activation rows \(x_i\), mean \(\bar{x}\), and reconstructions \(\hat{x}_i\):

\[
\operatorname{FVE} = 1 -
\frac{\sum_i\|x_i-\hat{x}_i\|_2^2}
{\sum_i\|x_i-\bar{x}\|_2^2}.
\]

FVE alone is insufficient because a dense or semantically poor dictionary can reconstruct well.

## 14. Mutual coherence

Dictionary mutual coherence is:

\[
\mu(D) = \max_{i\ne j}|d_i^Td_j|.
\]

For an overcomplete dictionary, \(\mu(D)=0\) is impossible. The metric is diagnostic, not an objective that guarantees semantic disentanglement.

## 15. Orthogonal Procrustes alignment

Given centered matrices \(X,Y\), solve:

\[
R^* = \arg\min_{R^TR=I}\|XR-Y\|_F^2.
\]

If:

\[
X^TY = U\Sigma V^T,
\]

then:

\[
R^* = UV^T.
\]

The test suite constructs an exact random rotation plus translation and recovers it to numerical precision.

## 16. Feature matching score

For source feature \(i\) and target feature \(j\):

\[
S_{ij} = \frac{
\lambda_d C_{ij}^{(d)}+
\lambda_a C_{ij}^{(a)}+
\lambda_c C_{ij}^{(c)}
}{\lambda_d+\lambda_a+\lambda_c}.
\]

The three cosine terms compare aligned decoder directions, activation signatures, and concept profiles.

A Hungarian assignment maximizes total continuation score. Split and merge edges are added only when multiple secondary scores exceed both a local threshold and a joint mass threshold.

## 17. Gradient-alignment approximation

At the actual baseline state \(\theta_t\), let the current batch gradient be:

\[
g_t = \nabla_\theta L_{B_t}(\theta_t).
\]

For the feature objective \(F_j\), a first-order Taylor expansion of a plain gradient step gives:

\[
F_j(\theta_t-\eta_t g_t) \approx F_j(\theta_t)-
\eta_t\nabla F_j(\theta_t)^T g_t.
\]

The implemented screening score is therefore:

\[
I_{t,j}=-\eta_t\nabla F_j(\theta_t)^T g_t.
\]

Candidate batches are evaluated sequentially on the evolving baseline model. After scoring step \(t\), the implementation applies the exact configured learning rate, gradient clipping, weight decay, momentum, and Nesterov optimizer update before evaluating step \(t+1\). The probe-set feature gradient is expensive, so it is refreshed every \(r\) steps and reused between refreshes:

\[
\nabla F_j(\theta_t) \approx \nabla F_j(\theta_{r\lfloor t/r\rfloor}).
\]

A unit test verifies that screening reaches exactly the same final model state as ordinary baseline training. Exact replay remains necessary because the score itself is first order and does not capture curvature, the full optimizer-state response to an intervention, or later nonlinear interactions.

## 18. Causal effect under replay

Let \(R_{t_0\rightarrow t_1}\) be the deterministic optimizer map from the anchor state at \(t_0\) to \(t_1\). Let \(R^{I}\) apply intervention \(I\) to selected training events.

The feature effect is:

\[
\Delta_j(I) = F_j\left(R^I_{t_0\rightarrow t_1}(\theta_{t_0})\right)
-F_j\left(R_{t_0\rightarrow t_1}(\theta_{t_0})\right).
\]

Because both trajectories share every non-intervened detail, \(\Delta_j\) is a causal effect of the intervention within the specified replay system.

It is not automatically the effect of permanently removing the example from the entire original training history. The interval and intervention must be stated exactly.

## 19. Model-level fidelity

Let task loss after substituting SAE reconstruction be \(L_{\text{SAE}}\), original loss be \(L_{\text{base}}\), and zero-ablation loss be \(L_0\). The recovered-loss fraction is:

\[
\operatorname{Recovered} = 1-
\frac{L_{\text{SAE}}-L_{\text{base}}}
{L_0-L_{\text{base}}}.
\]

This asks how much of the layer's task-relevant information the SAE reconstruction preserves relative to removing the layer output entirely.

## 20. Feature ablation units

If the SAE normalizes input by scale \(s\), the raw activation contribution of feature \(j\) is:

\[
c_j = s z_j d_j.
\]

Therefore exact feature ablation subtracts \(sz_jd_j\), not merely \(z_jd_j\). A dedicated unit test checks this conversion.

## 21. Non-identifiability warning

Even with perfect reconstruction, sparse dictionaries can be non-canonical. Features can permute, split, merge, or change across seeds. Consequently:

\[
\text{SAE latent birth} \not\Rightarrow \text{model feature birth}.
\]

The methodology requires agreement among independent SAEs, backward projection, semantic interventions, supervised factors, and behavioral effects before making a developmental claim.
