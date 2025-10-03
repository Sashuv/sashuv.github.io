---
title: "Forward Diffusion Process: Scaling and Alphas"
date: 2025-10-03
layout: post
usemathjax: true
---

# Forward Diffusion Process: Scaling and Alphas

In this post, we derive how to properly scale the forward diffusion step and introduce the **alpha notation**, which makes computations more efficient.

---

Suppose we start with data \(X_0\) with variance 1:

\[
X_0 \sim \mathcal{N}(0,1)
\]

If we add Gaussian noise \(\epsilon \sim \mathcal{N}(0,1)\), the naive forward step is:

\[
X_1 = X_0 + \epsilon
\]

The variance becomes:

\[
\text{Var}(X_1) = \text{Var}(X_0) + \text{Var}(\epsilon) = 1 + 1 = 2
\]

This variance increases too quickly, which is undesirable.

---

To control variance growth, we introduce a **variance scheduler** \(\beta_t\):

- \(\beta_t\) defines how much noise is added at timestep \(t\).  
- Typically, \(\beta_t\) is small (e.g., 0.0001 to 0.02).  

We now scale the previous image and noise:

\[
X_1 = a X_0 + \sqrt{\beta_1} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
\]

---

To keep variance constant at 1:

\[
\begin{aligned}
\text{Var}(X_1) &= \text{Var}(a X_0 + \sqrt{\beta_1} \, \epsilon) \\
&= a^2 \text{Var}(X_0) + \beta_1 \text{Var}(\epsilon) \\
&= a^2 + \beta_1
\end{aligned}
\]

Setting \(\text{Var}(X_1) = 1\):

\[
a^2 + \beta_1 = 1 \quad \Rightarrow \quad a = \sqrt{1 - \beta_1}
\]

So the first step becomes:

\[
X_1 = \sqrt{1 - \beta_1} \, X_0 + \sqrt{\beta_1} \, \epsilon
\]

---

Generally, at timestep \(t\):

\[
X_t = \sqrt{1 - \beta_t} \, X_{t-1} + \sqrt{\beta_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
\]

---

### Alphas for Efficiency

Define:

\[
\alpha_t = 1 - \beta_t
\]

and

\[
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
\]

Then we can write a **single-step formula from \(X_0\) to \(X_t\)** without iterating:

\[
X_t = \sqrt{\bar{\alpha}_t} \, X_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon
\]

- This avoids repeated multiplications and is computationally efficient.  
- \(\bar{\alpha}_t\) accumulates the effect of all previous noise additions.  

---

This formulation ensures that the **variance grows gradually** while maintaining control over the mean, and allows us to compute \(X_t\) directly from \(X_0\) using alphas.
