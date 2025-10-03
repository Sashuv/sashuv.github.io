---
title: "Visualizing Distribution Change in Forward Diffusion Process"
date: 2025-10-03
layout: post
usemathjax: true
---

# Visualizing Distribution Change in Forward Diffusion Process

In this post, we derive how the **variance changes in a forward diffusion process** and how to properly scale the added noise, leading to the efficient closed-form solution used in modern Diffusion Models.

---

## Step 1: Initial Data and Assumptions

Let \(X_0\) be the initial data (e.g., an image vector). For mathematical tractability, the data is typically normalized to approximate the standard normal distribution:

The distribution of the initial data is:
\\(
X_0 \sim \mathcal{N}(\mu, 1)
\\)

For simplicity in diffusion models, we often assume the mean is zero: \(\mu = 0\) and the variance is 1: \(\text{Var}(X_0) = 1\).

---

## Step 2: Uncontrolled Noise Addition

If we were to add unscaled Gaussian noise \(\epsilon \sim \mathcal{N}(0,1)\) at every step, the variance would rapidly increase:

The next state \(X_1\) is defined as:
\\(
X_1 = X_0 + \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
\\)

The resulting variance for the next step would be:
\\(
\begin{align*}
\text{Var}(X_1) &= \text{Var}(X_0) + \text{Var}(\epsilon) \\
&= 1 + 1 = 2
\end{align*}
\\)

This rapid variance growth is undesirable as it loses the signal too quickly.

---

## Step 3: Introducing Variance Scheduling (\(\beta_t\))

To control the signal-to-noise ratio, we introduce a time-dependent **variance scheduler** \(\beta_t\), which determines the small amount of variance added at time \(t\).

- \(\beta_t\) is typically very small (e.g., $0.0001$ to $0.02$), ensuring noise is added gradually.

The noise \(\epsilon_t\) for step \(t\) is defined as:
\\(
\epsilon \sim \mathcal{N}(0, \beta_t)
\\)

---

## Step 4: Scaling the Signal and Noise

For the general step from \(X_{t-1}\) to \(X_t\), we introduce scaling constants \(a\) and \(b\) to the previous image and the noise term:

\\(
X_t = a X_{t-1} + b \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
\\)

To ensure the noise term has variance \(\beta_t\), we set \(b = \sqrt{\beta_t}\):
\\(
b \epsilon = \sqrt{\beta_t} \, \epsilon \sim \mathcal{N}(0, \beta_t)
\\)

Thus the single step can be written as:
\\(
X_t = a X_{t-1} + \sqrt{\beta_t} \, \epsilon
\\)

---

## Step 5: Deriving the Signal Scaling Factor \(a\)

The core idea of the forward process in DDPMs is to construct a **Markov chain** where the distribution \(X_t\) always converges toward \(\mathcal{N}(0, I)\). We enforce that the resulting variance of \(X_t\) is 1, provided the variance of \(X_{t-1}\) was 1.

Using the property that \(\text{Var}(A+B) = \text{Var}(A) + \text{Var}(B)\) for independent random variables:

\\(
\text{Var}(X_t) = \text{Var}(a X_{t-1}) + \text{Var}(\sqrt{\beta_t} \, \epsilon)
\\)

Assuming \(\text{Var}(X_{t-1}) = 1\) and \(\text{Var}(\epsilon) = 1\), and enforcing \(\text{Var}(X_t) = 1\):

\\(
\begin{align*}
1 &= a^2 \cdot \text{Var}(X_{t-1}) + \beta_t \cdot \text{Var}(\epsilon) \\
1 &= a^2 + \beta_t
\end{align*}
\\)

Solving for \(a\) gives:
\\(
a = \sqrt{1 - \beta_t}
\\)

The final forward diffusion step, defining the transition probability \(q(X_t | X_{t-1})\), is:
\\(
X_t = \sqrt{1-\beta_t} \, X_{t-1} + \sqrt{\beta_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
\\)

---

## Step 6: Introducing \(\alpha_t\) and the Closed-Form Expression (Efficiency)

To simplify the notation and enable efficient computation, we define:
\\(
\alpha_t = 1 - \beta_t
\\)

The one-step diffusion process is then written concisely as:
\\(
X_t = \sqrt{\alpha_t} \, X_{t-1} + \sqrt{1-\alpha_t} \, \epsilon
\\)

### The Cumulative Product \(\bar{\alpha}_t\)

The true power of this formulation comes from the **closed-form solution** for sampling \(X_t\) directly from \(X_0\). We define the cumulative product of the \(\alpha\) terms as **alpha bar**:
\\(
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
\\)

By recursively applying the one-step formula, the full forward process can be simplified to a single-step equation:

\\(
X_t = \sqrt{\bar{\alpha}_t} \, X_0 + \sqrt{1-\bar{\alpha}_t} \, \epsilon
\\)

### Efficiency and Convergence

- **Efficiency:** This closed-form expression means we can sample any noisy step \(X_t\) from the original data \(X_0\) **in one step**, avoiding the need to loop through \(t\) iterations. This is crucial for training the Denoising Diffusion Probabilistic Model (DDPM).
- **Convergence:** As \(t\) approaches the final step \(T\), the term \(\bar{\alpha}_t\) (the signal multiplier) approaches 0, and the term \(1-\bar{\alpha}_t\) (the noise variance) approaches 1. This guarantees that \(X_T\) is effectively pure Gaussian noise \(\mathcal{N}(0, I)\).

---

## Summary

| Term | Definition | Purpose |
| :--- | :--- | :--- |
| **\(\beta_t\)** | Variance schedule | Controls the small amount of noise added at time \(t\). |
| **\(\alpha_t\)** | \(\alpha_t = 1 - \beta_t\) | Scales the previous state \(X_{t-1}\) to preserve variance. |
| **\(\bar{\alpha}_t\)** | \(\prod_{s=1}^{t} \alpha_s\) | Enables **efficient direct sampling** of \(X_t\) from \(X_0\). |

This complete formulation is the stable and efficient foundation for the **forward diffusion process in DDPMs**.
