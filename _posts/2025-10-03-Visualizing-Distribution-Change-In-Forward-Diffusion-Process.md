---
title: "Visualizing Distribution Change in Forward Diffusion Process"
date: 2025-10-03
layout: post
usemathjax: true
---

# Visualizing Distribution Change in Forward Diffusion Process

In this post, we derive how the **variance changes in a forward diffusion process** and how to properly scale the added noise to ensure a controlled progression from data to pure Gaussian noise.

---

## Step 1: Initial Data and Assumption

Let $X_0$ be the initial data (e.g., an image). While the true data distribution is complex, for mathematical tractability in the forward process, we often assume the data has been **normalized** to have a specific mean and variance.

We assume:
$$
X_0 \sim \mathcal{N}(\mu, 1)
$$

**Note on $\mu$:** In the context of Diffusion Models (specifically DDPM), the data is typically pre-processed to ensure the initial distribution is $\mathcal{N}(0, I)$, or at least $X_0$ is scaled such that its mean is $\approx 0$ (e.g., pixel values from $[0, 255]$ are scaled to $[-1, 1]$). For the purpose of deriving the $\text{variance scaling}$, we primarily focus on $\text{Var}(X_0) = 1$.

---

## Step 2: Uncontrolled Noise Addition (Undesirable)

Suppose we add unscaled Gaussian noise $\epsilon$ with unit variance to $X_0$:

$$
X_1 = X_0 + \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

If we add the same variance at every step:
$$
\text{Var}(X_1) = \text{Var}(X_0) + \text{Var}(\epsilon) = 1 + 1 = 2
$$

This variance increases too quickly, resulting in complete noise (loss of signal) in very few steps, which is undesirable for a gradual process.

---

## Step 3: Introducing Variance Scheduling ($\beta_t$)

To control the variance growth, we introduce a **variance scheduler** $\beta_t$:

- $\beta_t$ defines the small amount of noise added at step $t$.
- Typically, $\beta_t$ is very small (e.g., $0.0001$ to $0.02$) to ensure noise is added gradually.

We now define the noise $\epsilon_t$ to satisfy:

$$
\epsilon_t \sim \mathcal{N}(0, \beta_t)
$$

The noise is added to the previous step's output $X_{t-1}$ (using $t=1$ for the first step):
$$
X_1 = X_0 + \epsilon_1
$$

---

## Step 4: Deriving the Scaling Factor $a$

To ensure that the final distribution at step $T$ is the **standard normal distribution** $\mathcal{N}(0, 1)$, and to stabilize the intermediate steps, we must **scale down** the signal $X_{t-1}$ when adding the noise.

The general form of one forward diffusion step is:
$$
X_t = a_t X_{t-1} + b_t \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$
where $b_t = \sqrt{\beta_t}$ to ensure the noise term has variance $\beta_t$.

The goal is to maintain $\text{Var}(X_t) = \text{Var}(X_{t-1})$ or, more commonly in $\text{DDPM}$, to maintain a well-defined relationship between the signal and noise. Following the $\text{DDPM}$ derivation where $X_t$ is constructed to converge to $\mathcal{N}(0, I)$, we enforce $\text{Var}(X_t) = 1$:

$$
\text{Var}(X_t) = \text{Var}(a_t X_{t-1} + \sqrt{\beta_t} \epsilon)
$$
Since $X_{t-1}$ and $\epsilon$ are independent:
$$
\text{Var}(X_t) = \text{Var}(a_t X_{t-1}) + \text{Var}(\sqrt{\beta_t} \epsilon)
$$
Assuming $\text{Var}(X_{t-1}) = 1$ (which is true at $t=1$ if $\text{Var}(X_0)=1$, and maintained if we enforce $\text{Var}(X_t)=1$):
$$
1 = a_t^2 \cdot \text{Var}(X_{t-1}) + \beta_t \cdot \text{Var}(\epsilon)
$$
$$
1 = a_t^2 \cdot 1 + \beta_t \cdot 1
$$
Solving for $a_t$:
$$
a_t^2 = 1 - \beta_t \quad \implies \quad a_t = \sqrt{1 - \beta_t}
$$

## Step 5: Single Forward Step and $\alpha_t$ Notation

Let $\alpha_t = 1 - \beta_t$. The properly scaled forward diffusion step from $X_{t-1}$ to $X_t$ is:

$$
q(X_t | X_{t-1}) = \mathcal{N}(X_t; \sqrt{\alpha_t} X_{t-1}, \beta_t I)
$$

Expressed as a linear transformation:
$$
\mathbf{X}_t = \sqrt{\alpha_t} \mathbf{X}_{t-1} + \sqrt{\beta_t} \mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)
$$

- $\sqrt{\alpha_t}$ scales the signal (the previous image).
- $\sqrt{\beta_t}$ scales the noise $\mathbf{\epsilon}$ to ensure the correct variance $\beta_t$.

---

## Step 6: The Closed-Form Expression (Efficiency)

The most critical part of the forward process is the ability to sample $\mathbf{X}_t$ directly from the initial data $\mathbf{X}_0$ without running the full iterative Markov chain (i.e., without knowing $X_1, X_2, \dots, X_{t-1}$).

We define the cumulative product of the $\alpha_t$ terms as **alpha bar** ($\bar{\alpha}_t$):
$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

By recursively substituting the expression for $X_{t-1}$, $X_{t-2}$, and so on, it can be proven that $X_t$ has a closed-form Gaussian distribution conditioned on $X_0$:

$$
q(\mathbf{X}_t | \mathbf{X}_0) = \mathcal{N}(\mathbf{X}_t; \sqrt{\bar{\alpha}_t} \mathbf{X}_0, (1 - \bar{\alpha}_t) I)
$$

Expressed as a linear transformation:
$$
\mathbf{X}_t = \sqrt{\bar{\alpha}_t} \mathbf{X}_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon}
$$

### Why is this efficient?

1.  **Direct Sampling:** For any timestep $t$, we can **directly sample** $\mathbf{X}_t$ from $\mathbf{X}_0$ using the above equation in a single step, rather than $t$ sequential steps.
2.  **Variance Control:** As $t \to T$ (the total number of steps):
    - $\bar{\alpha}_t \to 0$, so the signal term $\sqrt{\bar{\alpha}_t} \mathbf{X}_0$ vanishes.
    - $1 - \bar{\alpha}_t \to 1$, so the noise term $\sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon}$ converges to $\mathcal{N}(0, I)$.
    - This mathematically guarantees that $\mathbf{X}_T$ is pure Gaussian noise, a required starting point for the reverse (generative) process.
