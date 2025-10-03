---
title: "Visualizing Distribution Change in Forward Diffusion Process"
date: 2025-10-03
layout: post
usemathjax: true
---

# Visualizing Distribution Change in Forward Diffusion Process

In this post, we derive how the **variance changes in a forward diffusion process** and how to properly scale the added noise.

---

## Step 1: Initial Data

Let $X_0$ be the initial data. We assume:

$$
X_0 \sim \mathcal{N}(\mu, 1)
$$

That is, $X_0$ follows a normal distribution with mean $\mu$ and variance 1.

---

## Step 2: Adding Noise

Suppose we add Gaussian noise $\epsilon$ to $X_0$:

$$
X_1 = X_0 + \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

If we add the same variance at every step:

$$
\text{Var}(X_1) = \text{Var}(X_0) + \text{Var}(\epsilon) = 1 + 1 = 2
$$

This variance increases too quickly, which is undesirable.

---

## Step 3: Variance Scheduling

To control the variance growth, we introduce a **variance scheduler** $\beta_t$:

- $\beta_t$ defines how noise changes over time (linear, cosine, or any schedule).  
- Typically, $\beta_t$ is very small (e.g., 0.0001 to 0.02) so that noise is added gradually.

We now want the noise to satisfy:

$$
\epsilon \sim \mathcal{N}(0, \beta_t)
$$

and

$$
X_1 = X_0 + \epsilon
$$

---

## Step 4: Scaling Noise and Previous Image

Introduce constants $a, b \in \mathbb{R}$ to scale the previous image and noise:

$$
X_1 = a X_0 + b \epsilon
$$

If $\epsilon \sim \mathcal{N}(0,1)$, we can scale it to match the desired variance:

$$
b \epsilon = \sqrt{\beta_t} \, \epsilon \sim \mathcal{N}(0, \beta_t)
$$

because

$$
\text{Var}(\sqrt{\beta_t} \, \epsilon) = \beta_t \cdot \text{Var}(\epsilon) = \beta_t \cdot 1 = \beta_t
$$

---

## Step 5: Final Forward Diffusion Step

The properly scaled forward diffusion step is:

$$
X_1 = a X_0 + \sqrt{\beta_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

- $a$ scales the contribution of the previous image.  
- $\sqrt{\beta_t} \, \epsilon$ adds noise with the correct variance.  

This ensures that the **variance increases gradually** as noise is added in each step.
