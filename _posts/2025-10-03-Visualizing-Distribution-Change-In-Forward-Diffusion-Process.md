---
title: "Visualizing Distribution Change in Forward Diffusion Process"
date: 2025-10-03
layout: post
usemathjax: true
---

# Visualizing Distribution Change in the Forward Diffusion Process

In this post, we derive how the **variance changes in a forward diffusion process** and how to properly scale the added noise.

---

## Initial Setup

Let $X_0$ be the initial data. We assume:

$$
X_0 \sim \mathcal{N}(\mu, 1)
$$

That is, $X_0$ follows a normal distribution with mean $\mu$ and variance $1$.

Ideally, we want $X_0$ to follow a normal distribution with mean $0$ and variance $1$.

---

## Adding Noise

Suppose we add Gaussian noise $\epsilon$ to $X_0$:

$$
X_1 = X_0 + \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

If we add the same variance at every step:

$$
\text{Var}(X_1) = \text{Var}(X_0) + \text{Var}(\epsilon) = 1 + 1 = 2
$$

The variance grows too quickly, which is undesirable.

---

## Variance Scheduler

To control variance growth, we introduce a **variance scheduler** $\beta_t$:

- $\beta_t$ defines how noise changes over time (linear, cosine, or other schedules).  
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

## Scaling the Components

Introduce constants $a, b \in \mathbb{R}$ to scale the previous sample and noise:

$$
X_1 = a X_0 + b \epsilon
$$

If $\epsilon \sim \mathcal{N}(0,1)$, then scaling gives:

$$
b \epsilon = \sqrt{\beta_t} \, \epsilon \sim \mathcal{N}(0, \beta_t)
$$

since

$$
\text{Var}(\sqrt{\beta_t} \, \epsilon) = \beta_t \cdot \text{Var}(\epsilon) = \beta_t
$$

---

## Forward Diffusion Step

Thus, the properly scaled forward diffusion step is:

$$
X_1 = a X_0 + \sqrt{\beta_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

- $a$ scales the contribution of the previous sample.  
- $\sqrt{\beta_t} \, \epsilon$ adds noise with the correct variance.  

This ensures the variance increases gradually as noise is added.

---

## Finding $a$

Now, we want $\text{Var}(X_1) = 1$.  

$$
\begin{aligned}
\text{Var}(X_1) &= \text{Var}(a X_0 + \sqrt{\beta_t} \, \epsilon) \\
&= a^2 \, \text{Var}(X_0) + \beta_t \, \text{Var}(\epsilon) \\
&= a^2 \cdot 1 + \beta_t \cdot 1 \\
&= a^2 + \beta_t
\end{aligned}
$$

To keep the variance at $1$:

$$
a^2 + \beta_t = 1 \quad \implies \quad a = \sqrt{1 - \beta_t}
$$

So the update becomes:

$$
X_1 = \sqrt{1 - \beta_t} \, X_0 + \sqrt{\beta_t} \, \epsilon
$$

---

## General Case

By recursion, the forward diffusion process is:

$$
X_t = \sqrt{1 - \beta_t} \, X_{t-1} + \sqrt{\beta_t} \, \epsilon_t, 
\quad \epsilon_t \sim \mathcal{N}(0,1)
$$

---

## Introducing $\alpha_t$

It is common to define:

$$
\alpha_t = 1 - \beta_t
$$

so that the forward step becomes:

$$
X_t = \sqrt{\alpha_t} \, X_{t-1} + \sqrt{1 - \alpha_t} \, \epsilon_t
$$

This notation is convenient because:

- $\alpha_t$ directly represents the **retained information** from the previous step.  
- $1 - \alpha_t$ is the **noise contribution**.  
- Products of $\alpha_t$ across steps (called $\bar{\alpha}_t$) make it efficient to compute the distribution of $X_t$ directly in closed form.

---

## Summary

- Naively adding noise grows variance too quickly.  
- A scheduler $\beta_t$ controls noise growth.  
- Scaling with $a = \sqrt{1 - \beta_t}$ ensures variance remains stable.  
- Using $\alpha_t = 1 - \beta_t$ simplifies notation and allows efficient closed-form expressions for the forward diffusion process.

---
