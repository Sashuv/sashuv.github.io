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

In diffusion models, the forward process gradually adds noise to data over several timesteps.  
At each step, we take the previous state and inject a small amount of Gaussian noise.

Formally, we define a sequence of random variables:

$$
X_0, X_1, X_2, \dots, X_t
$$

where $X_0$ is the original data and each subsequent $X_t$ is obtained by adding noise.

---

## Naive approach: adding noise directly

Suppose we add Gaussian noise $epsilon$ at each step:

$$
X_1 = X_0 + \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

If we repeat this step, we get:

$$
X_2 = X_1 + \epsilon_2, \quad \epsilon_2 \sim \mathcal{N}(0,1)
$$

and so on, until $X_t$.

At the very first step, the variance becomes:

$$
\mathrm{Var}(X_1) = \mathrm{Var}(X_0) + \mathrm{Var}(\epsilon) = 1 + 1 = 2
$$

At the next step:

$$
\mathrm{Var}(X_2) = \mathrm{Var}(X_1) + \mathrm{Var}(\epsilon_2) = 2 + 1 = 3
$$

By repeating this process, the variance grows linearly with the number of steps.

---

## Why is this a problem?

Such a rapid growth in variance causes the signal to be overwhelmed by noise too quickly.  
This makes the forward process unstable and prevents the model from learning a meaningful reverse process.

To avoid this, we need a **variance scheduler** that controls how much noise is added at each step.  
Instead of adding full unit-variance noise every time, we add only a small fraction, gradually increasing the variance over many steps.

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

We introduce constants $\(a, b \in \mathbb{R}\)$ to scale the contribution of the previous sample and the noise:

$$
X_1 = a X_0 + b \epsilon
$$

where $\(\epsilon \sim \mathcal{N}(0,1)\)$.


---

### Finding \(b\)

We want the noise term to contribute variance $\(\beta_t\)$.  
Currently, since $\(\epsilon \sim \mathcal{N}(0,1)\)$, its variance is 1.  
To scale it properly, we set:

$$
b \epsilon = \sqrt{\beta_t} \ \epsilon
$$

so that

$$
\text{Var}(b \epsilon) = \text{Var}(\sqrt{\beta_t} \ \epsilon) = \beta_t \cdot \text{Var}(\epsilon) = \beta_t
$$

Thus:

$$
b = \sqrt{\beta_t}
$$

---


## Forward Diffusion Step

Thus, the properly scaled forward diffusion step is:

$$
X_1 = a X_0 + \sqrt{\beta_t} \ \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

- $a$ scales the contribution of the previous sample.  
- $\sqrt{\beta_t} \ \epsilon$ adds noise with the correct variance.  

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

## Introducing $\alpha_t$ and Cumulative Product

It is common to define:

$$
\alpha_t = 1 - \beta_t
$$

so that the forward step can be written as:

$$
X_t = \sqrt{\alpha_t} \, X_{t-1} + \sqrt{1 - \alpha_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,1)
$$

---

### Cumulative Product: $\bar{\alpha}_t$

To compute the distribution of $X_t$ efficiently, we define the cumulative product of $\alpha_t$:

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

This represents the **total retained signal** from the original data $X_0$ after $t$ steps.

---

### Closed-Form Expression for $X_t$

Using the cumulative product, we can write $X_t$ in **closed form**:

$$
X_t = \sqrt{\bar{\alpha}_t} \, X_0 + \sqrt{1 - \bar{\alpha}_t} \, \tilde{\epsilon}, \quad \tilde{\epsilon} \sim \mathcal{N}(0,1)
$$

Here:

- $\sqrt{\bar{\alpha}_t} \, X_0$ is the fraction of the original signal retained after $t$ steps.  
- $\sqrt{1 - \bar{\alpha}_t} \, \tilde{\epsilon}$ is the total accumulated noise.  

This formula is extremely convenient because it allows us to **sample $X_t$ directly** without iterating through all the intermediate steps.


