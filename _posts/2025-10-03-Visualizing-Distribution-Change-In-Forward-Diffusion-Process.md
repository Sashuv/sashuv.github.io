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

Let \(X_0\) be the initial data. We assume:

$$
X_0 \sim \mathcal{N}(\mu, 1)
$$

That is, \(X_0\) follows a normal distribution with mean \(\mu\) and variance 1.  
For simplicity in diffusion models, we often want \(\mu = 0\) and \(\text{Var}(X_0) = 1\).

---

## Step 2: Adding Noise

Suppose we add Gaussian noise \(\epsilon\) to \(X_0\):

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

To control the variance growth, we introduce a **variance scheduler** \(\beta_t\):

- \(\beta_t\) defines how noise changes over time (linear, cosine, or other schedules).  
- Typically, \(\beta_t\) is very small (e.g., 0.0001 to 0.02) so that noise is added gradually.

Now we want the noise to satisfy:

$$
\epsilon \sim \mathcal{N}(0, \beta_t)
$$

and the forward step becomes:

$$
X_1 = X_0 + \epsilon
$$

---

## Step 4: Scaling Noise and Previous Image

Introduce constants \(a, b \in \mathbb{R}\) to scale the previous image and noise:

$$
X_1 = a X_0 + b \epsilon
$$

If \(\epsilon \sim \mathcal{N}(0,1)\), we can scale it to match the desired variance:

$$
b \epsilon = \sqrt{\beta_t} \, \epsilon \sim \mathcal{N}(0, \beta_t)
$$

because

$$
\text{Var}(\sqrt{\beta_t} \, \epsilon) = \beta_t \cdot \text{Var}(\epsilon) = \beta_t
$$

Thus the forward step can be written as:

$$
X_1 = a X_0 + \sqrt{\beta_t} \, \epsilon
$$

---

## Step 5: Adjusting \(a\) for Variance Preservation

We want the resulting variance to stay 1 (so the distribution doesn't blow up):

\[
\text{Var}(X_1) = \text{Var}(a X_0 + \sqrt{\beta_t} \, \epsilon) = a^2 \text{Var}(X_0) + \beta_t \text{Var}(\epsilon)
\]

Assuming \(\text{Var}(X_0) = 1\) and \(\text{Var}(\epsilon) = 1\), we get:

\[
a^2 + \beta_t = 1
\]

\[
a = \sqrt{1 - \beta_t}
\]

So the final forward diffusion step becomes:

$$
X_1 = \sqrt{1-\beta_1} \, X_0 + \sqrt{\beta_1} \, \epsilon
$$

Generally, for step \(t\):

$$
X_t = \sqrt{1-\beta_t} \, X_{t-1} + \sqrt{\beta_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

---

## Step 6: Introducing \(\alpha_t\) for Efficiency

Instead of writing \(\sqrt{1-\beta_t}\) every step, we define:

$$
\alpha_t = 1 - \beta_t
$$

So the forward step becomes:

$$
X_t = \sqrt{\alpha_t} \, X_{t-1} + \sqrt{1-\alpha_t} \, \epsilon
$$

- Using \(\alpha_t\) simplifies computations and makes code implementation easier.  
- It also allows **precomputing cumulative products** over time for efficient sampling:

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

Then the forward process can be sampled directly from:

$$
X_t = \sqrt{\bar{\alpha}_t} \, X_0 + \sqrt{1-\bar{\alpha}_t} \, \epsilon
$$

This avoids looping through all intermediate steps and is much more **computationally efficient**.

---

## Summary

- Noise is gradually added to the original data using a small \(\beta_t\).  
- Scaling constants \(a = \sqrt{1-\beta_t}\) and \(b = \sqrt{\beta_t}\) preserve variance.  
- \(\alpha_t = 1-\beta_t\) simplifies calculations.  
- Using cumulative \(\bar{\alpha}_t\) allows **direct sampling of \(X_t\)** without iterating through all steps.  

This formulation is the foundation for **forward diffusion in DDPMs** (Denoising Diffusion Probabilistic Models).

---

*Optional Visualization:*

![Forward Diffusion](plot1.png)

*Figure: Gradual variance increase in the forward diffusion process.*
