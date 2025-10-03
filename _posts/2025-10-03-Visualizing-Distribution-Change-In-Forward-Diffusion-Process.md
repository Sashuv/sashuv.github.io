---
title: "Visualizing Distribution Change in Forward Diffusion Process"
date: 2025-10-03
layout: post
usemathjax: true
---

# Visualizing Distribution Change in Forward Diffusion Process

In this post, we derive how the **variance changes in a forward diffusion process** and how to properly scale the added noise.

---



Let \\(X_0\\) be the initial data. We assume:

$$
X_0 \sim \mathcal{N}(\mu, 1)
$$

That is, \\(X_0\\) follows a normal distribution with mean \\(\mu\\) and variance 1.
What we want is X0 to follow a normal distribution with mean 0 and variance 1.
---


Suppose we add Gaussian noise \\(\epsilon\\) to \\(X_0\\):

$$
X_1 = X_0 + \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

If we add the same variance at every step:

$$
\text{Var}(X_1) = \text{Var}(X_0) + \text{Var}(\epsilon) = 1 + 1 = 2
$$

This variance increases too quickly, which is undesirable.

---



To control the variance growth, we introduce a **variance scheduler** \\(\beta_t\\):

- \\(\beta_t\\) defines how noise changes over time (linear, cosine, or any schedule).  
- Typically, \\(\beta_t\\) is very small (e.g., 0.0001 to 0.02) so that noise is added gradually.

We now want the noise to satisfy:

$$
\epsilon \sim \mathcal{N}(0, \beta_t)
$$

and

$$
X_1 = X_0 + \epsilon
$$

---


Introduce constants \\(a, b \in \mathbb{R}\\) to scale the previous image and noise:

$$
X_1 = a X_0 + b \epsilon
$$

If \\(\epsilon \sim \mathcal{N}(0,1)\\), we can scale it to match the desired variance:

$$
b \epsilon = \sqrt{\beta_t} \, \epsilon \sim \mathcal{N}(0, \beta_t)
$$

because

$$
\text{Var}(\sqrt{\beta_t} \, \epsilon) = \beta_t \cdot \text{Var}(\epsilon) = \beta_t \cdot 1 = \beta_t
$$

---



The properly scaled forward diffusion step is:

$$
X_1 = a X_0 + \sqrt{\beta_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)
$$

- \\(a\\) scales the contribution of the previous image.  
- \\(\sqrt{\beta_t} \, \epsilon\\) adds noise with the correct variance.  

This ensures that the **variance increases gradually** as noise is added in each step.

So now,  

\[
\text{Var}(X_1) = \text{Var}(X_0) + \text{Var}\big(\sqrt{\beta_t} \, \epsilon\big)
\]  

\[
\text{Var}(X_1) = \text{Var}(X_0) + \beta_t \, \text{Var}(\epsilon)
\]  

\[
\text{Var}(X_1) = 1 + \beta_t
\]  

---

Now finding \(a\):  
We also scale the image at the previous timestep because there is no way we are getting \(\mu = 0\) with just this:  

\[
X_1 = a X_0 + \sqrt{\beta_t} \, \epsilon, 
\quad \epsilon \sim \mathcal{N}(0,1)
\]  

\[
\text{Var}(X_1) = \text{Var}\left(a X_0 + \sqrt{\beta_1}\,\epsilon\right) = 1
\]  

\[
\text{Var}(a X_0) + \text{Var}(\sqrt{\beta_1}\,\epsilon) = 1
\]  

\[
a^2 \, \text{Var}(X_0) + \beta_1 \, \text{Var}(\epsilon) = 1
\]  

\[
a^2 + \beta_1 = 1
\quad \Rightarrow \quad
a = \sqrt{1 - \beta_1}
\]  

So now:  

\[
X_1 = \sqrt{1-\beta_1}\,X_0 + \sqrt{\beta_1}\,\epsilon
\]  

Generally,  

\[
X_t = \sqrt{1-\beta_t}\,X_{t-1} + \sqrt{\beta_t}\,\epsilon
\]  

---

### Introducing Alphas  
To make the process more efficient, we define:  

\[
\alpha_t = 1 - \beta_t
\]  

and the cumulative product:  

\[
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
\]  

This allows us to directly write the forward diffusion at any timestep \(t\) as:  

\[
X_t = \sqrt{\bar{\alpha}_t}\,X_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon
\]  

This is efficient because instead of simulating each step iteratively, we can **sample \(X_t\) at any arbitrary timestep** in closed form using \(\bar{\alpha}_t\).




format this whole code. and also explain some parts that's missing
