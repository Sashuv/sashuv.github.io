---
layout: post
title: "Residual Connections & Gradient Flow"
date: 2025-05-30
tags: [deep learning, PyTorch, neural networks, residual connections]
---

Do they actually help? I measured it myself.  

You’ve probably heard the hype: Residual Connections help deep networks train better. They’re what made ResNets possible, solved vanishing gradients, and let us go deeper than ever before.

But honestly, I didn’t want to just read that. I wanted to see it. Measure it. Feel it in the gradients.

So I decided to build a tiny experiment in PyTorch to compare two deep fully connected networks:

- one with residual connections  
- one without  

And then look at the mean gradient magnitudes layer by layer, to see what’s really going on under the hood.

---

## The Setup

I created a 5-layer fully connected neural network (no bias terms), with ReLU activations between each layer.

Then I made a toggle: you can turn residual connections on or off. When they’re on, each layer adds a skip connection from the previous input — like in ResNets.

Then I passed a random input, did a forward and backward pass, and printed the mean absolute value of the gradients of each layer’s weights.

---

## Why Gradient Magnitudes?

Because if gradients vanish — especially in early layers — the network can’t learn. You might not see exploding loss or NaNs, but it just won’t train well.

Gradient magnitude is a simple but powerful signal:

- If it’s near zero, learning in that layer is basically frozen.  
- If it’s large and healthy — good news, your network might actually learn something.

---

## The Results

| Layer               | Without Residuals | With Residuals |
|--------------------|-----------------|----------------|
| layers.0.weight     | 4.81e-05        | 0.01099        |
| layers.1.weight     | 1.10e-05        | 0.00601        |
| layers.2.weight     | 1.67e-05        | 0.01124        |
| layers.3.weight     | 3.22e-05        | 0.00882        |
| layers.4.weight     | 1.35e-04        | 0.10311        |

In the residual version, the gradients are **orders of magnitude larger** — especially in the early layers, which is where gradients usually vanish in deep nets.

---

## What This Tells Me

Residual connections aren’t just a theoretical trick. They really do help gradients flow better through the network.

They create **shortcut paths** that let the gradients skip backward through the network more easily, instead of getting squashed by layers of matrix multiplication and ReLU.

---

## The Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, shortcut_connections=False):
        super().__init__()
        self.shortcut_connections = shortcut_connections
        self.layers = nn.ModuleList([
            nn.Linear(10, 10, bias=False) for _ in range(4)
        ] + [nn.Linear(10, 1, bias=False)])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.shortcut_connections and i < len(self.layers) - 1:
                residual = x
                x = F.relu(layer(x)) + residual
            else:
                x = layer(x) if i == len(self.layers) - 1 else F.relu(layer(x))
        return x


model = NeuralNetwork(shortcut_connections=True)  # or False
out = model(input)
loss = criterion(out, target)
loss.backward()

for name, param in model.named_parameters():
    print(f"{name}: {param.grad.abs().mean().item()}")
