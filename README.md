# Talbot-Ruin-Probabilities

This repository contains the numerical implementation and experimental results for my Master's Thesis: **"Approximating Ruin Probabilities via Talbot's Method for numerical inversion of the Laplace Transform."**

The project provides a Python framework for calculating ruin probabilities in an insurance risk model (Cramér-Lundberg model with diffusion) by applying **Talbot's Method** to invert the Laplace transform of the ruin probability function $\psi(u)$.
It is also possible to approximate the probability of ruin due to diffusion $\psi^{(1)}(u)$ and the probability due to a claim $\psi^{(2)}$.

## Overview

In ruin theory, one is interested in the probability of ruin (i.e. having no/negative capital at some point), in dependency of the initial capital $u$.
In most situations, the probability is not computable analytically.
The code provided here approximates the probabilities by inverting the Laplace transform using Talbot's Method, by implementing a version from **Abate-Valkó (2004)**.
By using arbitrary-precision arithmetic, the absolute error remains stable even when calculating extremely small probabilities (e.g. of magnitude $10^{-14}$).

### Key Features

* **Arbitrary Precision:** Uses `mpmath` to maintain numerical stability across high $M$ values.
* **Modular Architecture:** Clean separation between the inversion algorithm, the model-specific Laplace transforms, and utility helpers.
* **Multiple Distributions:** Includes transforms for Exponential, Gamma, Log-normal, and Pareto claim sizes.

## Repository Structure

```text
├── src/
│   ├── talbot.py             # Implementation of the Talbot inversion method
│   ├── laplace_transforms.py # Laplace transforms for ruin probabilities and some distributions
│   └── utils.py              # Plotting, residue calculations, data export and true probabilites for example 1
├── notebooks/
│   ├── 01_example_exponential.ipynb  # Hyperexponential case (Example 1)
│   ├── 02_example_pareto.ipynb       # Pareto distribution (Example 2)
│   └── 03_example_lognormal.ipynb    # Log-normal distribution (Example 3)
├── requirements.txt          # Project dependencies (mpmath, matplotlib, tqdm)
├── LICENSE                   # MIT License
└── README.md

```

## Mathematical Context

The Talbot method relies on a specific deformation of the Bromwich contour in the complex plane. The contour is defined by:

$$s(\theta) = r \theta (\cot \theta + i), \quad -\pi < \theta < +\pi$$

This implementation specifically solves for the ruin probability $\psi(u)$ using the transform:

$$\mathcal{L}(\psi)(s) = \frac{1}{s}\left(1 - \frac{qcs}{cs + \frac{cs^2}{\zeta} - \lambda + \lambda\mathcal{L}(f_X)(s)}\right).$$

Here $c, \zeta, q, \lambda$ are parameters of the model. Specifically,

* $c$ is the premium rate;
* $\lambda$ is the intensity of the Poisson process;
* $f_X$ is the density of a single claim $X_1$ (gives mass only to positive values);
* $\mu$ is the mean of $X_1$ (assumed to be finite);
* $\sigma^2$ is the variance of the diffusion term;
* $q = \frac{c - \lambda \mu}{c}$ is the relative security loading, assumed to be positive;
* $\zeta = \frac{2c}{\sigma^2}$.

The transformation of $\psi^{(1)}$ and $\psi^{(2)}$ are given by

$$\mathcal{L}(\psi^{(1)})(s) = \frac{1}{\zeta + s - \frac{\lambda\zeta}{cs}(1 - \mathcal{L}(f_X)(s))}.$$

$$\mathcal{L}(\psi^{(2)})(s) = \frac{\lambda (\mu s - 1 + \mathcal{L}(f_X)(s))}{s(cs + s^2\frac{c}{\zeta} - \lambda + \lambda \mathcal{L}(f_X)(s))}$$

## Quick Start


### Running an Example

Open any of the files in `notebooks/` to see the method in action. Alternatively, use the library in a script:

```python
import mpmath as mp
from src.talbot import talbot_method_abate
from src.laplace_transforms import lapl_psi, lapl_exponential

# Set precision
M = 60
ctx = mp.mp
ctx.dps = M

# Define parameters, already with high precision
lmbda = ctx.mpf('1')
c = ctx.mpf('1')
sigmasq = ctx.mpf('0.5')
mu = ctx.mpf(...) # Has to equal the expectation of the chosen distribution
q = (c - lmbda * mu)/c
zeta = 2 * c / sigmasq

# Compute one probability
u = 10
prob = talbot_method_abate(
    lapl_psi, t=u, M=60, # Parameters for talbot
    lmbda=lmbda, q=q, c=c, zeta=zeta, # Arguments for lapl_psi
    lapl_X_func=lapl_exponential # potentially add other parameters for lapl_exponential
)
print(f"Ruin probability at u={u}: {prob}")

```

For approximating $\psi(u)$, the parameters $\lambda, c, q, \zeta$ have to be passed also to `talbot_method_abate`.
For $\psi^{(1)}(u)$, $\lambda, c, \sigma^2$ are required and for $\psi^{(2)}(u)$, $\lambda, \mu, c \sigma^2$ are needed.

When appling the procedure to a different claim distribution, the implementation of $\mathcal{L}(f_X)$ should have the following signature: 
```
def lapl_X(s, ctx, ...):
```
where `ctx` is a mpmath context (ensures that the same precision is used for evaluating $\mathcal{L}(f_X)(s)$ as for the other calculations)  and `...` are distribution specific parameters.


## Citation

If you use this code or the results in your own work, please cite the original thesis:

> **R. Waller**, "Approximating Ruin Probabilities via Talbot's Method for numerical inversion of the Laplace Transform", Master's Thesis, University of Bern, 2026.

## License

This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/Xkwyck/talbot-ruin-probabilities/blob/main/LICENSE) file for details.


