import mpmath

# --- General Ruin Theory Model Functions --- #

def lapl_psi(s, ctx, lmbda, q, c, zeta, lapl_X_func, **kw):
    """
    Laplace transform of the ruin probability psi(u), cf eq. (3.9) in the thesis.

    Keyword arguments:
    :param s: (complex number) where to evaluate the Laplace transform
    :param ctx: mpmath context for precision
    :param lmbda: claim arrival rate (Poisson process)
    :param mu: mean claim size
    :param c: premium rate
    :param sigmasq: variance of the noise (oscillation)
    :param lapl_X_func: function that computes the Laplace transform of the claim size distribution
    :param kw: additional keyword arguments to be passed to lapl_X_func
    """
    s = ctx.convert(s)

    denom = c * s + c * s**2 / zeta - lmbda + lmbda * lapl_X_func(s, ctx, **kw)

    return (ctx.mpf('1') - q * c * s / denom) / s


def lapl_psi_1(s, ctx, lmbda, c, sigmasq, lapl_X_func, **kw):
    """
    Laplace transform of the ruin probability due to oscillation (noise), cf eq (3.13) in the thesis

    Keyword arguments:
    :param s: (complex number) where to evaluate the Laplace transform
    :param ctx: mpmath context for precision
    :param lmbda: claim arrival rate (Poisson process)
    :param c: premium rate
    :param sigmasq: variance of the noise (oscillation)
    :param lapl_X_func: function that computes the Laplace transform of the claim size distribution
    :param kw: additional keyword arguments to be passed to lapl_X_func
    """
    s = ctx.convert(s)
    zeta = ctx.mpf('2') * c / sigmasq
    denom = zeta + s - lmbda * zeta / (c * s) * (ctx.mpf('1') - lapl_X_func(s, ctx=ctx, **kw))
    return ctx.mpf('1') / denom


def lapl_psi_2(s, ctx, lmbda, mu, c, sigmasq, lapl_X_func, **kw):
    """
    Laplace transform of the ruin probability due to claims, cf eq (3.14) in the thesis

    Keyword arguments:
    :param s: (complex number) where to evaluate the Laplace transform
    :param ctx: mpmath context for precision
    :param lmbda: claim arrival rate (Poisson process)
    :param mu: mean claim size
    :param c: premium rate
    :param sigmasq: variance of the noise (oscillation)
    :param lapl_X_func: function that computes the Laplace transform of the claim size distribution
    :param kw: additional keyword arguments to be passed to lapl_X_func
    """
    s = ctx.convert(s)
    zeta = ctx.mpf('2') * c / sigmasq
    lapl_val = lapl_X_func(s, ctx=ctx, **kw)
    denom = (c * s + c * s ** 2 / zeta - lmbda * (ctx.mpf('1') - lapl_val)) * s
    return lmbda * (mu * s - ctx.mpf('1') + lapl_val) / denom


# --- Specific Distribution Transforms (lapl_X) --- #

def lapl_exponential(s, ctx, beta=1):
    """
    Laplace transform of the Exponential distribution PDF: f(x) = beta * exp(-beta * x)
    L{f(x)} = beta / (beta + s)

    Keyword arguments:
    :param s: (complex number) where to evaluate the Laplace transform
    :param ctx: mpmath context for precision
    :param beta: rate parameter of the exponential distribution
    """
    s = ctx.convert(s)
    beta_val = ctx.convert(beta)
    return beta_val / (beta_val + s)


def lapl_hyperexponential(s, ctx):
    """
    Example 1 from thesis: Combination of exponentials
    f(x) = 12 * (exp(-3x) - exp(-4x))
    """
    s = ctx.convert(s)
    return ctx.mpf('12') * (ctx.mpf('1') / (ctx.mpf('3') + s) - ctx.mpf('1') / (ctx.mpf('4') + s))


def lapl_pareto(s, ctx, alpha, x_min):
    """
    Laplace transform of the Pareto PDF: f(x) = alpha * x_min^alpha / x^(alpha + 1) for x >= x_min, 0 otherwise
    L({f(x)}) = alpha * (x_min * s)^alpha * gammainc(-alpha, s * x_min)

    Parameters:
    :param alpha: The shape parameter of the Pareto distribution.
    :param x_min: The scale parameter (minimum value) of the Pareto distribution.
    :param s: The complex frequency variable.
    """
    s = ctx.convert(s)
    return alpha * (x_min * s)**alpha * ctx.gammainc(-alpha, s * x_min)


def lapl_LN(s, ctx, mu_N=0, sigma_N=1, alpha=15, maxSum=70):
    """
    Series approximation of the Laplace transform of the Lognormal distribution
    cf https://arxiv.org/abs/1803.05878, Theorem 3

    :param s: (complex number) where to evaluate the Laplace transform
    :param mu_N: mean of the normal
    :param sigma_N: standard deviation of the normal
    :param alpha: factor that influences accuracy
    :param maxSum: where to truncate series representation
    :param ctx: mpmath context
    """
    s = ctx.convert(s)

    summands = [(-s)**n / ctx.factorial(n) * ctx.exp(mu_N * n + (n * sigma_N)**2 / 2) * 1/2 *
                ctx.erfc((mu_N + ctx.log(s / alpha) + sigma_N**2 * n) / (ctx.sqrt(2) * sigma_N)) for n in range(maxSum + 1)]
    return ctx.fsum(summands)


# Other Laplace transforms, not considered in the thesis

def lapl_gamma(s, ctx, alpha=1, beta=1):
    """
    Laplace transform of the Gamma distribution PDF.
    L{f(x)} = (beta / (beta + s))^alpha

    Keyword parameters:
    :param s: (complex number) where to evaluate the Laplace transform
    :param ctx: mpmath context for precision
    :param alpha: shape parameter of the gamma distribution
    :param beta: rate parameter of the gamma distribution
    """
    s = ctx.convert(s)
    a = ctx.convert(alpha)
    b = ctx.convert(beta)
    return (b / (b + s)) ** a
