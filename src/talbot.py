import mpmath


def talbot_method_abate(f, t, M=50, r=0, **kw):
    """
    Talbot's Method as described in Abate, Valko (2004)

    :param f: the laplace transform to be inverted
    :param t: time at which to evaluate the inverse laplace transform
    :param M: number of terms in the sum
    :param r: (optional) radius of the contour
    :param kw: keyword arguments to be passed to f
    :return: numerical approximation of the inverse laplace transform at time t
    """
    # Create a local context to ensure precision is encapsulated
    ctx = mpmath.mp

    # Set precision of the context
    orig_dps = ctx.dps
    ctx.dps = int(M)

    t = ctx.convert(t)

    # If no r was specified, use rule of thumb
    if r == 0:
        r = ctx.fraction('2', '5') * M

    theta = ctx.linspace(0, ctx.pi, M + 1)        # Angles to consider

    # Calculate points on contour and cot of theta
    pts_exp = [ctx.mpf(r)] * M                    # Points for exp
    pts_f = [pts_exp[0] / t] * M                  # Points for f
    cot_theta = [ctx.mpf(0)] * M                  # Cotan of thetas

    # Compute remaining points
    for i in range(1, M):
        cot_theta[i] = ctx.cot(theta[i])
        pts_exp[i] = r * theta[i] * (cot_theta[i] + ctx.j)
        pts_f[i] = pts_exp[i] / t

    # Evaluate f (use same context for evaluation)
    f_val = [f(p, ctx=ctx, **kw) for p in pts_f]

    # Calculate time domain solution
    ans = [ctx.mpc(0)] * M                        # Save values of the integrand
    ans[0] = ctx.exp(pts_exp[0]) * f_val[0] / 2   # 0th therm is scaled by .5

    for i in range(1, M):                         # Compute remaining terms
        # Compute weight
        weight = (ctx.mpf('1') + ctx.j * theta[i] * (ctx.mpf('1') + cot_theta[i] ** 2) - ctx.j * cot_theta[i])
        ans[i] = ctx.exp(pts_exp[i]) * f_val[i] * weight

    # Sum all values and apply final scaling
    result = ctx.fraction('2', '5') * ctx.fsum(ans) / t

    # Restore original precision
    ctx.dps = orig_dps

    return result.real