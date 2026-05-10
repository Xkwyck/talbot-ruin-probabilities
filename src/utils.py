import mpmath

# Funcitons to display/save results in a nice format

def print_results(arrays: list, header="", num_digits=10):
    """
    Prints the results of a list of arrays formatted in columns to the console.
    """
    ctx = mpmath.mp
    # Transpose the list of arrays into a matrix for column-wise printing
    data = ctx.matrix(arrays).T

    if header:
        print(header)

    for k in range(data.rows):
        row_strings = [ctx.nstr(data[k, j], num_digits) for j in range(data.cols)]
        print("\t".join(row_strings))


def write_results(arrays: list, filename: str, header="", sep="\t", num_digits=10):
    """
    Writes the results of a list of arrays to a text file.
    Useful for saving numerical results to the /data folder.
    """
    ctx = mpmath.mp
    data = ctx.matrix(arrays).T

    with open(filename, "w") as f:
        if header:
            f.write(header + "\n")
        for k in range(data.rows):
            # Create a string for each row with the specified separator
            row_strings = [ctx.nstr(data[k, j], num_digits) for j in range(data.cols)]
            f.write(sep.join(row_strings) + "\n")


# Helper functions for example 2: Check if a point is in the contour and approximate residues

def in_contour(z, r):
    """
    Checks if a complex point z is inside the Talbot contour of radius r.
    """
    ctx = mpmath.mp
    if z.imag / r > ctx.pi:
        return False
    return z.real < z.imag * ctx.cot(z.imag / r)


def residue(F, z0, r=1, n=1000, dps=50, **kw):
    """
    Approximates the residue of a complex function F at point z0
    using numerical integration over a small circle.
    """
    ctx = mpmath.mp
    dps_orig = ctx.dps
    ctx.dps = dps

    # Construct a circle around z0 with radius r
    phi_values = ctx.linspace(-ctx.pi, ctx.pi, n, endpoint=False)
    z_values = [z0 + r * ctx.exp(ctx.j * phi) for phi in phi_values]

    # Cauchy's Integral Formula approximation
    integral = ctx.fsum([F(z, ctx, **kw) * r * ctx.exp(ctx.j * phi) for z, phi in zip(z_values, phi_values)]) / n

    ctx.dps = dps_orig
    return integral


# Functions to compute the true ruin probability and its parameters for example 1

def get_parameters(dps=60):
    """
    Computes the parameters of the true ruin probability for example 1, cf Appendix
    """
    # Create context and set precision
    ctx = mpmath.mp
    dps_orig = ctx.dps
    ctx.dps = dps

    # Find r values: Roots of certain polynomial of degree 3
    roots = ctx.polyroots([1, -12, 42, -25])

    # Define parameters used to compute function
    beta1, beta2 = ctx.mpf('3'), ctx.mpf('4')

    # Find the C values
    M, B = ctx.matrix(3, 3), ctx.matrix([1, 1, 1])

    for j in range(3):
        M[0, j] = beta1 / (beta1 - roots[j])
        M[1, j] = beta2 / (beta2 - roots[j])
        M[2, j] = ctx.mpf(1)
    # Solve linear system (using LU decomposition)
    C = ctx.lu_solve(M, B)

    ctx.dps = dps_orig

    return roots, C


def psi_true(u, roots, C, ctx=mpmath.mp):
    """
    Computes the true ruin probability
    """
    ans = ctx.fsum(C[j] * ctx.exp(-roots[j] * u) for j in range(3))
    return ans.real


def psi_1_true(u, roots, C, q, zeta, ctx=mpmath.mp):
    """
    Computes the true ruin probability due to oscillation
    """
    ans = ctx.fsum(roots[j] * C[j] * ctx.exp(-roots[j] * u) for j in range(3))
    return ans.real / (q * zeta)


def psi_2_true(u, roots, C, q, zeta, ctx=mpmath.mp):
    """
    Computes the true ruin probability due to claim
    """
    ans = ctx.fsum((1 - roots[j] / (q * zeta)) * C[j] * ctx.exp(-roots[j] * u) for j in range(3))
    return ans.real
