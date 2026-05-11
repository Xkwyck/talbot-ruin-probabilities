import mpmath
from tqdm import tqdm

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


# Helper functions: Check if a point is in the contour, approximate residues and approximate number of zeroes in a region

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


def talbot_zeroes(lapl_X, c, zeta, lmbda, dps, eps, r=1, Im_max=0, Re_max=0, N_talbot=1e4, linear_rate=4, **kw):
    """
    Approximates the number of zeroes of the function D(z) = c*z + c*z^2/zeta - lmbda + lmbda*lapl_X(z) in a closed contour
    Numerically verify assumption (B1) from the thesis in this region

    :param lapl_X: laplace transform of Claim amount
    :param c: premium rate
    :param zeta: zeta parameter (2*c/sigma^2)
    :param lmbda: intensity of Poisson process
    :param dps: dps for mpmath context (precision) - may need to be increased when eps is small
    :param eps: control how far along the Talbot contour we go (up to pi - eps)
    :param r: scaling of Talbots contour
    :param Im_max: control how far up we go
    :param Re_max: control how far right we go
    :param N_talbot: number of points on Talbots contour
    :param linear_rate: number of points along linear segments per unit length
    :param kw: other keyword arguments to be passed to lapl_X
    :return: (approximate) number of zeros in a certain contour
    """

    # Create context
    ctx = mpmath.mp
    orig_dps = ctx.dps

    # May need to be increased, when eps is small
    ctx.dps = dps

    z_values = talbot_zeroes_path(ctx, eps, r, Im_max, Re_max, N_talbot, linear_rate)

    # Define function D(z) (LHS of (4.1))
    # if D(z) = 0, then z is a pole of the Laplace transform of the ruin probability
    def D(s):
        return c * s + c / zeta * s**2 - lmbda + lmbda * lapl_X(s, ctx, **kw)

    print("Evaluating D(z) along the closed bounding contour...")
    D_values = [D(z) for z in tqdm(z_values)]

    # Calculate and unwrap phase
    print("Calculating and unwrapping phase...")
    atan_vals = [ctx.atan2(z.imag, z.real) for z in D_values]

    def mpmath_unwrap(phases):
        unwrapped = [phases[0]]
        shift = ctx.mpf(0)
        pi_val = ctx.pi
        two_pi = 2 * pi_val

        for i in range(1, len(phases)):
            diff = phases[i] - phases[i-1]
            if diff < -pi_val:
                shift += two_pi
            elif diff > pi_val:
                shift -= two_pi
            unwrapped.append(phases[i] + shift)
        return unwrapped

    unwrapped_phases = mpmath_unwrap(atan_vals)

    # compute the Final Result
    # For a closed loop, the total phase shift must be EXACTLY 0 to have 0 roots inside.
    # (Note: due to computers being computers, it will be close to 0)
    total_numerical_shift = unwrapped_phases[-1] - unwrapped_phases[0]

    if abs(total_numerical_shift) > 1e-10:
        # If there are roots, the phase shift will be a multiple of 2*pi (e.g., -6.28, -12.56)
        roots_outside = round(float(abs(total_numerical_shift) / (2 * ctx.pi)))
        print(f"WARNING: Phase shifted! Detected approximately {roots_outside} root(s) OUTSIDE Talbot in the bounding region.")

    return total_numerical_shift / (2 * ctx.pi)


def talbot_zeroes_path(ctx, eps, r=1, Im_max=0, Re_max=0, N_talbot=1e3, linear_rate=2):
    """
    Helper function to generate the path for the talbot_zeros function, so that it can be plotted separately if desired.
    This is useful for debugging and visualizing the region being checked for singularities.

    :param ctx: mpmath context
    :param eps: control how far along the Talbot contour we go (up to pi - eps)
    :param r: scaling of Talbots contour
    :param Im_max: control how far up we go
    :param Re_max: control how far right we go
    :param N_talbot: number of points on Talbots contour
    :param linear_rate: number of points along linear segments per unit length
    :return: points of contour
    """

    # Generate curve:
    # 1. Define Talbot contour segment
    def contour(theta):
        return r * theta * ctx.mpc(ctx.cot(theta), 1)

    # Generate theta values for the Talbot curve (from 0 to pi-eps)
    # This starts at r+0i and moves into the upper LHP
    theta_values = ctx.linspace(1e-12, ctx.pi - eps, N_talbot)
    path_gamma = [contour(t) for t in theta_values]
    path_gamma[0] = ctx.mpc(r, 0)

    # 2. Define the Linear Segments
    # Point A: The end of the Talbot contour (-R + ih)
    z_A = path_gamma[-1]
    Re_min = z_A.real

    # Point B: Straight up to -Re_min + iIm_max
    if not Im_max:  # If not specified, make a square
        Im_max = -Re_min
    z_B = ctx.mpc(Re_min, Im_max)

    # Point C: Straight across to Re_max + iIm_max
    if not Re_max or Re_max < r:
        Re_max = 2 * r
    z_C = ctx.mpc(Re_max, Im_max)

    # Point D: Straight down to Re_max + 0i
    z_D = ctx.mpc(Re_max, 0)

    # Point E: Back to start (r + 0i)
    z_E = ctx.mpc(r, 0)

    # Helper to create linear paths
    def get_line(start, end, N):
        return [start + (end - start) * t for t in ctx.linspace(0, 1, N)]

    # 3. Combine paths
    # Note: We slice [1:] to avoid duplicating the joint pixels
    path_AB = get_line(z_A, z_B, int((z_B.imag - z_A.imag) * linear_rate))[1:]
    path_BC = get_line(z_B, z_C, int((z_C.real - z_B.real) * linear_rate))[1:]
    path_CD = get_line(z_C, z_D, int((z_C.imag - z_D.imag) * linear_rate))[1:]
    path_DE = get_line(z_D, z_E, int((z_D.real - z_A.real) * linear_rate))[1:]

    z_values = path_gamma + path_AB + path_BC + path_CD + path_DE

    return z_values


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
