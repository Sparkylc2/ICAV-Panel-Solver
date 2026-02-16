from typing import NamedTuple, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────


class PanelGeometry(NamedTuple):
    nodes: np.ndarray  # (N+1, 2) — clockwise node coordinates
    ds: np.ndarray  # (N,)    — panel lengths
    xc: np.ndarray  # (N,)    — control point x
    yc: np.ndarray  # (N,)    — control point y
    theta: np.ndarray  # (N,)    — panel angle (atan2 from node p-1 to p)
    n_panels: int


class PanelSolution(NamedTuple):
    cp: np.ndarray  # (N,) pressure coefficient at each control point
    q: np.ndarray  # (N,) surface tangential velocity / U_inf
    cl: float  # lift coefficient
    m: np.ndarray  # (N,) source densities


# ──────────────────────────────────────────────────────────────
#  NACA 4-digit airfoil generator
# ──────────────────────────────────────────────────────────────


def naca4_points(code: str, n_panels: int, closed_te: bool = True) -> np.ndarray:
    """
    Parameters
    ----------
    code : str
        4-digit NACA code, e.g. "0012", "2412".
    n_panels : int
        Total number of panels (must be even). Half on upper, half on lower.
    closed_te : bool
        If True, use the coefficient that closes the trailing edge.

    Returns
    -------
    nodes : ndarray, shape (n_panels + 1, 2)
        Airfoil coordinates in clockwise order (TE -> lower -> LE -> upper -> TE).
    """
    assert len(code) == 4, "NACA code must be 4 digits"
    assert n_panels % 2 == 0, "n_panels must be even"

    max_camber = int(code[0]) / 100.0
    camber_loc = int(code[1]) / 10.0
    thickness = int(code[2:]) / 100.0

    n_half = n_panels // 2

    # Chebyshev spacing: cluster points near LE and TE
    beta = np.linspace(0.0, np.pi, n_half + 1)
    x = 0.5 * (1.0 - np.cos(beta))  # 0 -> 1

    # Thickness distribution (NACA standard)
    a4 = -0.1015 if closed_te else -0.1036
    yt = (
        5.0
        * thickness
        * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    )

    # Camber line and gradient
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    if max_camber > 0.0 and camber_loc > 0.0:
        p = camber_loc
        m = max_camber
        front = x < p
        rear = ~front

        yc[front] = (m / p**2) * (2.0 * p * x[front] - x[front] ** 2)
        dyc[front] = (2.0 * m / p**2) * (p - x[front])

        yc[rear] = (m / (1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * x[rear] - x[rear] ** 2
        )
        dyc[rear] = (2.0 * m / (1.0 - p) ** 2) * (p - x[rear])

    angle = np.arctan(dyc)

    # Upper and lower surface coordinates
    xu = x - yt * np.sin(angle)
    yu = yc + yt * np.cos(angle)
    xl = x + yt * np.sin(angle)
    yl = yc - yt * np.cos(angle)

    # Assemble clockwise: TE_lower -> LE -> TE_upper
    # Lower surface: reverse so x goes from 1 -> 0
    x_lower = xl[::-1]  # TE to LE
    y_lower = yl[::-1]

    # Upper surface: LE to TE (skip LE to avoid duplicate)
    x_upper = xu[1:]
    y_upper = yu[1:]

    nodes_x = np.concatenate([x_lower, x_upper])
    nodes_y = np.concatenate([y_lower, y_upper])

    return np.column_stack([nodes_x, nodes_y])


# ──────────────────────────────────────────────────────────────
#  Panel geometry
# ──────────────────────────────────────────────────────────────


def compute_geometry(nodes: np.ndarray) -> PanelGeometry:
    """Compute panel lengths, control points, and angles from node array.

    Parameters
    ----------
    nodes : ndarray, shape (N+1, 2)
        Clockwise-ordered body coordinates. First and last may coincide
        (closed body).

    Returns
    -------
    PanelGeometry
    """
    n = len(nodes) - 1
    dx = np.diff(nodes[:, 0])
    dy = np.diff(nodes[:, 1])

    ds = np.sqrt(dx**2 + dy**2)
    xc = 0.5 * (nodes[:-1, 0] + nodes[1:, 0])
    yc = 0.5 * (nodes[:-1, 1] + nodes[1:, 1])
    theta = np.arctan2(dy, dx)

    return PanelGeometry(nodes=nodes, ds=ds, xc=xc, yc=yc, theta=theta, n_panels=n)


# ──────────────────────────────────────────────────────────────
#  Influence coefficients
# ──────────────────────────────────────────────────────────────


def _source_velocity_local(
    x_pq: np.ndarray, y_pq: np.ndarray, ds_q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute velocity (per unit source density) in element q's local frame.

    From the notes equations (1) and (2):
        v_xq = (1/4pi) ln[ ((x + ds/2)^2 + y^2) / ((x - ds/2)^2 + y^2) ]
        v_yq = (1/2pi) atan2( y*ds, x^2 + y^2 - (ds/2)^2 )

    Parameters
    ----------
    x_pq, y_pq : ndarray, shape (N, N)
        Local coordinates of control point p in element q's frame.
    ds_q : ndarray, shape (N,)
        Panel lengths (broadcast over axis 1).

    Returns
    -------
    vx, vy : ndarray, shape (N, N)
        Velocity components per unit source density, in q's local frame.
    """
    half_ds = 0.5 * ds_q[np.newaxis, :]  # (1, N) for broadcasting

    x_plus = x_pq + half_ds
    x_minus = x_pq - half_ds
    y2 = y_pq**2

    num = x_plus**2 + y2
    den = x_minus**2 + y2

    # Guard against log(0) -- only happens at self-influence, handled separately
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = (1.0 / (4.0 * np.pi)) * np.log(num / den)

    # atan2 formulation for v_y (more robust than atan with division)
    vy = (1.0 / (2.0 * np.pi)) * np.arctan2(
        y_pq * ds_q[np.newaxis, :],
        x_pq**2 + y2 - half_ds**2,
    )

    return vx, vy


def compute_influence_coefficients(
    geom: PanelGeometry,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the normal (A) and tangential (B) influence coefficient matrices.

    A[p,q] gives the normal velocity at control point p per unit source
    density on panel q. B[p,q] gives the tangential velocity.

    Uses outward-pointing normal convention (consistent with the notes'
    boundary condition:  sum A*m = -U_inf sin(alpha - theta_p)).

    Self-influence: A[p,p] = +0.5,  B[p,p] = 0.

    Parameters
    ----------
    geom : PanelGeometry

    Returns
    -------
    A, B : ndarray, shape (N, N)
    """
    n = geom.n_panels
    xc, yc, theta, ds = geom.xc, geom.yc, geom.theta, geom.ds

    # Relative positions of control point p w.r.t. control point q
    dx = xc[:, np.newaxis] - xc[np.newaxis, :]  # (N, N)
    dy = yc[:, np.newaxis] - yc[np.newaxis, :]

    cos_q = np.cos(theta)[np.newaxis, :]  # (1, N)
    sin_q = np.sin(theta)[np.newaxis, :]

    # Rotate into element q's local frame
    # Local x along panel tangent, local y to the LEFT of tangent (outward for CW)
    x_pq = dx * cos_q + dy * sin_q
    y_pq = -dx * sin_q + dy * cos_q

    # Source velocity in q's local frame (per unit density)
    vx, vy = _source_velocity_local(x_pq, y_pq, ds)

    # Angle difference: delta = theta_q - theta_p
    dtheta = theta[np.newaxis, :] - theta[:, np.newaxis]  # (N, N): [p, q]
    cos_d = np.cos(dtheta)
    sin_d = np.sin(dtheta)

    # Project into panel p's normal and tangential directions.
    # Outward normal at p in q's local frame: (sin(theta_q - theta_p), cos(theta_q - theta_p))
    # v_n = vx * sin(delta) + vy * cos(delta)
    # v_t = vx * cos(delta) - vy * sin(delta)
    A = vx * sin_d + vy * cos_d
    B = vx * cos_d - vy * sin_d

    # Self-influence (diagonal): known jump conditions for source panel
    np.fill_diagonal(A, 0.5)
    np.fill_diagonal(B, 0.0)

    return A, B


# ──────────────────────────────────────────────────────────────
#  Non-lifting solver
# ──────────────────────────────────────────────────────────────


def solve_nonlifting(
    geom: PanelGeometry,
    A: np.ndarray,
    B: np.ndarray,
    alpha_deg: float,
    u_inf: float = 1.0,
) -> PanelSolution:
    """Solve the source-only (non-lifting) panel method.

    Boundary condition (no penetration):
        sum_q A[p,q] m[q] = -U_inf sin(alpha - theta_p)

    Surface tangential velocity:
        Q[p] = sum_q B[p,q] m[q] + U_inf cos(alpha - theta_p)

    Parameters
    ----------
    geom : PanelGeometry
    A, B : ndarray, shape (N, N)
    alpha_deg : float -- angle of attack in degrees
    u_inf : float -- free stream speed

    Returns
    -------
    PanelSolution
    """
    alpha = np.radians(alpha_deg)
    theta = geom.theta

    rhs = -u_inf * np.sin(alpha - theta)

    m = np.linalg.solve(A, rhs)

    q = B @ m + u_inf * np.cos(alpha - theta)
    cp = 1.0 - (q / u_inf) ** 2
    cl = 0.0  # non-lifting by construction

    return PanelSolution(cp=cp, q=q, cl=cl, m=m)


# ──────────────────────────────────────────────────────────────
#  Lifting solver (Kutta condition via internal point vortex)
# ──────────────────────────────────────────────────────────────


def _vortex_velocity(
    xc: np.ndarray,
    yc: np.ndarray,
    xv: float,
    yv: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity induced by a point vortex at (xv, yv) with circulation gamma.

    Returns (vx, vy) in global coordinates at each control point.
    """
    dx = xc - xv
    dy = yc - yv
    r2 = dx**2 + dy**2
    vx = -gamma * dy / (2.0 * np.pi * r2)
    vy = gamma * dx / (2.0 * np.pi * r2)
    return vx, vy


def solve_lifting(
    geom: PanelGeometry,
    A: np.ndarray,
    B: np.ndarray,
    alpha_deg: float,
    u_inf: float = 1.0,
    chord: float = 1.0,
    vortex_loc: Tuple[float, float] | None = None,
) -> PanelSolution:
    """Solve the panel method with Kutta condition enforced at the trailing edge.

    Procedure (from the notes):
    1. Solve non-lifting problem -> m_q, Q_p, dQ_te
    2. Place unit vortex (Gamma = U_inf * c) inside the aerofoil
    3. Solve for source strengths m_Gamma_q that make the surface a streamline
       for the vortex-induced flow alone
    4. Compute Q_pGamma, dQ_Gamma_te
    5. Choose K = -dQ_te / dQ_Gamma_te so that Kutta condition is satisfied
    6. Final velocity: Q_hat_p = Q_p + K * Q_pGamma
    7. C_L = 2K

    Parameters
    ----------
    geom : PanelGeometry
    A, B : ndarray, shape (N, N)
    alpha_deg : float
    u_inf : float
    chord : float
    vortex_loc : (float, float) or None
        Location of the internal point vortex. Default: quarter-chord.

    Returns
    -------
    PanelSolution
    """
    n = geom.n_panels
    theta = geom.theta

    # Step 1: non-lifting solution
    sol_nl = solve_nonlifting(geom, A, B, alpha_deg, u_inf)
    q_nl = sol_nl.q

    # Trailing edge panels: panel 0 is first lower-surface panel from TE,
    # panel N-1 is last upper-surface panel ending at TE.
    # Q_T + Q_{T+1} in the notes' convention.
    delta_q_te = q_nl[-1] + q_nl[0]

    # Step 2-3: unit vortex solution
    if vortex_loc is None:
        vortex_loc = (0.25 * chord, 0.0)
    xv, yv = vortex_loc

    gamma_unit = u_inf * chord  # Gamma/(U_inf * c) = 1

    # Velocity at control points due to the unit vortex
    vx_vort, vy_vort = _vortex_velocity(geom.xc, geom.yc, xv, yv, gamma_unit)

    # Normal and tangential components of vortex velocity at each panel
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Outward normal n_hat = (-sin theta, cos theta)
    u_n_vort = -vx_vort * sin_t + vy_vort * cos_t
    u_t_vort = vx_vort * cos_t + vy_vort * sin_t

    # Source strengths for the vortex: A * m_Gamma = -U_nGamma
    rhs_vort = -u_n_vort
    m_gamma = np.linalg.solve(A, rhs_vort)

    # Tangential velocity from vortex sources + vortex itself
    q_gamma = B @ m_gamma + u_t_vort

    delta_q_gamma_te = q_gamma[-1] + q_gamma[0]

    # Step 5: Kutta condition
    if abs(delta_q_gamma_te) < 1e-14:
        K = 0.0
    else:
        K = -delta_q_te / delta_q_gamma_te

    # Step 6: combined solution
    q_total = q_nl + K * q_gamma
    m_total = sol_nl.m + K * m_gamma
    cp = 1.0 - (q_total / u_inf) ** 2
    cl = 2.0 * K

    return PanelSolution(cp=cp, q=q_total, cl=cl, m=m_total)


# ──────────────────────────────────────────────────────────────
#  Superposition solver (Note 4: alpha=0, alpha=90, Gamma)
# ──────────────────────────────────────────────────────────────


def precompute_superposition(
    geom: PanelGeometry,
    A: np.ndarray,
    B: np.ndarray,
    u_inf: float = 1.0,
    chord: float = 1.0,
    vortex_loc: Tuple[float, float] | None = None,
) -> dict:
    """Precompute the three base solutions (alpha=0, alpha=90, unit Gamma).

    After calling this, use ``solve_alpha`` to get the solution at any alpha
    without re-inverting the matrix.

    Returns
    -------
    dict with keys: q_0, q_90, q_gamma, dq_te_0, dq_te_90, dq_te_gamma,
    m_0, m_90, m_gamma
    """
    theta = geom.theta

    if vortex_loc is None:
        vortex_loc = (0.25 * chord, 0.0)
    xv, yv = vortex_loc
    gamma_unit = u_inf * chord

    # alpha = 0
    rhs_0 = -u_inf * np.sin(-theta)  # sin(0 - theta) = -sin(theta)
    m_0 = np.linalg.solve(A, rhs_0)
    q_0 = B @ m_0 + u_inf * np.cos(-theta)

    # alpha = 90
    rhs_90 = -u_inf * np.sin(np.pi / 2.0 - theta)
    m_90 = np.linalg.solve(A, rhs_90)
    q_90 = B @ m_90 + u_inf * np.cos(np.pi / 2.0 - theta)

    # Unit vortex
    vx_v, vy_v = _vortex_velocity(geom.xc, geom.yc, xv, yv, gamma_unit)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    u_n_v = -vx_v * sin_t + vy_v * cos_t
    u_t_v = vx_v * cos_t + vy_v * sin_t
    m_g = np.linalg.solve(A, -u_n_v)
    q_g = B @ m_g + u_t_v

    return {
        "q_0": q_0,
        "q_90": q_90,
        "q_gamma": q_g,
        "dq_te_0": q_0[-1] + q_0[0],
        "dq_te_90": q_90[-1] + q_90[0],
        "dq_te_gamma": q_g[-1] + q_g[0],
        "m_0": m_0,
        "m_90": m_90,
        "m_gamma": m_g,
    }


def solve_alpha(
    pre: dict, alpha_deg: float, u_inf: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fast solution at arbitrary alpha from precomputed base solutions.

    Returns (cp, q, cl).
    """
    alpha = np.radians(alpha_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)

    q_freestream = pre["q_0"] * ca + pre["q_90"] * sa
    dq_te = pre["dq_te_0"] * ca + pre["dq_te_90"] * sa
    dq_te_g = pre["dq_te_gamma"]

    K = -dq_te / dq_te_g if abs(dq_te_g) > 1e-14 else 0.0

    q = q_freestream + K * pre["q_gamma"]
    cp = 1.0 - (q / u_inf) ** 2
    cl = 2.0 * K

    return cp, q, cl


# ──────────────────────────────────────────────────────────────
#  Utility: circle geometry (for validation)
# ──────────────────────────────────────────────────────────────


def circle_nodes(n_panels: int, radius: float = 1.0) -> np.ndarray:
    """Generate clockwise-ordered nodes on a circle of given radius."""
    # Clockwise: angles decrease from 0 to -2pi
    angles = np.linspace(0.0, -2.0 * np.pi, n_panels + 1)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])


# ──────────────────────────────────────────────────────────────
#  Test suite
# ──────────────────────────────────────────────────────────────


def run_tests() -> None:
    print("=" * 64)
    print("  Panel Method -- Validation Tests")
    print("=" * 64)
    all_passed = True

    # ──────────────────────────────────────────
    #  Test 1: Circular cylinder, non-lifting
    # ──────────────────────────────────────────
    print("\n[Test 1] Circular cylinder, non-lifting, alpha = 0")
    n_cyl = 100
    nodes_cyl = circle_nodes(n_cyl, radius=1.0)
    geom_cyl = compute_geometry(nodes_cyl)
    A_cyl, B_cyl = compute_influence_coefficients(geom_cyl)

    sol_cyl = solve_nonlifting(geom_cyl, A_cyl, B_cyl, alpha_deg=0.0)

    # Analytical Cp for non-lifting cylinder: Cp = 1 - 4 sin^2(theta)
    angles_cp = np.arctan2(geom_cyl.yc, geom_cyl.xc)
    cp_exact = 1.0 - 4.0 * np.sin(angles_cp) ** 2

    err_cp = np.max(np.abs(sol_cyl.cp - cp_exact))
    passed = err_cp < 0.02
    status = "PASS" if passed else "FAIL"
    print(f"  Max |Cp_numerical - Cp_exact| = {err_cp:.6f}  [{status}]")
    all_passed &= passed

    # Check source strengths sum to zero (closed body)
    source_sum = np.sum(sol_cyl.m * geom_cyl.ds)
    print(f"  sum(m*ds) = {source_sum:.2e}  (should be ~ 0)")

    # ──────────────────────────────────────────
    #  Test 2: Circular cylinder at alpha = 10
    # ──────────────────────────────────────────
    print("\n[Test 2] Circular cylinder, non-lifting, alpha = 10")
    alpha_test = 10.0
    sol_cyl10 = solve_nonlifting(geom_cyl, A_cyl, B_cyl, alpha_deg=alpha_test)

    alpha_rad = np.radians(alpha_test)
    cp_exact_10 = 1.0 - 4.0 * np.sin(angles_cp - alpha_rad) ** 2

    err_cp10 = np.max(np.abs(sol_cyl10.cp - cp_exact_10))
    passed = err_cp10 < 0.02
    status = "PASS" if passed else "FAIL"
    print(f"  Max |Cp error| = {err_cp10:.6f}  [{status}]")
    all_passed &= passed

    # ──────────────────────────────────────────
    #  Test 3: NACA 0012, alpha = 0 (symmetric -> CL = 0)
    # ──────────────────────────────────────────
    print("\n[Test 3] NACA 0012, non-lifting, alpha = 0 (symmetry check)")
    n_af = 120
    nodes_0012 = naca4_points("0012", n_af)
    geom_0012 = compute_geometry(nodes_0012)
    A_0012, B_0012 = compute_influence_coefficients(geom_0012)

    sol_0012_nl = solve_nonlifting(geom_0012, A_0012, B_0012, alpha_deg=0.0)

    # Cp should be symmetric about the chord line
    n_half = n_af // 2
    cp_lower = sol_0012_nl.cp[:n_half][::-1]
    cp_upper = sol_0012_nl.cp[n_half:]

    sym_err = np.max(np.abs(cp_lower - cp_upper))
    passed = sym_err < 1e-10
    status = "PASS" if passed else "FAIL"
    print(f"  Max |Cp_lower - Cp_upper| = {sym_err:.2e}  [{status}]")
    all_passed &= passed

    # ──────────────────────────────────────────
    #  Test 4: NACA 0012, lifting, alpha = 5
    # ──────────────────────────────────────────
    print("\n[Test 4] NACA 0012, lifting, alpha = 5")
    sol_0012_5 = solve_lifting(geom_0012, A_0012, B_0012, alpha_deg=5.0, chord=1.0)

    # Thin airfoil theory: CL = 2*pi*sin(alpha)
    cl_tat = 2.0 * np.pi * np.sin(np.radians(5.0))
    cl_err = abs(sol_0012_5.cl - cl_tat) / cl_tat * 100.0
    passed = cl_err < 10.0  # within 10% of thin airfoil theory
    status = "PASS" if passed else "FAIL"
    print(f"  CL (panel)        = {sol_0012_5.cl:.4f}")
    print(f"  CL (thin airfoil) = {cl_tat:.4f}")
    print(f"  Relative error    = {cl_err:.1f}%  [{status}]")
    all_passed &= passed

    # ──────────────────────────────────────────
    #  Test 5: NACA 2412, CL vs alpha (linearity)
    # ──────────────────────────────────────────
    print("\n[Test 5] NACA 2412, CL vs alpha (linearity check)")
    nodes_2412 = naca4_points("2412", 120)
    geom_2412 = compute_geometry(nodes_2412)
    A_2412, B_2412 = compute_influence_coefficients(geom_2412)
    pre_2412 = precompute_superposition(geom_2412, A_2412, B_2412)

    alphas = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0])
    cls = np.array([solve_alpha(pre_2412, a)[2] for a in alphas])

    # Fit a line: CL = a0 + a1*alpha -> a1 should be ~ 2*pi/rad
    coeffs = np.polyfit(alphas, cls, 1)
    cl_alpha_deg = coeffs[0]
    cl_alpha_rad = cl_alpha_deg * 180.0 / np.pi
    expected_slope = 2.0 * np.pi  # per radian

    slope_err = abs(cl_alpha_rad - expected_slope) / expected_slope * 100.0
    passed = slope_err < 8.0
    status = "PASS" if passed else "FAIL"
    print(f"  dCL/dalpha = {cl_alpha_rad:.4f} /rad  (expected ~ {expected_slope:.4f})")
    print(f"  Slope error = {slope_err:.1f}%  [{status}]")
    all_passed &= passed

    # Print CL values
    print(f"  {'alpha':>8s} {'CL':>10s}")
    for a, c in zip(alphas, cls):
        print(f"  {a:8.1f} {c:10.4f}")

    # ──────────────────────────────────────────
    #  Test 6: Superposition consistency
    # ──────────────────────────────────────────
    print("\n[Test 6] Superposition vs direct solve (NACA 2412, alpha = 5)")
    sol_direct = solve_lifting(geom_2412, A_2412, B_2412, alpha_deg=5.0)
    cp_super, q_super, cl_super = solve_alpha(pre_2412, 5.0)

    cp_diff = np.max(np.abs(sol_direct.cp - cp_super))
    cl_diff = abs(sol_direct.cl - cl_super)
    passed = cp_diff < 1e-10 and cl_diff < 1e-10
    status = "PASS" if passed else "FAIL"
    print(f"  Max |Cp_direct - Cp_super| = {cp_diff:.2e}")
    print(f"  |CL_direct - CL_super|     = {cl_diff:.2e}  [{status}]")
    all_passed &= passed

    # ──────────────────────────────────────────
    #  Test 7: Source strength conservation
    # ──────────────────────────────────────────
    print("\n[Test 7] Source conservation (NACA 0012, non-lifting)")
    src_sum = np.sum(sol_0012_nl.m * geom_0012.ds)
    passed = abs(src_sum) < 1e-10
    status = "PASS" if passed else "FAIL"
    print(f"  sum(m*ds) = {src_sum:.2e}  [{status}]")
    all_passed &= passed

    # ──────────────────────────────────────────
    print("\n" + "=" * 64)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 64)


run_tests()
