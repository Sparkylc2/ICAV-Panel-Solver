import matplotlib.pyplot as plt
import numpy as np


# from claude, just a naca 4 generator for now
def naca4(designation: str, n_panels: int) -> np.ndarray:
    """
    Generate NACA 4-series airfoil with Chebyshev-spaced panels.

    Returns coordinates ordered clockwise from the LE along the upper
    surface to the TE, then back along the lower surface to the LE.
    First and last points are coincident at the LE.

    Parameters
    ----------
    designation : str
        4-digit NACA string, e.g. '2412'.
    n_panels : int
        Total number of panels (must be even).

    Returns
    -------
    coords : np.ndarray, shape (n_panels + 1, 2)
    """
    assert len(designation) == 4, "designation must be a 4-digit string"
    assert n_panels % 2 == 0, "n_panels must be even"

    m = int(designation[0]) / 100.0  # max camber
    p = int(designation[1]) / 10.0  # camber location
    t = int(designation[2:]) / 100.0  # max thickness

    n_half = n_panels // 2

    # chebyshev node spacing on [0, 1]
    beta = np.linspace(0.0, np.pi, n_half + 1)
    x = 0.5 * (1.0 - np.cos(beta))

    # thickness distribution
    yt = (
        5.0
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )

    # camber line and its gradient
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)

    if p > 0.0:
        front = x <= p
        back = ~front

        yc[front] = (m / p**2) * (2.0 * p * x[front] - x[front] ** 2)
        yc[back] = (m / (1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * x[back] - x[back] ** 2
        )

        dyc[front] = (2.0 * m / p**2) * (p - x[front])
        dyc[back] = (2.0 * m / (1.0 - p) ** 2) * (p - x[back])

    theta = np.arctan(dyc)

    # upper and lower surface coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # assemble clockwise from LE:
    #   upper (LE -> TE), then lower (TE -> LE)
    #   lower reversed from index n_half-1 to 0 (TE shared with upper)
    x_coords = np.concatenate([xu, xl[-2::-1]])
    y_coords = np.concatenate([yu, yl[-2::-1]])

    return np.column_stack([x_coords, y_coords])


U_inf = 10  # in m/s
AoA = 5  # AoA in degrees
Chord = 0.5  # in m
Density = 1.225  # in kg/m^3


# just make sure coords is a 2d vector with first column x_p and second column y_p
coords = naca4("2158", 100)

# just making sure AoA is in radians
AoA = np.radians(AoA)

# number of panels (faster to save it now)
num_panels = len(coords[:, 0]) - 1

# [x_p - x_p-1, y_p - y_p-1]
d_coords = np.diff(coords, axis=0)
# [x_p + x_p+1, y_p + y_p-1]
a_coords = coords[:-1] + coords[1:]

# panel lengths
panel_lens = np.sqrt(d_coords[:, 0] ** 2 + d_coords[:, 1] ** 2)

# control point coordinates
ctrl_point_coords = a_coords / 2

# panel angles (counterclockwise from global x-axis to inside of panel)
panel_angles = np.arctan2(d_coords[:, 1], d_coords[:, 0])

# compute these already since we use them a bit
sin_theta = np.sin(panel_angles)
cos_theta = np.cos(panel_angles)


# we now neep to for every p'th panel loop around every other q'th panel and calculate
# the normal and tangential contributions of the q'th panel on the p'th one
# The source is in panel q's frame of reference (FOR) so we first need to convert our
# coordinates from global FOR to panel q's FOR, and then once we calculated the
# induced velocity, we convert that velocity to panel p's FOR

# we can quickly compute the RHS
u_y = U_inf * np.sin(AoA - panel_angles)
u_x = U_inf * np.cos(AoA - panel_angles)

# our influence coefficient matrices
A_pq = np.zeros((num_panels, num_panels))
B_pq = np.zeros((num_panels, num_panels))


for p in range(num_panels):
    for q in range(num_panels):

        # 99% sure its this way around
        # x_p - x_q
        d_x = ctrl_point_coords[p][0] - ctrl_point_coords[q][0]

        # y_p - y_q
        d_y = ctrl_point_coords[p][1] - ctrl_point_coords[q][1]

        # the distance between the two panels in q's FOR
        x_pq = d_x * cos_theta[q] + d_y * sin_theta[q]
        y_pq = d_y * cos_theta[q] - d_x * sin_theta[q]

        # --- the velocity calculation --- #
        v_xq_num = (x_pq + panel_lens[q] / 2) ** 2 + y_pq**2
        v_xq_denom = (x_pq - panel_lens[q] / 2) ** 2 + y_pq**2

        # the tangential velocity contribution in q's FOR
        v_xq = 1 / (4 * np.pi) * np.log(v_xq_num / v_xq_denom)

        v_yq_num = y_pq * panel_lens[q]
        v_yq_denom = x_pq**2 + y_pq**2 - (panel_lens[q] / 2) ** 2

        # the normal velocity contribution in q's FOR
        v_yq = 1 / (2 * np.pi) * np.atan2(v_yq_num, v_yq_denom)

        # --- moving to p's FOR --- #
        # theta_q - theta_p
        d_theta = panel_angles[q] - panel_angles[p]

        sin_d_theta = np.sin(d_theta)
        cos_d_theta = np.cos(d_theta)

        # the normal velocity (in p's FOR)
        v_n = v_yq * cos_d_theta - v_xq * sin_d_theta

        # the tangential velocity (in p's FOR)
        v_t = v_xq * cos_d_theta + v_yq * sin_d_theta

        # --- putting into the influence matrix ---
        A_pq[p][q] = v_n
        B_pq[p][q] = v_t


# now we can solve for our first influence coefficients m_q
# basically doing v_n + U_inf*cos(alpha - theta_p) = 0, enforcing no pen
m_q = np.linalg.solve(A_pq, -u_y)

# solving for the tangential velocity for each panel
# summing the panel contributions from each panel + the freestream,
# for each panel
Q_p = B_pq @ m_q + u_x


# --- vortex part --- #
# this is so we can enforce the kutta condition
# we basically redo what we did before to get a new set of influence coefficients
# and we superimpose the new answers (plus scale a bit) to get a final tangential
# velocity for each panel, lift, pressure, etc

# we define (Circulation/[U_inf * chord] = 1), so that way later scaling is very easy
gamma = U_inf * Chord


# now we place a vortex somewhere in the airfoil
# apparently a convention is (c/4, 0)
x_v = 0.25
y_v = 0

# u_npgamma in notes
u_n_circ = np.zeros(num_panels)  # normal velocity contribution (in panel FOR)
u_t_circ = np.zeros(num_panels)  # tangential velocity contribution (in panel FOR)

for p in range(num_panels):
    x_p = ctrl_point_coords[p][0]
    y_p = ctrl_point_coords[p][1]

    # computing the velocity contribution in global FOR
    denom = (x_p - x_v) ** 2 + (y_p - y_v) ** 2
    u_g_num = y_p - y_v
    v_g_num = x_p - x_v

    # the normal and tangential velocities (global FOR)
    u_g = -gamma / (2 * np.pi) * u_g_num / denom
    v_g = gamma / (2 * np.pi) * v_g_num / denom

    # the normal and tangential velocities (in panel p's FOR)
    u_n_circ[p] = v_g * cos_theta[p] - u_g * sin_theta[p]
    u_t_circ[p] = u_g * cos_theta[p] + v_g * sin_theta[p]


# to enforce kutta we again do the same thing and invert
# we are summing normal velocity contributions from our sources,
# but adding the new normal velocity increase due to the vortex and getting some new
# coefficient strengths to get the second solution we need
m_qg = np.linalg.solve(A_pq, -u_n_circ)

# solving for the tangential velocity for each panel
# summing the panel contributions from each panel + the vortex induced velocity
# at the panel, for each panel
Q_pg = B_pq @ m_qg + np.array(u_t_circ).transpose()


# we can now superimpose our solutions
# we take the base solution and sum it with a scaled version of our new solution
# Q = Q_p + K * Q_pg
# to find the scaling factor K, we enforce the kutta-condition
# dQ = dQ_p + K * dQ_pg = 0 (dQ_p = Q_p+1 + Q_p)
te_coord_idx = np.argmax(coords[:, 0])

T = te_coord_idx - 1
dQ_te = Q_p[T] + Q_p[T + 1]
dQ_g_te = Q_pg[T] + Q_pg[T + 1]
# print(dQ_g_te)

# the scaling factor
K = -dQ_te / dQ_g_te

Q = Q_p + K * Q_pg

# we can now compute all the stuff we need
Lift = Density * U_inf**2 * K * Chord  # in N

C_L = 2 * K  # dimless

C_p = 1 - (Q / U_inf) ** 2  # vector containing panel_lens C_p values


print("=" * 70)
print(f"Lift: {Lift}")
print(f"C_L: {C_L}")
print(f"C_p: {C_p}")
print("=" * 70)
