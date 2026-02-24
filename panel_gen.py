import numpy as np

def cylinder(n_panels: int) -> np.ndarray:
    """
    generates a cylinder with n panels
    first and last points are coincident at the LE.
    """
    # get our theta values
    theta = np.linspace(2 * np.pi, 0, n_panels + 1)
    # generate the circle (has radius 0.5, and we shift so the TE is at 1)
    x_coords = 0.5 * np.cos(theta) + 0.5
    y_coords = 0.5 * np.sin(theta)
    x_coords[-1] = x_coords[0]
    y_coords[-1] = y_coords[0]
    return np.column_stack([x_coords, y_coords])


# from claude, just a naca 4 generator for now
def naca4(designation: str, n_panels: int) -> np.ndarray:
    """
    Generate NACA 4-series airfoil with Chebyshev-spaced panels.
    Returns coordinates ordered clockwise from the LE along the upper
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

    # FIX: Close the trailing edge by averaging upper/lower TE coordinates.
    # Without this, a spurious near-vertical panel bridges the TE gap,
    # breaking symmetry (e.g. NACA 0012 at 0° gave C_L ≠ 0).
    xu[-1] = (xu[-1] + xl[-1]) / 2.0
    yu[-1] = (yu[-1] + yl[-1]) / 2.0

    # assemble clockwise from LE:
    #   upper (LE -> TE), then lower (TE -> LE)
    #   lower reversed from index n_half-1 to 0 (TE shared with upper)
    x_coords = np.concatenate([xu, xl[-2::-1]])
    y_coords = np.concatenate([yu, yl[-2::-1]])

    return np.column_stack([x_coords, y_coords])
