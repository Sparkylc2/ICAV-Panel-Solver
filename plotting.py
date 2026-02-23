import matplotlib.pyplot as plt
import numpy as np

from structs import (GeometryConfig, InfluenceData, PanelInfo, SolverResult,
                     SolverState)


def add_geometry(geometry_cfg: GeometryConfig) -> None:
    coords = geometry_cfg.coords
    plt.plot(coords[:, 0], coords[:, 1], marker="o", linestyle="-", label="Geometry")


def plot_geometry(geometry_cfg: GeometryConfig) -> None:

    plt.figure()
    plt.title("Airfoil Geometry")

    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis("equal")

    add_geometry(geometry_cfg)

    plt.show()


def plot_chosen_te_panels(
    geometry_cfg: GeometryConfig, panels: PanelInfo, result: SolverResult
) -> None:
    plt.figure()
    plt.title("Trailing Edge Chosen")

    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis("equal")

    add_geometry(geometry_cfg)
    plt.plot(
        panels.ctrl_point_coords[result.te_panel_indices[0]][0],
        panels.ctrl_point_coords[result.te_panel_indices[0]][1],
        marker="x",
        color="red",
        label="First TE Ctrl Point",
    )
    plt.plot(
        panels.ctrl_point_coords[result.te_panel_indices[1]][0],
        panels.ctrl_point_coords[result.te_panel_indices[1]][1],
        marker="x",
        color="green",
        label="Second TE Ctrl Point",
    )
    plt.legend()

    plt.show()


def plot_cp_distribution(
    geometry_cfg: GeometryConfig, panels: PanelInfo, result: SolverResult
) -> None:
    plt.figure()
    plt.title("$$C_p$$ distribution")

    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis("equal")

    add_geometry(geometry_cfg)

    # panel normal
    nx = -panels.sin_angles
    ny = panels.cos_angles

    # add the cp mag
    u = result.C_p_p * nx
    v = result.C_p_p * ny

    plt.quiver(
        panels.ctrl_point_coords[:, 0],
        panels.ctrl_point_coords[:, 1],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1,
        label="$$C_p$$",
    )
    plt.legend()
    plt.show()
