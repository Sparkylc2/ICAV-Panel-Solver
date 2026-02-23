import matplotlib.pyplot as plt
import numpy as np

from panel_gen import cylinder
from solver import run_solver
from structs import (FlowConfig, GeometryConfig, InfluenceData, PanelInfo,
                     SolverResult, SolverState)


def test_source_cylinder_tangential() -> None:
    print("=" * 70)
    print("testing the source pass with a cylinder against analytical")
    coords = cylinder(100)
    state = run_solver(GeometryConfig(coords=coords), flow_cfg=FlowConfig())

    # from analytical results, the Q about the cylinder (on its surface) should be
    # Q = - 2 U_inf * sin(theta)
    print(state.get_panels().panel_angles)
    ctrl = state.get_panels().ctrl_point_coords
    theta_polar = np.arctan2(ctrl[:, 1], ctrl[:, 0] - 0.5)
    Q_analytical = -2 * state.get_flow_cfg().u_inf * np.sin(theta_polar)
    Q_calculated = state.get_source_influence().Q

    Q_comparison = np.column_stack([Q_analytical, Q_calculated])
    print("Q_analytical, Q_calculated")
    print(Q_comparison)
    print("diagonal terms of A_pq (should all be 0.5)")
    print(np.diag(state.get_source_influence().A_pq))
    print("plotting")
    residual = Q_calculated - Q_analytical
    plt.figure()
    plt.title("Cylinder Test Results")
    plt.plot(
        range(state.get_panels().n_panels),
        residual,
        "o",
        color="red",
        label="Residual (calculated - analytical)",
    )
    plt.xlabel("Index")
    plt.ylabel("$Q_{calculated} - Q_{analytical}$")
    plt.legend()
    plt.show()
    print("=" * 70)
