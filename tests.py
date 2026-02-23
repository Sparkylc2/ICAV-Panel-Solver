import matplotlib.pyplot as plt
import numpy as np

from panel_gen import cylinder, naca4
from solver import run_solver
from structs import (FlowConfig, GeometryConfig, InfluenceData, PanelInfo,
                     SolverResult, SolverState)

ENABLE_TEST_PLOTTING = False


def test_source_cylinder_tangential() -> None:
    print("=" * 70)
    print("testing the source pass with a cylinder against analytical")
    coords = cylinder(100)
    state = run_solver(GeometryConfig(coords=coords), flow_cfg=FlowConfig())

    # from analytical results, the Q about the cylinder (on its surface) should be
    # Q = 2*U_inf * sin(theta)
    # its actually -2*... but we are traversing clockwise not counterclockwise

    ctrl = state.get_panels().ctrl_point_coords
    theta_polar = np.arctan2(ctrl[:, 1], ctrl[:, 0] - 0.5)
    Q_analytical = 2 * state.get_flow_cfg().u_inf * np.sin(theta_polar)
    Q_calculated = state.get_source_influence().Q
    residual = Q_calculated - Q_analytical

    Q_comparison = np.column_stack([Q_analytical, Q_calculated, residual])

    print("Q_analytical, Q_calculated, residual (difference)")
    print(Q_comparison)
    print("diagonal terms of A_pq (should all be 0.5)")
    print(np.diag(state.get_source_influence().A_pq))
    print(
        "m_q range:",
        state.get_source_influence().m.min(),
        state.get_source_influence().m.max(),
    )
    print(
        "B_contribution range:",
        state.get_source_influence().B_contribution.min(),
        state.get_source_influence().B_contribution.max(),
    )
    print(
        "B@m range:",
        (state.get_source_influence().B_pq @ state.get_source_influence().m).min(),
        (state.get_source_influence().B_pq @ state.get_source_influence().m).max(),
    )
    if ENABLE_TEST_PLOTTING:
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

        plt.figure()
        plt.title("self-influence sign check")

        plt.plot(
            state.get_geometry_cfg().coords[:, 0],
            state.get_geometry_cfg().coords[:, 1],
            "-",
            color="black",
            alpha=0.3,
            label="Cylinder Outline",
        )

        diag = np.diag(state.get_source_influence().A_pq)
        ctrl = state.get_panels().ctrl_point_coords

        mask_pos = np.where(diag > 0)[0]
        mask_neg = np.where(diag < 0)[0]
        plt.scatter(
            ctrl[mask_pos, 0],
            ctrl[mask_pos, 1],
            marker="o",
            color="green",
            label=f"Positive (0.5): {len(mask_pos)} panels",
        )
        plt.scatter(
            ctrl[mask_neg, 0],
            ctrl[mask_neg, 1],
            marker="x",
            color="red",
            s=100,
            label=f"Negative (-0.5): {len(mask_neg)} panels",
        )

        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    print("=" * 70)


def test_lifting_cylinder():
    print("=" * 70)
    print("testing the full solver (source + vortex) on a cylinder")
    coords = cylinder(160)
    flow_cfg = FlowConfig(u_inf=10.0, aoa=np.radians(5.0))
    state = run_solver(GeometryConfig(coords=coords), flow_cfg=flow_cfg)

    result = state.get_result()
    ctrl = state.get_panels().ctrl_point_coords
    theta_polar = np.arctan2(ctrl[:, 1], ctrl[:, 0] - 0.5)

    # gamma_total = K * chord * u_inf
    # analytical Q for a lifting cylinder:
    Q_analytical = 2 * flow_cfg.u_inf * np.sin(theta_polar - flow_cfg.aoa) + (
        result.K * flow_cfg.u_inf * state.get_geometry_cfg().chord
    )

    # the circulation strength (Gamma) is K * U_inf * chord
    # the induced tangential velocity on a cylinder is Gamma / (pi * D)
    # since D = chord
    v_circulation = (result.K * state.get_flow_cfg().u_inf) / (
        np.pi * state.get_geometry_cfg().chord
    )

    Q_analytical = (
        2 * state.get_flow_cfg().u_inf * np.sin(theta_polar - flow_cfg.aoa)
        + v_circulation
    )

    print("Q_analytical, Q_calculated", "residual ")
    print(
        np.column_stack(
            [
                Q_analytical,
                result.Q,
                result.Q - Q_analytical,
            ]
        )
    )

    print(f"calculated lift coefficient (C_L): {result.C_L:.4f}")
    if ENABLE_TEST_PLOTTING:
        plt.figure(figsize=(10, 5))
        plt.title("Tangential Velocity: Lifting Cylinder")
        plt.plot(theta_polar, result.Q, "ro", label="calculated Q (superimposed)")
        plt.plot(theta_polar, Q_analytical, "k-", label="analytical Q")
        plt.legend()
        plt.show()
    print("=" * 70)


def test_kutta() -> None:
    flow_cfg = FlowConfig(10)
    geometry_cfg = GeometryConfig(naca4("2412", 150), 1)
    state = run_solver(geometry_cfg, flow_cfg)
    source_te_sum = (
        state.get_source_influence().Q[state.get_result().te_panel_indices[0]]
        + state.get_source_influence().Q[state.get_result().te_panel_indices[1]]
    )
    vortex_te_sum = state.get_result().K * (
        state.get_vortex_influence().Q[state.get_result().te_panel_indices[0]]
        + state.get_vortex_influence().Q[state.get_result().te_panel_indices[1]]
    )
    print(f"Source TE sum: {source_te_sum}")
    print(f"Vortex TE sum: {vortex_te_sum}")
    print(f"Total TE sum: {source_te_sum + vortex_te_sum}")

    B_source = state.get_source_influence().B_pq
    print(f"Max B value: {np.max(np.abs(B_source))}")
    print(f"Min B value: {np.min(np.abs(B_source))}")
