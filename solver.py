import matplotlib.pyplot as plt
import numpy as np

from structs import (FlowConfig, GeometryConfig, InfluenceData, PanelInfo,
                     SolverConfig, SolverResult, SolverState)


def passive_rotation(theta: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


def get_panel_info(geometry_cfg: GeometryConfig) -> PanelInfo:
    """
    returns all the info we need about the panel for calculations
    """

    coords = geometry_cfg.coords

    # [x_p - x_p-1, y_p - y_p-1]
    d_coords = np.diff(coords, axis=0)
    # [x_p + x_p+1, y_p + y_p+1]
    a_coords = coords[:-1] + coords[1:]

    # number of panels (-1 as we are joining points with edges)
    n_panels = coords.shape[0] - 1

    # panel lengths
    l_panels = np.sqrt(d_coords[:, 0] ** 2 + d_coords[:, 1] ** 2)
    # control point coordinates
    ctrl_point_coords = a_coords / 2
    # panel angles (counterclockwise from global x-axis to inside of panel)
    panel_angles = np.arctan2(d_coords[:, 1], d_coords[:, 0])

    # compute these already since we use them a bit
    sin_angles = np.sin(panel_angles)
    cos_angles = np.cos(panel_angles)

    return PanelInfo(
        n_panels=n_panels,
        l_panels=l_panels,
        ctrl_point_coords=ctrl_point_coords,
        panel_angles=panel_angles,
        sin_angles=sin_angles,
        cos_angles=cos_angles,
    )


def get_source_influence_coefficients(
    panels: PanelInfo, flow_cfg: FlowConfig
) -> InfluenceData:
    """
    for every p'th panel we calculate the normal and tangential velocity
    contributions from every other q'th panel

    the source is in panel q's frame of reference (FOR), so we first convert
    the displacement from global FOR to panel q's FOR, compute the induced
    velocity, and then rotate into panel p's FOR
    """
    N = panels.n_panels

    # freestream contributions (in each panel p's FOR)
    A_rhs = -flow_cfg.u_inf * np.sin(flow_cfg.aoa - panels.panel_angles)
    B_contribution = flow_cfg.u_inf * np.cos(flow_cfg.aoa - panels.panel_angles)

    # displacement from q's control point to p's control point (global FOR)
    # d_global[p, q, :] = [x_p - x_q, y_p - y_q]
    d_global = (
        panels.ctrl_point_coords[:, np.newaxis, :]
        - panels.ctrl_point_coords[np.newaxis, :, :]
    )  # (N, N, 2)

    # rotation matrices for each panel (global FOR -> panel FOR)
    R = np.moveaxis(passive_rotation(panels.panel_angles), -1, 0)  # (N, 2, 2)

    # transform displacement into panel q's FOR
    # d_local[p, q, :] = R_q @ d_global[p, q, :]
    d_local = np.einsum("qij,pqj->pqi", R, d_global)  # (N, N, 2)
    x_pq = d_local[:, :, 0]
    y_pq = d_local[:, :, 1]

    # induced velocity components in q's FOR (per unit source strength)
    half_l = 0.5 * panels.l_panels[np.newaxis, :]  # (1, N) indexed by q
    v_xq = (1 / (4 * np.pi)) * np.log(
        ((x_pq + half_l) ** 2 + y_pq**2) / ((x_pq - half_l) ** 2 + y_pq**2)
    )
    v_yq = (1 / (2 * np.pi)) * np.arctan2(
        y_pq * panels.l_panels[np.newaxis, :],
        x_pq**2 + y_pq**2 - half_l**2,
    )

    # transforming velocity from q's FOR to p's FOR
    # R_pq = R_p @ R_q^T (composed rotation from q's FOR to p's FOR)
    v_q = np.stack([v_xq, v_yq], axis=-1)  # (N, N, 2)
    R_pq = np.einsum("pij,qkj->pqik", R, R)  # (N, N, 2, 2)
    v_p = np.einsum("pqij,pqj->pqi", R_pq, v_q)  # (N, N, 2)

    # influence matrices (in p's FOR)
    B_pq = v_p[:, :, 0]  # tangential
    A_pq = v_p[:, :, 1]  # normal

    # solve for source strengths m_q (enforcing no-penetration)
    m_q = np.linalg.solve(A_pq, A_rhs)

    # tangential velocity at each panel
    Q_p = B_pq @ m_q + B_contribution

    return InfluenceData(
        A_pq=A_pq, B_pq=B_pq, A_rhs=A_rhs, B_contribution=B_contribution, m=m_q, Q=Q_p
    )


def get_vortex_influence_coefficients(
    panels: PanelInfo,
    source_influence: InfluenceData,
    flow_cfg: FlowConfig,
    geometry_cfg: GeometryConfig,
) -> InfluenceData:
    """
    we redo what we did before except now considering uniformly distributed point
    vortices along each panel. we need the influence data from the previous source step
    so we also pass it in

    we use the results of this to superimpose the two solutions together,
    one which now obeys the kutta-condition (ie we do this second pass to later
    enforce the kutta-condition)

    we define gamma/(u_inf * chord) = 1

    we can use the relationship in the notes:
        A_p,q_gamma = B_p,q (sources)
        B_p,q_gamma = -A_p,q (sources)

    to compute the extra contribution terms to enforce no-pen and to get
    """

    # we need our gamma, as we have it uniformly distributed over the entire airfoil
    gamma = flow_cfg.u_inf * geometry_cfg.chord / np.sum(panels.l_panels)

    # get our influence matrices (relation from the paper)
    A_pq_gamma = -source_influence.B_pq
    B_pq_gamma = source_influence.A_pq

    # the total velocity contributions (in p's FOR) due to the other q panels
    U_np_gamma = gamma * A_pq_gamma.sum(axis=1)  # normal velocity
    U_tp_gamma = gamma * B_pq_gamma.sum(axis=1)  # tangential velocity

    A_rhs = -U_np_gamma
    B_contribution = U_tp_gamma

    # solving to get the second set of source strengths (again just enforcing no-pen)
    m_q_gamma = np.linalg.solve(source_influence.A_pq, A_rhs)

    # with the strengths we compute the tangential velocity again by adding
    # the contributions from the other panels (in p's FOR)
    Q_p_gamma = source_influence.B_pq @ m_q_gamma + B_contribution

    # return the data
    return InfluenceData(
        A_pq=A_pq_gamma,
        B_pq=B_pq_gamma,
        A_rhs=A_rhs,
        B_contribution=B_contribution,
        m=m_q_gamma,
        Q=Q_p_gamma,
    )


def get_vortex_influence_coefficients_single_vortex(
    panels: PanelInfo,
    source_influence: InfluenceData,
    flow_cfg: FlowConfig,
    geometry_cfg: GeometryConfig,
) -> InfluenceData:

    # vortex coordinates (in global FOR, slightly arbitrary)
    x_v = geometry_cfg.chord / 4
    y_v = 0.0

    # we still normalize as having gamma/(u_inf * c) = 1
    gamma = flow_cfg.u_inf * geometry_cfg.chord

    # the control point coordinates (in global FOR)
    x_p = panels.ctrl_point_coords[:, 0]
    y_p = panels.ctrl_point_coords[:, 1]

    # the two numerators for the induced velocity by the vortex
    dx = x_p - x_v
    dy = y_p - y_v

    # common term
    k_inv_r2 = gamma / (2 * np.pi * (dx**2 + dy**2))

    # the components of induced velocity (in global FOR)
    u_x_gamma = -k_inv_r2 * dy
    u_y_gamma = k_inv_r2 * dx

    # set up the rotation matrix
    R = np.moveaxis(passive_rotation(panels.panel_angles), -1, 0)  # (N, 2, 2)
    # our rotated u (panel p's FOR)
    u_p_gamma = np.einsum("nij,nj->ni", R, np.column_stack([u_x_gamma, u_y_gamma]))

    A_rhs = -u_p_gamma[:, 1]  # induced normal velocity (in panel p's FOR)
    B_contribution = u_p_gamma[:, 0]  # induced tangential velocity (in panel p's FOR)

    # solve for m_q_gamma using the normal velocity and satisfying the no-penetration condition
    m_q_gamma = np.linalg.solve(source_influence.A_pq, A_rhs)

    # solving for our induced tangential velocity
    Q_p_gamma = source_influence.B_pq @ m_q_gamma + B_contribution

    return InfluenceData(
        A_pq=source_influence.A_pq,
        B_pq=source_influence.B_pq,
        A_rhs=A_rhs,
        B_contribution=B_contribution,
        m=m_q_gamma,
        Q=Q_p_gamma,
    )


def superimpose_solutions(
    source_influence: InfluenceData,
    vortex_influence: InfluenceData,
    flow_cfg: FlowConfig,
    geometry_cfg: GeometryConfig,
) -> SolverResult:
    """
    we can superimpose our solutions and use Q = Q_p + K * Q_p_gamma

    enforcing the kutta condition at the trailing edge means that:
    dQ = dQ_te + K * dQ_te_gamma = 0 (dQ_te = Q_T+1 - Q_T)
    with T+1 and T being the the lower and upper panels respectively,
    closest to the trailing edge
    """

    # get the trailing edge location
    te_coordinate_idx = np.argmax(geometry_cfg.coords[:, 0])
    te_panel_indices = [te_coordinate_idx - 1, te_coordinate_idx]

    # find dQ_te for both solutions
    dQ_te = source_influence.Q[te_panel_indices].sum()
    dQ_te_gamma = vortex_influence.Q[te_panel_indices].sum()

    # enforcing the kutta-condition
    K = -dQ_te / dQ_te_gamma

    # calculating the remaining information for the solution
    # superimposing the solutions to get final tangential velocity
    Q = source_influence.Q + K * vortex_influence.Q

    L = flow_cfg.density * flow_cfg.u_inf**2 * K * geometry_cfg.chord  # total lift
    C_L = 2 * K  # lift coefficient
    C_p_p = 1 - (Q / flow_cfg.u_inf) ** 2  # vector of C_p at each panel

    # return the results
    return SolverResult(
        te_coordinate_idx=te_coordinate_idx,
        te_panel_indices=te_panel_indices,
        K=K,
        Q=Q,
        L=L,
        C_L=C_L,
        C_p_p=C_p_p,
    )


def run_solver(
    geometry_cfg: GeometryConfig,
    flow_cfg: FlowConfig,
    solver_cfg: SolverConfig = SolverConfig(),
) -> SolverState:
    """
    self contained function to run the solver given a flow config,
    geometry config, and solver config
    """
    state = SolverState(
        geometry_cfg=geometry_cfg, flow_cfg=flow_cfg, solver_cfg=solver_cfg
    )
    # get our panel data
    state.panels = get_panel_info(state.geometry_cfg)

    # compute the source influence data
    state.source_influence = get_source_influence_coefficients(
        panels=state.panels, flow_cfg=state.flow_cfg
    )

    # compute the vortex influence data
    if solver_cfg.USE_SINGLE_VORTEX_METHOD:
        state.vortex_influence = get_vortex_influence_coefficients_single_vortex(
            panels=state.panels,
            source_influence=state.source_influence,
            flow_cfg=state.flow_cfg,
            geometry_cfg=state.geometry_cfg,
        )
    else:
        state.vortex_influence = get_vortex_influence_coefficients(
            panels=state.panels,
            source_influence=state.source_influence,
            flow_cfg=state.flow_cfg,
            geometry_cfg=state.geometry_cfg,
        )

    # compute the result
    state.result = superimpose_solutions(
        source_influence=state.source_influence,
        vortex_influence=state.vortex_influence,
        flow_cfg=state.flow_cfg,
        geometry_cfg=state.geometry_cfg,
    )
    return state
