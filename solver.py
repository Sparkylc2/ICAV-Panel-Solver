import matplotlib.pyplot as plt
import numpy as np

from structs import (FlowConfig, GeometryConfig, InfluenceData, PanelInfo,
                     SolverResult, SolverState)


def get_panel_info(geometry_cfg: GeometryConfig) -> PanelInfo:
    """
    returns all the info we need about the panel for calculations
    """

    coords = geometry_cfg.coords

    # [x_p - x_p-1, y_p - y_p-1]
    d_coords = np.diff(coords, axis=0)
    # [x_p + x_p+1, y_p + y_p-1]
    a_coords = coords[:-1] + coords[1:]

    # number of panels (-1 as we are joining points with edges)
    n_panels = len(coords[:, 0]) - 1
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
    for every p'th panel we loop around every other q'th panel and calculate
    the normal and tangential contributions of the q'th panel on the p'th one

    the source is in panel q's frame of reference (FOR), so we first need to convert
    the displacement between the panels from the global FOR to panel q's FOR
    and then once we calculate the induced velocity we convert that to panel p's
    FOR
    """

    # we can quickly compute the two freestream contributions
    A_rhs = -flow_cfg.u_inf * np.sin(flow_cfg.aoa - panels.panel_angles)
    B_contribution = flow_cfg.u_inf * np.cos(flow_cfg.aoa - panels.panel_angles)

    # initializing the influence matrices
    A_pq = np.zeros((panels.n_panels, panels.n_panels))
    B_pq = np.zeros((panels.n_panels, panels.n_panels))

    for p in range(panels.n_panels):
        for q in range(panels.n_panels):

            # self influence is always -0.5, 0.0
            # if p == q:
            #     A_pq[p][q] = 0.5
            #     B_pq[p][q] = 0.0
            #     continue

            # 99% sure its this way around
            # x_p - x_q
            d_x = panels.ctrl_point_coords[p][0] - panels.ctrl_point_coords[q][0]
            # y_p - y_q
            d_y = panels.ctrl_point_coords[p][1] - panels.ctrl_point_coords[q][1]

            # transform the relative displacement into panel q's FOR
            if p == q:
                x_pq = 0.0
                y_pq = 0.0
            else:
                x_pq = d_x * panels.cos_angles[q] + d_y * panels.sin_angles[q]
                y_pq = d_y * panels.cos_angles[q] - d_x * panels.sin_angles[q]

            # compute the normal and tangential velocity contributions (in panel q's FOR)
            len_panel_q = panels.l_panels[q]

            v_xq_num = (x_pq + 0.5 * len_panel_q) ** 2 + y_pq**2
            v_xq_denom = (x_pq - 0.5 * len_panel_q) ** 2 + y_pq**2
            # the tangential velocity component (in q's FOR)
            v_xq = 1 / (4 * np.pi) * np.log(v_xq_num / v_xq_denom)

            v_yq_num = y_pq * len_panel_q
            v_yq_denom = x_pq**2 + y_pq**2 - (0.5 * len_panel_q) ** 2
            # the normal velocity component (in q's FOR)
            v_yq = 1 / (2 * np.pi) * np.arctan2(v_yq_num, v_yq_denom)

            # now we convert to p's FOR to get the normal and tangential velocity contributions
            # theta_q - theta_p
            d_theta = panels.panel_angles[q] - panels.panel_angles[p]
            sin_d_theta = np.sin(d_theta)
            cos_d_theta = np.cos(d_theta)

            # the velocity contribution (in p's FOR)
            # v_n = v_yq * cos_d_theta - v_xq * sin_d_theta  # normal
            # v_t = v_xq * cos_d_theta + v_yq * sin_d_theta  # tangential
            v_n = v_yq * cos_d_theta + v_xq * sin_d_theta  # normal
            v_t = v_xq * cos_d_theta - v_yq * sin_d_theta  # tangential

            # placing in our influence matrices
            A_pq[p][q] = v_n
            B_pq[p][q] = v_t

    # now we can solve for the source strengths m_q (this just enforces no-pen)
    m_q = np.linalg.solve(A_pq, A_rhs)

    # with our strengths we can now solve for the tangential velocity by adding
    # the freestream contribution (in p's FOR)
    Q_p = B_pq @ m_q + B_contribution

    # returning our influence data
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
    A_pq_gamma = source_influence.B_pq
    B_pq_gamma = -source_influence.A_pq

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
    print("Q_te source", source_influence.Q[te_panel_indices])
    print("Q_te vortex", vortex_influence.Q[te_panel_indices])

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


def run_solver(geometry_cfg: GeometryConfig, flow_cfg: FlowConfig) -> SolverState:
    """
    self contained function to run the solver given a flow config and
    geometry config
    """
    state = SolverState(geometry_cfg=geometry_cfg, flow_cfg=flow_cfg)
    # get our panel data
    state.panels = get_panel_info(state.geometry_cfg)

    # compute the source influence data
    state.source_influence = get_source_influence_coefficients(
        panels=state.panels, flow_cfg=state.flow_cfg
    )

    # compute the vortex influence data
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
