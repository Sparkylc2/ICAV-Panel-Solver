from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FlowConfig:
    u_inf: float = 10.0  # free-stream velocity (m/s)
    aoa: float = 0.0  # angle of attack (radians)
    density: float = 1.225  # air density (kg/m^3)


@dataclass(frozen=True)
class GeometryConfig:
    coords: np.ndarray  # (x, y) coordinates as an (Nx2) matrix
    chord: float = 1.0  # chord (m)


@dataclass(frozen=True)
class SolverConfig:
    ENABLE_DEBUG_PLOTTING: bool = True  # determines if we enable plots in the tests
    USE_SINGLE_VORTEX_METHOD: bool = True  # determines which vortex function we use


@dataclass(frozen=True)
class PanelInfo:
    n_panels: int  # number of panels in our geometry
    l_panels: np.ndarray  # lengths of the panels
    ctrl_point_coords: np.ndarray  # control point coordinates for each panel (x, y)
    panel_angles: np.ndarray  # the angles of each panel (rad)
    # these two are useful to reuse
    sin_angles: np.ndarray  # the sin of the panel angles
    cos_angles: np.ndarray  # the cos of the panel angles


@dataclass
class InfluenceData:
    A_pq: np.ndarray  # normal influence coefficient matrix (NxN)
    B_pq: np.ndarray  # tangential influence coefficient matrix (NxN)

    A_rhs: np.ndarray  # rhs of the no-penetration equation A*m = A_rhs (Nx1)
    B_contribution: np.ndarray  # the velocity contribution to B (N)

    m: np.ndarray  # source strengths (Nx1)
    Q: np.ndarray  # tangential surface velocity (Nx1)


@dataclass
class SolverResult:
    te_coordinate_idx: int  # the trailing edge coordinate index
    te_panel_indices: np.ndarray  # the indices of the upper and lower TE panels

    K: float  # the scaling factor from superimposing the solutions
    Q: np.ndarray  # tangential velocity at each panel (Nx1)

    L: float  # the total lift (in Newtons)
    C_L: float  # the lift coefficient
    C_p_p: np.ndarray  # the pressure coefficient at each panel

    def print_result(self) -> None:
        print("=" * 70)
        print(f"Lift: {self.L}")
        print(f"C_L: {self.C_L}")
        print(f"C_p: {self.C_p_p}")
        print("=" * 70)


@dataclass
class SolverState:
    geometry_cfg: GeometryConfig  # the geometry of the airfoil
    flow_cfg: FlowConfig  # the flow configuration
    solver_cfg: SolverConfig = SolverConfig(
        ENABLE_DEBUG_PLOTTING=True, USE_SINGLE_VORTEX_METHOD=True
    )  # the solver configuration
    panels: PanelInfo | None = None  # the panel geometry information
    source_influence: InfluenceData | None = None  # the influence data for source pass
    vortex_influence: InfluenceData | None = None  # the influence data for vortex pass
    result: SolverResult | None = None  # the final result from the solver

    def get_geometry_cfg(self) -> GeometryConfig:
        assert self.geometry_cfg is not None
        return self.geometry_cfg

    def get_flow_cfg(self) -> FlowConfig:
        assert self.flow_cfg is not None
        return self.flow_cfg

    def get_panels(self) -> PanelInfo:
        assert self.panels is not None
        return self.panels

    def get_source_influence(self) -> InfluenceData:
        assert self.source_influence is not None
        return self.source_influence

    def get_vortex_influence(self) -> InfluenceData:
        assert self.vortex_influence is not None
        return self.vortex_influence

    def get_result(self) -> SolverResult:
        assert self.result is not None
        return self.result
