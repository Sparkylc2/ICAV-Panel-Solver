import matplotlib.pyplot as plt
import numpy as np

from panel_gen import cylinder, naca4
from plotting import plot_chosen_te_panels, plot_cp_distribution, plot_geometry
from solver import run_solver
from structs import (FlowConfig, GeometryConfig, PanelInfo, SolverConfig,
                     SolverState)
from tests import (test_kutta, test_lifting_cylinder,
                   test_source_cylinder_tangential)

U_inf = 10  # m/s
AoA = np.radians(0)  # rad
Chord = 0.15  # m
Density = 1.225  # kg/m^3


# initialize our initial flow configuration
flow_cfg = FlowConfig(u_inf=U_inf, aoa=AoA, density=Density)
geometry_cfg = GeometryConfig(coords=naca4("2412", 200), chord=Chord)
solver_cfg = SolverConfig(ENABLE_DEBUG_PLOTTING=False, USE_SINGLE_VORTEX_METHOD=False)

# run the solver based on the geometry and flow configuration
state = run_solver(geometry_cfg, flow_cfg, solver_cfg)
state.get_result().print_result()


# test_kutta()
# test_source_cylinder_tangential()
# test_lifting_cylinder()

# plotting
plot_geometry(geometry_cfg=state.get_geometry_cfg())
plot_chosen_te_panels(
    geometry_cfg=geometry_cfg, panels=state.get_panels(), result=state.get_result()
)
plot_cp_distribution(
    geometry_cfg=geometry_cfg, panels=state.get_panels(), result=state.get_result()
)
