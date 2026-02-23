import matplotlib.pyplot as plt
import numpy as np

from panel_gen import cylinder, naca4
from plotting import plot_chosen_te_panels, plot_cp_distribution, plot_geometry
from solver import run_solver
from structs import FlowConfig, GeometryConfig, PanelInfo, SolverState
from tests import test_source_cylinder_tangential

U_inf = 10  # m/s
AoA = np.radians(5)  # rad
Chord = 0.15  # m
Density = 1.225  # kg/m^3

Coords = cylinder(100)

# initialize our initial flow configuration
flow_cfg = FlowConfig(u_inf=U_inf, aoa=AoA, density=Density)
geometry_cfg = GeometryConfig(coords=Coords, chord=Chord)

# run the solver based on the geometry and flow configuration
state = run_solver(geometry_cfg, flow_cfg)
state.get_result().print_result()


test_source_cylinder_tangential()

# plotting
# plot_geometry(geometry_cfg=state.get_geometry_cfg())
# plot_chosen_te_panels(
#     geometry_cfg=geometry_cfg, panels=state.get_panels(), result=state.get_result()
# )
