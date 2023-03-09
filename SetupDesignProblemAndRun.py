"""The main file for problem setup and execution.

User parameters are specified for the macroscale problem, the representative
volume element problem, and for the optimizer. Subsequently, the optimization
problem is executed in the "main" function at the end of the file.
"""
import os
import logging
import numpy as np

import src.material_topopt.utilities as utils
from src.material_topopt.optimizer import MaterialOptimizer

logging_level = logging.INFO
utils.setup_logging(logging_level, logfile_path='') # Specify a logfile_path for console output to go into a file


#######################################################################################################################
#######################################################################################################################
def simp_exponent_continuation_function(iteration_number: int) -> float:
    """Computes the SIMP exponent.

    Using the optimization iteration number, the user should return a float
    corresponding to the SIMP exponent used in the elastic modulus interpolation
    within the representative volume element. The returned value must be greater
    than or equal to 1.0 and less than 5.

    Args:
        iteration_number: An integer containing the current optimization iteration number.

    Returns:
        A positive float for the SIMP exponent satisfying, 1 <= p < 5
    """
    if iteration_number < 50:
        return 3.0
    elif iteration_number < 75:
        return 3.5
    return 4.0


#######################################################################################################################
#######################################################################################################################
def smooth_heaviside_projection_continuation_function(iteration_number: int) -> float:
    """Computes the smooth Heaviside projection parameter.

    Using the optimization iteration number, the user should return a float
    corresponding to the smooth Heaviside projection parameter, \\beta in the function below.

    .. math::
        \\rho(\\hat{\\rho}) = \\frac{\\tanh{\\beta \\eta} + \\tanh{\\beta \\left( \\hat{\\rho} - \\eta \\right)}}
            {\\tanh{\\beta \\eta} + \\tanh{\\beta \\left( 1 - \\eta \\right)}}

    Args:
        iteration_number: An integer containing the current optimization iteration number.

    Returns:
        A smooth Heaviside projection parameter, beta, that is a positive float.
    """
    if iteration_number < 100:
        return 1.0
    elif iteration_number < 125:
        return 2.0
    elif iteration_number < 150:
        return 3.0
    elif iteration_number < 175:
        return 4.0
    return 5.0


#######################################################################################################################
#######################################################################################################################
def design_variable_initialization_function(nodal_x_coordinates: np.ndarray,
                                            nodal_y_coordinates: np.ndarray) -> np.ndarray:
    """Computes the initial design variables for the representative volume element.

    Given the X and Y coordinates of the nodes, this function should return an initial value
    for the design variable at each node using any method they wish. The design variables
    must all be between 0 and 1. Note that the X and Y nodal coordinates always range from 0 to 1.

    Args:
        nodal_x_coordinates: A numpy array containing the x coordinates of the RVE nodes.
        nodal_y_coordinates: A numpy array containing the y coordinates of the RVE nodes.

    Returns:
        A numpy array containing the initial design variables for each RVE node.
    """
    candidate_initial_design_variables = (0.5 + 0.5 * np.cos(2.0*np.pi * nodal_x_coordinates)) * \
                                         (0.5 + 0.5 * np.cos(2.0*np.pi * nodal_y_coordinates)) + \
                                         (0.5 + 0.5 * np.cos(2.0*np.pi * (nodal_x_coordinates - 0.5))) * \
                                         (0.5 + 0.5 * np.cos(2.0*np.pi * (nodal_y_coordinates - 0.5)))
    # Ensure design variables are between 0 and 1
    candidate_design_variable_min = np.amin(candidate_initial_design_variables)
    candidate_design_variable_extent = np.amax(candidate_initial_design_variables) - candidate_design_variable_min
    initial_design_variables = \
        (candidate_initial_design_variables - candidate_design_variable_min) / candidate_design_variable_extent
    return initial_design_variables


#######################################################################################################################
#######################################################################################################################
# User Defined Parameters

# The path to the directory in which the output files (e.g., VTK files, CSV files, figures, etc.) will be written.
output_directory_path = os.path.join(os.getcwd(), "output")

# The number of elements along each edge of the square RVE (must be a positive integer)
number_of_elements_along_each_edge = 100

# The radius of the design variable filter in units of the number of elements (must be a positive number > 1)
filter_radius_number_of_elements = 6

representative_volume_element_parameters = \
    {
        "number of elements along each edge": number_of_elements_along_each_edge,
        "stiff material elastic modulus": 1.0e3,
        "soft material elastic modulus": 1.0,
        "poissons ratio": 0.3,
        "SIMP exponent continuation function": simp_exponent_continuation_function,
        "smooth Heaviside projection continuation function": smooth_heaviside_projection_continuation_function,
        "design variable initialization function": design_variable_initialization_function,
        "density filter radius": float(filter_radius_number_of_elements) / float(number_of_elements_along_each_edge),
        "output directory path": output_directory_path,
        "enable vtk output": True,
        "enable matplotlib output": True
    }

# Paperclip
fixed_boundary_nodesets = {"Fixed X Displacement Nodesets": ["fixed_nodeset"],
                           "Fixed Y Displacement Nodesets": ["fixed_nodeset"]}
applied_load_1 = {"Nodeset": "load_nodeset", "Load in X direction": 0.2, "Load in Y direction": 0.0}
applied_loads = [applied_load_1]
macroscale_problem_parameters = \
    {
        "macroscale finite element mesh filepath": os.path.join(os.getcwd(), "mesh_files", "paperclip.inp"),
        "fixed boundary condition nodesets": fixed_boundary_nodesets,
        "applied loads": applied_loads,
        "output directory path": output_directory_path
    }

# Letter M
# fixed_boundary_nodesets = {"Fixed X Displacement Nodesets": ["NS1"], "Fixed Y Displacement Nodesets": ["NS1"]}
# applied_load_1 = {"Nodeset": "NS2", "Load in X direction": 0.0, "Load in Y direction": 0.4}
# applied_load_2 = {"Nodeset": "NS3", "Load in X direction": 0.1, "Load in Y direction": 0.0}
# applied_loads = [applied_load_1, applied_load_2]
# macroscale_problem_parameters = \
#     {
#         "macroscale finite element mesh filepath": os.path.join(os.getcwd(), "mesh_files", "letter_M_complete.exo"),
#         "fixed boundary condition nodesets": fixed_boundary_nodesets,
#         "applied loads": applied_loads,
#         "output directory path": output_directory_path
#     }

optimization_problem_parameters = \
    {
        "maximum number of iterations": 250,
        "volume fraction constraint upper bound": 0.5,
        "restart iteration number": 0,
        "restart file write frequency": 500,
        "MMA move limit": 0.1,
        "optimization history output filepath": os.path.join(output_directory_path, "MaterialTopOptData.csv")
    }


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    material_optimizer = \
        MaterialOptimizer(optimization_problem_parameters = optimization_problem_parameters,
                          macroscale_problem_parameters = macroscale_problem_parameters,
                          representative_volume_element_parameters = representative_volume_element_parameters)
    material_optimizer.macroscale_problem.plot_mesh()
    material_optimizer.macroscale_problem.representative_volume_element.plot_mesh()
    material_optimizer.run()
    material_optimizer.plot_history()
    plt.show()
