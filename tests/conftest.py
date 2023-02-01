"""pytest fixtures for material topopt unit tests.

To execute the tests, from the parent directory run the following command:
python -m pytest -v -s -k test
"""
import pytest
import os
import meshio
import numpy as np
from scipy.spatial import Delaunay
import logging
import src.material_topopt.utilities as utils


#######################################################################################################################
#######################################################################################################################
@pytest.fixture(scope="function")
def temporary_output_directory(tmp_path):
    print(f" ( test running with output in path {tmp_path} )")
    yield tmp_path


#######################################################################################################################
#######################################################################################################################
@pytest.fixture(scope="function")
def macroscale_mesh_filepath(temporary_output_directory):
    yield os.path.join(temporary_output_directory, "temporary_pytest_mesh.exo")


#######################################################################################################################
#######################################################################################################################
@pytest.fixture(scope="function")
def create_macroscale_mesh(macroscale_mesh_filepath):
    mesh_filepath = macroscale_mesh_filepath
    def _create_meshio_mesh(domain_width: float = 1.0, domain_height: float = 1.0,
                            number_of_elements_along_width: int = 10, number_of_elements_along_height: int = 10):
        logging_level = logging.CRITICAL
        utils.setup_logging(logging_level)
        x = np.linspace(0.0, domain_width,  number_of_elements_along_width  + 1)
        y = np.linspace(0.0, domain_height, number_of_elements_along_height + 1)
        smallest_element_size = min((x[1] - x[0]), (y[1] - y[0]))
        x_grid, y_grid = np.meshgrid(x, y)
        rectangular_domain_points = np.zeros((x.size * y.size, 2))
        rectangular_domain_points[:, 0] = x_grid.ravel()
        rectangular_domain_points[:, 1] = y_grid.ravel()
        triangulation = Delaunay(rectangular_domain_points)
        nodal_coordinates = triangulation.points
        element_connectivity = triangulation.simplices
        x_coordinates = nodal_coordinates[:, 0].ravel()
        y_coordinates = nodal_coordinates[:, 1].ravel()
        geometric_tolerance = min(1.0e-6, 0.5 * smallest_element_size)
        nodeset_left_boundary_indices = np.argwhere(x_coordinates < geometric_tolerance)
        nodeset_right_boundary_indices = np.argwhere(x_coordinates > (domain_width - geometric_tolerance))
        nodeset_bottom_boundary_indices = np.argwhere(y_coordinates < geometric_tolerance)
        nodeset_top_boundary_indices = np.argwhere(y_coordinates > (domain_height - geometric_tolerance))
        point_sets = {"LeftBoundaryNodeset": nodeset_left_boundary_indices,
                      "RightBoundaryNodeset": nodeset_right_boundary_indices,
                      "BottomBoundaryNodeset": nodeset_bottom_boundary_indices,
                      "TopBoundaryNodeset": nodeset_top_boundary_indices}
        cell_blocks = [meshio.CellBlock("triangle", element_connectivity)]
        meshio_mesh = meshio.Mesh(nodal_coordinates, cell_blocks, point_sets=point_sets)
        meshio_mesh.write(mesh_filepath)
    yield _create_meshio_mesh
    if os.path.exists(mesh_filepath):
        os.remove(mesh_filepath)


#######################################################################################################################
#######################################################################################################################
@pytest.fixture(scope="function")
def rve_and_macroscale_parameter_dictionaries(macroscale_mesh_filepath, temporary_output_directory):
    def simp_exponent_continuation_function(iteration_number: int) -> float:
        assert iteration_number >= 0
        return 2.0

    def smooth_heaviside_projection_continuation_function(iteration_number: int) -> float:
        assert iteration_number >= 0
        return 2.0

    def design_variable_initialization_function(nodal_x_coordinates: np.ndarray,
                                                nodal_y_coordinates: np.ndarray) -> np.ndarray:
        assert nodal_x_coordinates.size == nodal_y_coordinates.size
        return np.random.uniform(0.3, 0.7, nodal_x_coordinates.size)

    output_directory_path = os.path.join(temporary_output_directory, "output")

    representative_volume_element_parameters = \
        {
            "number of elements along each edge": 10,
            "stiff material elastic modulus": 1.0e3,
            "soft material elastic modulus": 1.0,
            "poissons ratio": 0.3,
            "SIMP exponent continuation function": simp_exponent_continuation_function,
            "smooth Heaviside projection continuation function": smooth_heaviside_projection_continuation_function,
            "design variable initialization function": design_variable_initialization_function,
            "density filter radius": 2.0/10.0,
            "output directory path": output_directory_path
        }

    fixed_boundary_nodesets = {"Fixed X Displacement Nodesets": ["LeftBoundaryNodeset"],
                               "Fixed Y Displacement Nodesets": ["LeftBoundaryNodeset"]}
    applied_load_1 = {"Nodeset": "RightBoundaryNodeset", "Load in X direction": 1.0, "Load in Y direction": 0.0}
    applied_loads = [applied_load_1]
    macroscale_problem_parameters = \
        {
            "macroscale finite element mesh filepath": macroscale_mesh_filepath,
            "fixed boundary condition nodesets": fixed_boundary_nodesets,
            "applied loads": applied_loads,
            "output directory path": output_directory_path
        }
    yield (representative_volume_element_parameters, macroscale_problem_parameters)


#######################################################################################################################
#######################################################################################################################
@pytest.fixture(scope="session")
def check_sensitivities():
    def percent_difference(analytical, numerical):
        return 100.0 * np.abs(analytical - numerical) / np.abs(numerical)

    def _check_sensitivities(macroscale_problem, identifying_string, print_values=False):
        initial_design_variables = macroscale_problem.get_initial_design_variables()
        macroscale_problem.run_macroscale_forward_analysis(initial_design_variables)

        _, analytical_compliance_sensitivity_vector = \
            macroscale_problem.get_macroscale_compliance_value_and_sensitivity()
        _, analytical_volume_fraction_sensitivity_vector = \
            macroscale_problem.get_volume_fraction_value_and_sensitivity()

        numerical_compliance_sensitivity_vector = np.zeros_like(analytical_compliance_sensitivity_vector)
        numerical_volume_fraction_sensitivity_vector = np.zeros_like(analytical_volume_fraction_sensitivity_vector)

        perturbation = 4.0e-3
        for i in range(initial_design_variables.size):
            new_design_variables = initial_design_variables.copy()
            new_design_variables[i] += 2.0 * perturbation
            macroscale_problem.run_macroscale_forward_analysis(new_design_variables)
            forward2_compliance_value, _ = macroscale_problem.get_macroscale_compliance_value_and_sensitivity()
            forward2_volume_fraction_value, _ = macroscale_problem.get_volume_fraction_value_and_sensitivity()

            new_design_variables = initial_design_variables.copy()
            new_design_variables[i] += perturbation
            macroscale_problem.run_macroscale_forward_analysis(new_design_variables)
            forward_compliance_value, _ = macroscale_problem.get_macroscale_compliance_value_and_sensitivity()
            forward_volume_fraction_value, _ = macroscale_problem.get_volume_fraction_value_and_sensitivity()

            new_design_variables = initial_design_variables.copy()
            new_design_variables[i] -= perturbation
            macroscale_problem.run_macroscale_forward_analysis(new_design_variables)
            backward_compliance_value, _ = macroscale_problem.get_macroscale_compliance_value_and_sensitivity()
            backward_volume_fraction_value, _ = macroscale_problem.get_volume_fraction_value_and_sensitivity()

            new_design_variables = initial_design_variables.copy()
            new_design_variables[i] -= 2.0*perturbation
            macroscale_problem.run_macroscale_forward_analysis(new_design_variables)
            backward2_compliance_value, _ = macroscale_problem.get_macroscale_compliance_value_and_sensitivity()
            backward2_volume_fraction_value, _ = macroscale_problem.get_volume_fraction_value_and_sensitivity()

            numerical_compliance_sensitivity = (backward2_compliance_value - forward2_compliance_value + \
                    8.0*(forward_compliance_value - backward_compliance_value)) / (12.0 * perturbation)
            numerical_compliance_sensitivity_vector[i] = numerical_compliance_sensitivity

            numerical_volume_fraction_sensitivity = (backward2_volume_fraction_value - forward2_volume_fraction_value \
                + 8.0*(forward_volume_fraction_value - backward_volume_fraction_value)) / (12.0 * perturbation)
            numerical_volume_fraction_sensitivity_vector[i] = numerical_volume_fraction_sensitivity

            if print_values:
                analytical_compliance_sensitivity = analytical_compliance_sensitivity_vector[i]
                perc_diff = percent_difference(analytical_compliance_sensitivity, numerical_compliance_sensitivity)
                ratio = analytical_compliance_sensitivity / numerical_compliance_sensitivity
                my_string = (f"A {analytical_compliance_sensitivity:11.4e} | "
                    f"N {numerical_compliance_sensitivity:11.4e} | PD {perc_diff:10.4f} | Ratio {ratio:8.2f}")

                analytical_volume_fraction_sensitivity = analytical_volume_fraction_sensitivity_vector[i]
                perc_diff = percent_difference(analytical_volume_fraction_sensitivity,
                                                numerical_volume_fraction_sensitivity)
                ratio = analytical_volume_fraction_sensitivity / numerical_volume_fraction_sensitivity
                my_string += (f"  ||||  A {analytical_volume_fraction_sensitivity:11.4e} "
                    f"| N {numerical_volume_fraction_sensitivity:11.4e} | PD {perc_diff:10.4f} | Ratio {ratio:8.2f}")
                print(my_string)

        compliance_sensitivity_percent_differences = \
            percent_difference(analytical_compliance_sensitivity_vector,
                               numerical_compliance_sensitivity_vector)
        volume_fraction_sensitivity_percent_differences = \
            percent_difference(analytical_volume_fraction_sensitivity_vector,
                               numerical_volume_fraction_sensitivity_vector)
        assert np.all(compliance_sensitivity_percent_differences < 1.0e-3), \
            f"Compliance sensitivity check failed. Indentifying string = '{identifying_string}'"
        assert np.all(volume_fraction_sensitivity_percent_differences < 5.0e-4), \
            f"Volume fraction sensitivity check failed. Indentifying string = '{identifying_string}'"
    return _check_sensitivities
