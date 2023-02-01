"""Test macroscale problem and homogenized properties."""
import numpy as np
import src.material_topopt.utilities as utils
from src.material_topopt.macroscale_problem import MacroscaleProblem2D


#######################################################################################################################
#######################################################################################################################
def test_homogeneous_stiff_material_properties_axial_load(create_macroscale_mesh,
                                                          rve_and_macroscale_parameter_dictionaries):
    domain_width = 4.0
    domain_height = 1.0
    create_macroscale_mesh(domain_width=domain_width,
                           domain_height=domain_height,
                           number_of_elements_along_width=8,
                           number_of_elements_along_height=1)
    representative_volume_element_parameters, macroscale_problem_parameters = rve_and_macroscale_parameter_dictionaries

    representative_volume_element_parameters["design variable initialization function"] = lambda x, y: np.ones_like(x)
    elastic_modulus = representative_volume_element_parameters["stiff material elastic modulus"]
    poissons_ratio = representative_volume_element_parameters["poissons ratio"]

    macroscale_problem_parameters["fixed boundary condition nodesets"]["Fixed Y Displacement Nodesets"][0] = \
        "BottomBoundaryNodeset"
    axial_load = 2.0
    macroscale_problem_parameters["applied loads"][0]["Load in X direction"] = axial_load
    macroscale_problem_parameters["applied loads"][0]["Load in Y direction"] = 0.0

    axial_stress = axial_load / domain_height
    axial_strain = axial_stress / elastic_modulus
    end_displacement = axial_strain * domain_width

    macroscale_problem = MacroscaleProblem2D(macroscale_problem_parameters, representative_volume_element_parameters)
    initial_design_variables = macroscale_problem.get_initial_design_variables()
    macroscale_problem.run_macroscale_forward_analysis(initial_design_variables)
    new_shape = (macroscale_problem.number_of_nodes, 2)
    nodal_displacements = macroscale_problem.displacement_solution_vector.reshape(new_shape)
    largest_x_displacement_magnitude = np.amax(np.abs(nodal_displacements[:, 0]))
    largest_y_displacement_magnitude = np.amax(np.abs(nodal_displacements[:, 1]))

    coarse_scale_constitutive_tensor = macroscale_problem.get_coarse_scale_constitutive_tensor()
    standard_isotropic_constitutive_tensor = utils.get_2d_isotropic_linear_elastic_4th_order_constitutive_tensor(
        elastic_modulus=elastic_modulus, poissons_ratio=poissons_ratio
    )
    assert np.allclose(coarse_scale_constitutive_tensor, standard_isotropic_constitutive_tensor)

    assert np.allclose(largest_x_displacement_magnitude, end_displacement)
    assert np.allclose(largest_y_displacement_magnitude, axial_strain * poissons_ratio * domain_height)


#######################################################################################################################
#######################################################################################################################
def test_homogeneous_soft_material_properties_axial_load(create_macroscale_mesh,
                                                         rve_and_macroscale_parameter_dictionaries):
    domain_width = 4.0
    domain_height = 1.0
    create_macroscale_mesh(domain_width=domain_width,
                           domain_height=domain_height,
                           number_of_elements_along_width=10,
                           number_of_elements_along_height=1)
    representative_volume_element_parameters, macroscale_problem_parameters = rve_and_macroscale_parameter_dictionaries

    representative_volume_element_parameters["design variable initialization function"] = lambda x, y: np.zeros_like(x)
    elastic_modulus = representative_volume_element_parameters["soft material elastic modulus"]
    poissons_ratio = representative_volume_element_parameters["poissons ratio"]

    macroscale_problem_parameters["fixed boundary condition nodesets"]["Fixed Y Displacement Nodesets"][0] = \
        "BottomBoundaryNodeset"
    axial_load = 2.0
    macroscale_problem_parameters["applied loads"][0]["Load in X direction"] = axial_load
    macroscale_problem_parameters["applied loads"][0]["Load in Y direction"] = 0.0

    axial_stress = axial_load / domain_height
    axial_strain = axial_stress / elastic_modulus
    end_displacement = axial_strain * domain_width

    macroscale_problem = MacroscaleProblem2D(macroscale_problem_parameters, representative_volume_element_parameters)
    initial_design_variables = macroscale_problem.get_initial_design_variables()
    macroscale_problem.run_macroscale_forward_analysis(initial_design_variables)
    new_shape = (macroscale_problem.number_of_nodes, 2)
    nodal_displacements = macroscale_problem.displacement_solution_vector.reshape(new_shape)
    largest_x_displacement_magnitude = np.amax(np.abs(nodal_displacements[:, 0]))
    largest_y_displacement_magnitude = np.amax(np.abs(nodal_displacements[:, 1]))

    coarse_scale_constitutive_tensor = macroscale_problem.get_coarse_scale_constitutive_tensor()
    standard_isotropic_constitutive_tensor = utils.get_2d_isotropic_linear_elastic_4th_order_constitutive_tensor(
        elastic_modulus=elastic_modulus, poissons_ratio=poissons_ratio
    )
    assert np.allclose(coarse_scale_constitutive_tensor, standard_isotropic_constitutive_tensor)

    assert np.allclose(largest_x_displacement_magnitude, end_displacement)
    assert np.allclose(largest_y_displacement_magnitude, axial_strain * poissons_ratio * domain_height)


#######################################################################################################################
#######################################################################################################################
def test_homogeneous_stiff_material_properties_cantilever_load(create_macroscale_mesh,
                                                               rve_and_macroscale_parameter_dictionaries):
    domain_width = 10.0
    domain_height = 1.0
    create_macroscale_mesh(domain_width=domain_width,
                           domain_height=domain_height,
                           number_of_elements_along_width=200,
                           number_of_elements_along_height=20)
    representative_volume_element_parameters, macroscale_problem_parameters = rve_and_macroscale_parameter_dictionaries

    representative_volume_element_parameters["design variable initialization function"] = lambda x, y: np.ones_like(x)
    elastic_modulus = representative_volume_element_parameters["stiff material elastic modulus"]
    poissons_ratio = 0.0
    representative_volume_element_parameters["poissons ratio"] = poissons_ratio

    y_load = 0.25
    macroscale_problem_parameters["applied loads"][0]["Load in X direction"] = 0.0
    macroscale_problem_parameters["applied loads"][0]["Load in Y direction"] = y_load

    moment_of_inertia = 1.0 / 12.0
    cantilever_siffness = 3.0 * elastic_modulus * moment_of_inertia / (domain_width**3)
    end_displacement = y_load / cantilever_siffness

    macroscale_problem = MacroscaleProblem2D(macroscale_problem_parameters, representative_volume_element_parameters)
    initial_design_variables = macroscale_problem.get_initial_design_variables()
    macroscale_problem.run_macroscale_forward_analysis(initial_design_variables)
    new_shape = (macroscale_problem.number_of_nodes, 2)
    nodal_displacements = macroscale_problem.displacement_solution_vector.reshape(new_shape)
    largest_y_displacement_magnitude = np.amax(np.abs(nodal_displacements[:, 1]))

    coarse_scale_constitutive_tensor = macroscale_problem.get_coarse_scale_constitutive_tensor()
    standard_isotropic_constitutive_tensor = utils.get_2d_isotropic_linear_elastic_4th_order_constitutive_tensor(
        elastic_modulus=elastic_modulus, poissons_ratio=poissons_ratio
    )
    assert np.allclose(coarse_scale_constitutive_tensor, standard_isotropic_constitutive_tensor)

    assert np.allclose(largest_y_displacement_magnitude, end_displacement, rtol=1.0e-2)


#######################################################################################################################
#######################################################################################################################
def test_inhomogeneous_material_properties_axial_load(create_macroscale_mesh,
                                                      rve_and_macroscale_parameter_dictionaries):
    domain_width = 1.0
    domain_height = 1.0
    create_macroscale_mesh(domain_width=domain_width,
                           domain_height=domain_height,
                           number_of_elements_along_width=1,
                           number_of_elements_along_height=1)
    representative_volume_element_parameters, macroscale_problem_parameters = rve_and_macroscale_parameter_dictionaries

    def initialize_design_variables(x_coordinates, y_coordinates):
        z = np.zeros_like(x_coordinates)
        z[np.abs(y_coordinates - 0.5) <= 0.25] = 1.0
        return z

    representative_volume_element_parameters["design variable initialization function"] = initialize_design_variables
    representative_volume_element_parameters["density filter radius"] = 1.0e-8
    representative_volume_element_parameters["number of elements along each edge"] = 151
    stiff_elastic_modulus = representative_volume_element_parameters["stiff material elastic modulus"]
    soft_elastic_modulus  = representative_volume_element_parameters["soft material elastic modulus"]
    elastic_modulus = 0.5 * (stiff_elastic_modulus + soft_elastic_modulus)
    poissons_ratio = 0.0
    representative_volume_element_parameters["poissons ratio"] = poissons_ratio

    macroscale_problem_parameters["fixed boundary condition nodesets"]["Fixed Y Displacement Nodesets"][0] = \
        "BottomBoundaryNodeset"
    axial_load = 2.0
    macroscale_problem_parameters["applied loads"][0]["Load in X direction"] = axial_load
    macroscale_problem_parameters["applied loads"][0]["Load in Y direction"] = 0.0

    axial_stress = axial_load / domain_height
    axial_strain = axial_stress / elastic_modulus
    end_displacement = axial_strain * domain_width

    macroscale_problem = MacroscaleProblem2D(macroscale_problem_parameters, representative_volume_element_parameters)
    initial_design_variables = macroscale_problem.get_initial_design_variables()
    macroscale_problem.run_macroscale_forward_analysis(initial_design_variables)
    new_shape = (macroscale_problem.number_of_nodes, 2)
    nodal_displacements = macroscale_problem.displacement_solution_vector.reshape(new_shape)
    largest_x_displacement_magnitude = np.amax(np.abs(nodal_displacements[:, 0]))

    coarse_scale_constitutive_tensor = macroscale_problem.get_coarse_scale_constitutive_tensor()
    voigt_constitutive_tensor = utils.get_2d_isotropic_linear_elastic_4th_order_constitutive_tensor(
        elastic_modulus=elastic_modulus, poissons_ratio=poissons_ratio
    )
    assert np.allclose(coarse_scale_constitutive_tensor[0, 0, 0, 0], voigt_constitutive_tensor[0, 0, 0, 0], rtol=1.0e-3)
    assert np.allclose(largest_x_displacement_magnitude, end_displacement, rtol=1.0e-3)
