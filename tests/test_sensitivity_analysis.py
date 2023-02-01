"""Test macroscale compliance and volume fraction sensitivities numerically."""
from src.material_topopt.macroscale_problem import MacroscaleProblem2D


#######################################################################################################################
#######################################################################################################################
def test_cantilever_sensitivities(create_macroscale_mesh,
                                  rve_and_macroscale_parameter_dictionaries,
                                  check_sensitivities):
    create_macroscale_mesh(domain_width=8.0,
                           domain_height=1.0,
                           number_of_elements_along_width=15,
                           number_of_elements_along_height=3)
    representative_volume_element_parameters, macroscale_problem_parameters = rve_and_macroscale_parameter_dictionaries

    macroscale_problem_parameters["applied loads"][0]["Load in Y direction"] = -0.1

    macroscale_problem = MacroscaleProblem2D(macroscale_problem_parameters, representative_volume_element_parameters)

    check_sensitivities(macroscale_problem, "test_cantilever_sensitivities", print_values=False)


#######################################################################################################################
#######################################################################################################################
def test_axial_sensitivities(create_macroscale_mesh,
                             rve_and_macroscale_parameter_dictionaries,
                             check_sensitivities):
    create_macroscale_mesh(domain_width=4.0,
                           domain_height=4.0,
                           number_of_elements_along_width=4,
                           number_of_elements_along_height=6)
    representative_volume_element_parameters, macroscale_problem_parameters = rve_and_macroscale_parameter_dictionaries

    macroscale_problem_parameters["fixed boundary condition nodesets"]["Fixed Y Displacement Nodesets"][0] = \
        "BottomBoundaryNodeset"
    macroscale_problem_parameters["applied loads"][0]["Load in X direction"] = -7.0
    macroscale_problem_parameters["applied loads"][0]["Load in Y direction"] =  0.0

    macroscale_problem = MacroscaleProblem2D(macroscale_problem_parameters, representative_volume_element_parameters)

    check_sensitivities(macroscale_problem, "test_axial_sensitivities", print_values=False)
