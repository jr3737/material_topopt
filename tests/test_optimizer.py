"""Test optimizer and its restart capability."""
import os
import logging
import numpy as np
import src.material_topopt.utilities as utils
from src.material_topopt.optimizer import MaterialOptimizer


#######################################################################################################################
#######################################################################################################################
def test_optimizer_and_restart(create_macroscale_mesh, rve_and_macroscale_parameter_dictionaries):
    logging_level = logging.ERROR
    utils.setup_logging(logging_level, logfile_path='')
    domain_width = 4.0
    domain_height = 1.0
    create_macroscale_mesh(domain_width=domain_width,
                           domain_height=domain_height,
                           number_of_elements_along_width=8,
                           number_of_elements_along_height=1)
    representative_volume_element_parameters, macroscale_problem_parameters = rve_and_macroscale_parameter_dictionaries

    representative_volume_element_parameters["design variable initialization function"] = \
        lambda x, y: np.random.uniform(0.1, 0.9, x.size)

    output_directory_path = macroscale_problem_parameters["output directory path"]
    optimization_problem_parameters = \
        {
            "maximum number of iterations": 10,
            "volume fraction constraint upper bound": 0.5,
            "restart iteration number": 0,
            "restart file write frequency": 5,
            "MMA move limit": 0.1,
            "optimization history output filepath": os.path.join(output_directory_path, "MaterialTopOptData.csv")
        }
    material_optimizer = \
        MaterialOptimizer(optimization_problem_parameters = optimization_problem_parameters,
                          macroscale_problem_parameters = macroscale_problem_parameters,
                          representative_volume_element_parameters = representative_volume_element_parameters)
    material_optimizer.macroscale_problem.plot_mesh()
    material_optimizer.macroscale_problem.representative_volume_element.plot_mesh()
    material_optimizer.run()
    material_optimizer.plot_history()

    restart_file_path = os.path.join(output_directory_path, "mma_restart_iteration_0010.npz")
    data_from_iteration_10 = np.load(restart_file_path)
    design_variables_at_iteration_10 = data_from_iteration_10['xval']

    # Now restart from iteration 5 and check to make sure you get the same answer at iteration 10
    optimization_problem_parameters["restart iteration number"] = 5
    material_optimizer_restarted = \
        MaterialOptimizer(optimization_problem_parameters = optimization_problem_parameters,
                          macroscale_problem_parameters = macroscale_problem_parameters,
                          representative_volume_element_parameters = representative_volume_element_parameters)
    material_optimizer_restarted.run()

    new_data_from_iteration_10 = np.load(restart_file_path)
    new_design_variables_at_iteration_10 = new_data_from_iteration_10['xval']

    assert np.allclose(design_variables_at_iteration_10, new_design_variables_at_iteration_10)
