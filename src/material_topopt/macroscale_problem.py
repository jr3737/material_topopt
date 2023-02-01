"""Module containing the MacroscaleProblem2D class.

Used to create a macroscale finite element problem which uses
homogenized constitutive properties obtained from a representative
volume element (RVE). Sensitivities of the macroscale compliance with respect
to the RVE design variables can be requested for material optimization.

Typical usage example:

  >>> my_macroscale_problem = MacroscaleProblem2D(macroscale_parameters, rve_parameters)
  >>> my_macroscale_problem.run_macroscale_forward_analysis(design_variables, optimization_iteration_number=0)
  >>> objective_value, objective_gradient = my_macroscale_problem.get_macroscale_compliance_value_and_sensitivity()
  >>> constraint_value, constraint_gradient = my_macroscale_problem.get_volume_fraction_value_and_sensitivity()
"""
import os
import shutil
import logging
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import meshio

import src.material_topopt.utilities as utils
from src.material_topopt.representative_volume_element import RepresentativeVolumeElement2D

#######################################################################################################################
#######################################################################################################################
class MacroscaleProblem2D:
    """A macroscale (i.e., coarse scale) linear elastic problem.

    Uses homogenized constitutive tensor from a representative volume element and can compute the necessary
    derivatives of residual with respect to the relevant components of the coarse scale constitutive tensor
    for minimizing the coarse scale compliance by optimizing the material microstructure.

    Attributes:
        fixed_boundary_conditions: A dictionary mapping "Fixed X Displacement Nodesets" or
            "Fixed Y Displacement Nodesets" to a nodeset name in the mesh file.
        applied_loads: A list of applied_load dictionaries each containing a "Nodeset", a
            "Load in X direction" and/or a "Load in Y direction".
        nodesets: A dictionary mapping from nodeset name to a numpy array of nodal indices.
        nodal_coordinates: A numpy array of nodal coordinates.
        number_of_nodes: An integer corresponding to the number of nodes in the coarse scale mesh.
        element_connectivity: A numpy array of integers containing the 3-node triangle element connectivity.
        number_of_elements: An integer containing the number of coarse scale elements in the mesh.
        number_of_nodes_per_element: An integer containing the number of nodes per element. (Should always be 3)
        displacement_solution_vector: A numpy array containing the displacement solution after a forward analysis.
    """


    ###################################################################################################################
    ###################################################################################################################
    def __init__(self, macroscale_problem_parameters: dict, representative_volume_element_parameters: dict):
        """Initialize a macroscale linear elastic 2D problem.

        Sets the user specified parameters and creates/reads the coarse scale finite element mesh,
        precomputes many numerical quantities for reuse throughout the optimization process, and
        creates a representative volume element with the user specified parameters.

        Args:
            macroscale_problem_parameters: A dictionary containing the parameters for the macroscale problem.
            representative_volume_element_parameters: A dictionary containing the parameters for the RVE problem.

        Returns:
            None
        """
        self.__space_dimension = 2 # This code only works in 2D with 3-node triangular elements
        self.logger = logging.getLogger("MaterialTopOpt.MacroscaleProblem2D")
        self.__set_macroscale_problem_parameters(macroscale_problem_parameters)
        self.logger.debug("Initialization begin")
        self.__create_coarse_scale_finite_element_mesh()
        self.__precompute_fixed_numerical_quantities()
        self.logger.debug("Initialization end")
        self.representative_volume_element = RepresentativeVolumeElement2D(representative_volume_element_parameters)


    ###################################################################################################################
    ###################################################################################################################
    def __set_macroscale_problem_parameters(self, macroscale_problem_parameters: dict):
        """Checks and sets the macroscale parameters.

        Takes the user supplied dictionary of macroscale problem parameters and checks them prior to setting
        the corresponding attributes of the class.

        Args:
            macroscale_problem_parameters: User-specified dictionary containing macroscale problem parameters.

        Returns:
            None

        Raises:
            AssertionError: An error is thrown when parameters are not the correct type,
                necessary parameters are not specified, or files dont exist.
        """
        self.__macroscale_finite_element_mesh_filepath = utils.get_parameter_or_default(
            macroscale_problem_parameters, "macroscale finite element mesh filepath", str, required=True,
            additional_message="Currently we do not provide the ability to construct the macroscale finite element" + \
                " mesh without an external file for the macroscale problem."
            )
        assert os.path.exists(self.__macroscale_finite_element_mesh_filepath),\
            ("The finite element mesh specified at path "
            f"'{self.__macroscale_finite_element_mesh_filepath}' does not exist.")

        self.fixed_boundary_conditions = macroscale_problem_parameters["fixed boundary condition nodesets"]
        self.applied_loads = macroscale_problem_parameters["applied loads"]

        self.__output_directory_path = utils.get_parameter_or_default(
            macroscale_problem_parameters, "output directory path", str, required=False,
            default_value=os.path.join(os.getcwd(), "output")
            )
        if not os.path.exists(self.__output_directory_path):
            os.mkdir(self.__output_directory_path)


    ###################################################################################################################
    ###################################################################################################################
    def __create_coarse_scale_finite_element_mesh(self):
        """Creates the macroscale finite element mesh.

        Reads the macroscale finite element mesh using meshio. The relevant quantities are then extracted from the
        mesh object and stored as attributes of the class.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug("Create the finite element mesh")

        my_mesh = meshio.read(self.__macroscale_finite_element_mesh_filepath)
        nodal_coordinates = my_mesh.points[:, :2]
        element_connectivity = my_mesh.cells[0].data
        self.nodesets = my_mesh.point_sets

        # Save some important data for later use
        self.nodal_coordinates = nodal_coordinates
        self.number_of_nodes = nodal_coordinates.shape[0]
        self.element_connectivity = element_connectivity
        self.number_of_elements, self.number_of_nodes_per_element = element_connectivity.shape


    ###################################################################################################################
    ###################################################################################################################
    def __compute_macroscale_compliance_derivative_wrt_constitutive_tensor_components(self):
        """Compute compliance derivative with respect to constitutive tensor.

        Compute the macroscale compliance derivative with respect to the 6 (in 2D) independent components
        of the homogenized coarse scale constitutive tensor.

        Args:
            None

        Returns:
            None
        """
        # six independent components of the coarse scale constitutive tensor in 2D
        self.__compliance_derivative_wrt_coarse_scale_constitutive_tensor_components = np.zeros((6,))

        nodal_displacements_elementwise = self.displacement_solution_vector[self.__residual_vector_indices_elementwise]
        small_strain_tensor_elementwise = \
            np.einsum('eaij,ea->eij', self.__virtual_displacement_symmetric_gradients, nodal_displacements_elementwise)

        adjoint_vector = -1.0 * self.displacement_solution_vector

        rows = self.__residual_vector_indices_elementwise.ravel()
        cols = np.zeros(rows.shape, dtype=int)

        independent_tensor_components, component_symmetries = \
            utils.get_list_of_2d_constitutive_tensor_independent_components_and_symmetries()
        for component_index, (i, j, k, l) in enumerate(independent_tensor_components):
            constitutive_tensor_derivative = np.zeros((2, 2, 2, 2))
            constitutive_tensor_derivative[i, j, k, l] = 1.0

            for component_symmetry in component_symmetries[component_index]:
                a, b, c, d = component_symmetry
                constitutive_tensor_derivative[a, b, c, d] = constitutive_tensor_derivative[i, j, k, l]

            residual_vector_derivative_elementwise = np.einsum('eaij,ijkl,ekl,e->ea',
                                                               self.__virtual_displacement_symmetric_gradients,
                                                               constitutive_tensor_derivative,
                                                               small_strain_tensor_elementwise,
                                                               self.__element_areas, optimize=True)
            data = residual_vector_derivative_elementwise.ravel()
            residual_vector_derivative = sp.sparse.coo_matrix((data, (rows, cols))).toarray().ravel()
            value = np.inner(adjoint_vector, residual_vector_derivative)
            self.__compliance_derivative_wrt_coarse_scale_constitutive_tensor_components[component_index] = value


    ###################################################################################################################
    ###################################################################################################################
    def __precompute_fixed_numerical_quantities(self):
        """Precompute time consuming quantities that are repeatedly used.

        Precomputes the following quantities that do not change throughout the optimization process.
        The symmetric gradients of the virtual displacements, the element areas, the indices for the matrix
        and vector assembly process, the fixed displacement dof indices, the external force vector

        Args:
            None

        Returns:
            None
        """
        self.logger.debug("Precompute fixed numerical quantities begin")
        _, shape_function_gradients, self.__element_areas = \
            utils.get_triangle_shape_functions_and_element_areas(self.nodal_coordinates, self.element_connectivity)
        self.__virtual_displacement_symmetric_gradients = \
            utils.get_virtual_displacement_symmetric_gradients(shape_function_gradients)
        nodal_dof_indices, matrix_row_indices, \
            matrix_column_indices, self.__residual_vector_indices_elementwise = \
            utils.get_2d_linear_elasticity_problem_dof_indices(self.nodal_coordinates, self.element_connectivity)
        self.__matrix_row_indices = matrix_row_indices
        self.__matrix_col_indices = matrix_column_indices

        # fix both X and/or Y displacement
        fixed_displacement_x_nodes = []
        fixed_displacement_y_nodes = []
        fixed_displacement_dofs = []
        for k, list_of_nodeset_names in self.fixed_boundary_conditions.items():
            if isinstance(list_of_nodeset_names, str):
                list_of_nodeset_names = [list_of_nodeset_names]
            if k == "Fixed X Displacement Nodesets":
                for nodeset_name in list_of_nodeset_names:
                    assert nodeset_name in self.nodesets.keys(),\
                        f"User specified fixed boundary conditions nodeset '{nodeset_name}' does not exist in mesh." + \
                            f" Nodeset names are {list(self.nodesets.keys())}."
                    fixed_displacement_x_nodes.append(self.nodesets[nodeset_name])
                    fixed_displacement_dofs.append(nodal_dof_indices[self.nodesets[nodeset_name], 0].ravel())
            elif k == "Fixed Y Displacement Nodesets":
                for nodeset_name in list_of_nodeset_names:
                    assert nodeset_name in self.nodesets.keys(),\
                        f"User specified fixed boundary conditions nodeset '{nodeset_name}' does not exist in mesh." + \
                            f" Nodeset names are {list(self.nodesets.keys())}."
                    fixed_displacement_y_nodes.append(self.nodesets[nodeset_name])
                    fixed_displacement_dofs.append(nodal_dof_indices[self.nodesets[nodeset_name], 1].ravel())
            else:
                raise ValueError(f"User specified fixed boundary condition key '{k}' but only " + \
                    "'Fixed X Displacement Nodesets'  and 'Fixed Y Displacement Nodesets' are valid keys.")
        self.__fixed_x_boundary_condition_node_indices = np.concatenate(fixed_displacement_x_nodes)
        self.__fixed_y_boundary_condition_node_indices = np.concatenate(fixed_displacement_y_nodes)
        self.__fixed_boundary_condition_dof_indices = np.concatenate(fixed_displacement_dofs)

        # Apply loads
        number_of_global_displacement_dofs = self.number_of_nodes * self.__space_dimension
        self.__external_force_vector = np.zeros((number_of_global_displacement_dofs, ))
        for applied_load in self.applied_loads:
            assert "Nodeset" in applied_load.keys(), "The 'Nodeset' must be specified within each applied load."
            nodeset_name = applied_load["Nodeset"]
            assert nodeset_name in self.nodesets.keys(),\
                f"User specified applied load nodeset '{nodeset_name}' does not exist in mesh." + \
                    " Nodeset names are {list(self.nodesets.keys())}."
            nodeset_nodal_indices = self.nodesets[nodeset_name].ravel()
            x_load = "Load in X direction" in applied_load.keys()
            y_load = "Load in Y direction" in applied_load.keys()
            assert x_load or y_load, f"User specified applied load for nodeset '{nodeset_name}' " + \
                "has neither an applied load in the X or Y specified"
            number_of_applied_force_nodes = float(nodeset_nodal_indices.size)
            assert nodeset_nodal_indices.size > 0, f"Nodeset '{nodeset_name} has 0 nodes."
            if x_load:
                applied_x_force_dof_indices = nodal_dof_indices[nodeset_nodal_indices, 0].ravel()
                load_x_magnitude = applied_load["Load in X direction"]
                self.__external_force_vector[applied_x_force_dof_indices] = \
                    load_x_magnitude / number_of_applied_force_nodes
            if y_load:
                applied_y_force_dof_indices = nodal_dof_indices[nodeset_nodal_indices, 1].ravel()
                load_y_magnitude = applied_load["Load in Y direction"]
                self.__external_force_vector[applied_y_force_dof_indices] = \
                    load_y_magnitude / number_of_applied_force_nodes

        self.logger.debug("Precompute fixed numerical quantities end")


    ###################################################################################################################
    ###################################################################################################################
    def __assemble_macroscale_stiffness_matrix_and_force_vector_and_solve(self):
        """Performs matrix assembly and linear solve.

        Assembles the macroscale stiffness matrix using the homogenized constitutive properties
        from the RVE. The system is modified for the fixed boundary conditions and then the linear
        solve is performed using a sparse direct solver.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug("Assemble stiffness matrix and force vector")
        stiffness_matrix_elementwise = \
            utils.get_2d_linear_elasticity_problem_stiffness_matrix_elementwise(
                self.__coarse_scale_constitutive_tensor,
                self.__virtual_displacement_symmetric_gradients,
                self.__element_areas)
        data = stiffness_matrix_elementwise.ravel()
        stiffness_matrix = sp.sparse.coo_matrix((data, (self.__matrix_row_indices, self.__matrix_col_indices))).tocsc()

        # Modify matrix for fixed boundary conditions
        for fixed_dof_index in self.__fixed_boundary_condition_dof_indices:
            row = stiffness_matrix.getrow(fixed_dof_index)
            col = stiffness_matrix.getcol(fixed_dof_index)
            row.data[:] = 0.0
            col.data[:] = 0.0
            stiffness_matrix[fixed_dof_index, :] = row
            stiffness_matrix[:, fixed_dof_index] = col
            stiffness_matrix[fixed_dof_index, fixed_dof_index] = 1.0

        self.logger.debug("Perform linear solve for coarse scale displacements")
        factored_matrix_solve = utils.factorized(stiffness_matrix)
        displacement_solution_vector = factored_matrix_solve(self.__external_force_vector)
        self.displacement_solution_vector = displacement_solution_vector.ravel()


    ###################################################################################################################
    ###################################################################################################################
    def run_macroscale_forward_analysis(self,
                                        free_nodal_design_variables: np.ndarray,
                                        optimization_iteration_number: int = 0):
        """Runs the macroscale forward analysis with provided design variables.

        Passes the free nodal design variables to the RVE for computing the coarse scale constititive properties.
        Using the coarse scale constitutive tensor, the macroscale boundary value problem is solved. Then the
        macroscale compliance is computed, along with the derivative of the compliance with respect to the 6
        independent components of the coarse scale constitutive tensor. Finally, the results are output to VTK files.

        Args:
            free_nodal_design_variables: A numpy array containing the values of the RVE free nodal design variables.
            optimization_iteration_number: An integer corresponding to the current optimization iteration. This
                quantity is used primarily for naming the output files appropriately. Additionally, it is used
                by the RVE to update certain parameters via continuation (e.g., the SIMP exponent, p)

        Returns:
            None
        """
        self.representative_volume_element.run_forward_analysis(
            free_nodal_design_variables,
            optimization_iteration_number=optimization_iteration_number)
        self.__coarse_scale_constitutive_tensor = \
            self.representative_volume_element.get_coarse_scale_constitutive_tensor()
        self.__assemble_macroscale_stiffness_matrix_and_force_vector_and_solve()

        self.__macroscale_compliance = np.inner(self.__external_force_vector, self.displacement_solution_vector)
        self.__compute_macroscale_compliance_derivative_wrt_constitutive_tensor_components()

        self.logger.debug("Output VTK file")
        points_3dim = np.zeros((self.number_of_nodes, 3))
        points_3dim[:, :2] = self.nodal_coordinates
        displacements_3dim = np.zeros((self.number_of_nodes, 3))
        displacements_3dim[:, :2] = \
            self.displacement_solution_vector.reshape((self.number_of_nodes, self.__space_dimension))
        point_data = {"Displacement": displacements_3dim}
        if optimization_iteration_number == 0:
            # In the first iteration, also output the nodes that are fixed and those to which the force is applied
            fixed_displacement_nodes = np.zeros((self.number_of_nodes, 3))
            fixed_displacement_nodes[self.__fixed_x_boundary_condition_node_indices, 0] = 1.0
            fixed_displacement_nodes[self.__fixed_y_boundary_condition_node_indices, 1] = 1.0
            external_force_vector = np.zeros((self.number_of_nodes, 3))
            external_force_vector[:, :2] = self.__external_force_vector.reshape((self.number_of_nodes, 2))
            additional_point_data = {"Fixed_Displacement_Nodes": fixed_displacement_nodes,
                                     "External_Force": external_force_vector}
            point_data.update(additional_point_data)
        cell_blocks = [meshio.CellBlock('triangle', self.element_connectivity)]
        my_mesh = meshio.Mesh(points_3dim, cell_blocks, point_data=point_data)
        macroscale_output_filepath = os.path.join(self.__output_directory_path,
                                                  f"macroscale.{optimization_iteration_number:04d}.vtk")
        my_mesh.write(macroscale_output_filepath)


    ###################################################################################################################
    ###################################################################################################################
    def get_macroscale_compliance_value_and_sensitivity(self):
        """Returns the macroscale compliance and its gradient.

        The previously computed macroscale compliance value is returned in addition to the total sensitivity of the
        compliance with respect to the design variables of the RVE.

        Args:
            None

        Returns:
            A tuple (compliance_value, compliance_gradient), where compliance_value is a float, and compliance_gradient
            is a numpy array containing the derivative with respect to the RVE design variables.
        """
        macroscale_compliance_sensitivity_wrt_design_variables = \
            self.representative_volume_element.get_macroscale_compliance_sensitivity(
                self.__compliance_derivative_wrt_coarse_scale_constitutive_tensor_components
                )
        return self.__macroscale_compliance, macroscale_compliance_sensitivity_wrt_design_variables


    ###################################################################################################################
    ###################################################################################################################
    def get_volume_fraction_value_and_sensitivity(self):
        """Returns RVE stiff material volume fraction and gradient.

        Uses the previously provided design variables to compute the volume fraction of stiff material in
        the RVE and returns this quantity along with the gradient.

        Args:
            None

        Returns:
            A tuple (volume_fraction, volume_fraction_gradient), where volume_fraction is a float, and
            volume_fraction_gradient is a numpy array containing the gradient.
        """
        return self.representative_volume_element.get_volume_fraction_value_and_sensitivity()


    ###################################################################################################################
    ###################################################################################################################
    def get_coarse_scale_constitutive_tensor(self):
        """Returns the coarse scale constitutive tensor computed by the RVE.

        Args:
            None

        Returns:
            A numpy array containing the 4th order homogenized constitutive tensor.
        """
        return self.__coarse_scale_constitutive_tensor


    ###################################################################################################################
    ###################################################################################################################
    def get_initial_design_variables(self):
        """Returns the initial design variables for the RVE.

        Calls the representative volume element method for obtaining the initial array of design variables.
        Typically, these values are intialized using a user-supplied function specified in the RVE parameters.

        Args:
            None

        Returns:
            A numpy array containing the initial nodal design variables for the RVE.
        """
        return self.representative_volume_element.get_initial_free_unfiltered_nodal_design_variables()


    ###################################################################################################################
    ###################################################################################################################
    def update_logged_values(self, logged_values: dict) -> dict:
        """Update the values of anything worth writing to a logfile at each optimization iteration.

        Takes in a dictionary of values to log and updates it with additional quantities. The previous quantities are
        retained in the dictionary and the new dictionary is returned.

        Args:
            logged_values: A dictionary of useful quantities to log at each optimization iteration.

        Returns:
            An updated dictionary of logged values.
        """
        updated_logged_values = self.representative_volume_element.update_logged_values(logged_values)
        macroscale_problem_logged_values = \
            {"Coarse Scale d_StressXX_d_StainXX": self.__coarse_scale_constitutive_tensor[0, 0, 0, 0],
             "Coarse Scale d_StressXX_d_StainYY": self.__coarse_scale_constitutive_tensor[0, 0, 1, 1],
             "Coarse Scale d_StressXX_d_StainXY": self.__coarse_scale_constitutive_tensor[0, 0, 0, 1],
             "Coarse Scale d_StressYY_d_StainYY": self.__coarse_scale_constitutive_tensor[1, 1, 1, 1],
             "Coarse Scale d_StressYY_d_StainXY": self.__coarse_scale_constitutive_tensor[1, 1, 0, 1],
             "Coarse Scale d_StressXY_d_StainXY": self.__coarse_scale_constitutive_tensor[0, 1, 0, 1]}
        updated_logged_values.update(macroscale_problem_logged_values)
        return updated_logged_values


    ###################################################################################################################
    ###################################################################################################################
    def plot_mesh(self, plot_boundary_conditions: bool = True, save_figure: bool = True):
        """Plots the macroscale mesh and boundary conditions.

        Creates a matplotlib plot of the macroscale finite element mesh and the user-defined boundary conditions.

        Args:
            plot_boundary_conditions: If True the boundary conditions are displayed, in addition to the mesh.
            save_figure: If True, the figure is saved as a PDF in the output directory.

        Returns:
            None
        """
        self.logger.debug("Plotting macroscale mesh with matplotlib")
        if shutil.which("tex") and shutil.which("dvipng") and shutil.which("gs"):
            plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.triplot(self.nodal_coordinates[:, 0], self.nodal_coordinates[:, 1], self.element_connectivity,
                   color='k', linewidth=1.0, alpha=0.4)
        if plot_boundary_conditions:
            #
            fixed_node_x_coordinates = self.nodal_coordinates[self.__fixed_x_boundary_condition_node_indices, 0]
            fixed_node_y_coordinates = self.nodal_coordinates[self.__fixed_x_boundary_condition_node_indices, 1]
            ax.plot(fixed_node_x_coordinates, fixed_node_y_coordinates, 'bo', label='Fixed X Boundary Condition')
            #
            fixed_node_x_coordinates = self.nodal_coordinates[self.__fixed_y_boundary_condition_node_indices, 0]
            fixed_node_y_coordinates = self.nodal_coordinates[self.__fixed_y_boundary_condition_node_indices, 1]
            ax.plot(fixed_node_x_coordinates, fixed_node_y_coordinates, 'rx', label='Fixed Y Boundary Condition')
            #
            my_colors = 5 * ['g', 'c', 'm', 'k', 'y']
            for i, applied_load in enumerate(self.applied_loads):
                nodeset_name = applied_load["Nodeset"]
                nodeset_nodal_indices = self.nodesets[nodeset_name].ravel()
                load_in_the_x = "Load in X direction" in applied_load.keys()
                load_in_the_y = "Load in Y direction" in applied_load.keys()
                assert nodeset_nodal_indices.size > 0, f"Nodeset '{nodeset_name} has 0 nodes."
                load_x_magnitude = applied_load["Load in X direction"] if load_in_the_x else 0.0
                load_y_magnitude = applied_load["Load in Y direction"] if load_in_the_y else 0.0
                applied_load_node_x_coordinates = self.nodal_coordinates[nodeset_nodal_indices, 0].ravel()
                applied_load_node_y_coordinates = self.nodal_coordinates[nodeset_nodal_indices, 1].ravel()
                load_x_vectors = load_x_magnitude * np.ones_like(applied_load_node_x_coordinates)
                load_y_vectors = load_y_magnitude * np.ones_like(applied_load_node_y_coordinates)
                my_label = f"Nodeset {nodeset_name} Applied Load"
                ax.quiver(applied_load_node_x_coordinates, applied_load_node_y_coordinates,
                          load_x_vectors, load_y_vectors, color=my_colors[i], label=my_label)
            ax.legend(bbox_to_anchor=(0.5, 1.35), loc='upper center')
        ax.set_xlabel('Macroscale X coordinate')
        ax.set_ylabel('Macroscale Y coordinate')
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        if save_figure:
            figure_filepath = os.path.join(self.__output_directory_path, "MacroscaleMesh.pdf")
            fig.savefig(figure_filepath)
