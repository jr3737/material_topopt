"""Module containing a periodic PDE filter class and a smooth Heaviside projection class.

The periodic density filter handles the periodic filter operation for a square RVE and also contains an
instance of the smooth Heaviside projection class. The forward operations are provided in addition to the
chain rule operations that are necessary when computing gradients of functions with respect to the free
nodal design variables.
"""
import logging
import numpy as np
import scipy as sp
from typing import Callable

import src.material_topopt.utilities as utils


#######################################################################################################################
#######################################################################################################################
class PeriodicDensityFilter:
    """PDE filter with periodic boundary conditions for a representative volume element.

    A spatially periodic PDE filter for a square RVE with a nodal design variable field. Currently only works with
    a square/rectangular RVE constructed with linear triangular finite elements.

    Attributes:
        smooth_heaviside_projection: A SmoothHeavisideProjection object that handles the projection operations.
    """

    ###################################################################################################################
    ###################################################################################################################
    def __init__(self,
                 parameters: dict,
                 smooth_heaviside_parameter_continuation_function: Callable,
                 maximum_heaviside_projection_parameter: float):
        """Initialization of the periodic density filter.

        Takes in a dictionary of parameters that were previously computed in the RVE. Additionally, this object creates
        the smooth Heaviside projection function using the second and third arguments and handles invoking this
        projection and the chain rule derivatives as well. The filter is important because it ensures that the nodal
        density field remains periodic.

        Args:
            parameters: A dictionary of many parameters that were previously computed by the RVE.
            smooth_heaviside_parameter_continuation_function: A callable function taking the iteration number and
                returning the current value of the smooth Heaviside projection parameter, \\beta.
            maximum_heaviside_projection_parameter: A float that is the upper bound on the smooth Heaviside projection
                parameter computed in the RVE using the filter radius and the element size.

        Returns:
            None
        """
        self.logger = logging.getLogger("MaterialTopOpt.PeriodicDensityFilter")
        self.logger.debug("Initialize")
        element_areas = parameters["element_areas"]
        shape_function_gradients = parameters["shape_function_gradients"]
        self.__shape_function_values = parameters["shape_function_values"]
        self.__filter_radius = parameters["filter_radius"]
        element_connectivity = parameters["element_connectivity"]
        self.__number_of_elements = element_connectivity.shape[0]
        self.__number_of_nodes_per_element = element_connectivity.shape[1]
        self.__nodal_coordinates = parameters["nodal_coordinates"]
        self.__number_of_nodes = self.__nodal_coordinates.shape[0]
        self.__interior_node_indices = parameters["interior_node_indices"]
        self.__free_boundary_node_indices = parameters["free_boundary_node_indices"]
        self.__forced_boundary_node_indices = parameters["forced_boundary_node_indices"]
        self.__bottom_left_corner_node_index = parameters["bottom_left_corner_node_index"]
        self.__forced_corner_node_indices = parameters["forced_corner_node_indices"]
        nodal_dof_indices = np.arange(self.__number_of_nodes, dtype=int)
        self.__nodal_dof_indices_elementwise = nodal_dof_indices[element_connectivity]
        self.__bottom_left_corner_and_interior_and_free_dof_indices = np.concatenate(
            [self.__bottom_left_corner_node_index, self.__interior_node_indices, self.__free_boundary_node_indices])
        self.__all_corner_node_indices = np.concatenate(
            [self.__bottom_left_corner_node_index, self.__forced_corner_node_indices])
        self.__element_areas_x_shape_function_values = np.einsum('e,n->en', element_areas, self.__shape_function_values)
        self.__element_areas = element_areas
        #
        self.__create_all_matrix_maps()
        self.__create_filter_matrix(shape_function_gradients, self.__shape_function_values, element_areas)
        self.smooth_heaviside_projection = \
            SmoothHeavisideProjection(smooth_heaviside_parameter_continuation_function,
                                      maximum_heaviside_projection_parameter)


    ###################################################################################################################
    ###################################################################################################################
    def __create_all_matrix_maps(self):
        """Create sparse matrix maps that are repetitively used for efficient linear mapping operations.

        Many important operations can be represented via sparse matrix multiplications and those matrices are
        constructed and stored here. This includes maps from free nodal values to all nodal values and
        nodal field values to element quadrature points. The reverse maps are also computed via the transpose.

        Args:
            None

        Returns:
            None
        """
        data = np.ones((self.__number_of_nodes,), dtype=float)
        cols = np.zeros(data.shape, dtype=int)
        rows = np.zeros(data.shape, dtype=int)
        reduced_bottom_left_corner_index = 0
        i = self.__interior_node_indices.size + 1
        reduced_interior_node_indices = np.arange(1, i, dtype=int)
        j = i + self.__free_boundary_node_indices.size
        reduced_free_boundary_indices = np.arange(i, j, dtype=int)
        assert (1 + reduced_interior_node_indices.size + reduced_free_boundary_indices.size) == \
            self.__bottom_left_corner_and_interior_and_free_dof_indices.size
        assert (self.__all_corner_node_indices.size + self.__interior_node_indices.size +
            self.__free_boundary_node_indices.size + self.__forced_boundary_node_indices.size) == self.__number_of_nodes
        # corner nodes
        rows[self.__all_corner_node_indices] = self.__all_corner_node_indices
        cols[self.__all_corner_node_indices] = reduced_bottom_left_corner_index
        # interior nodes
        rows[self.__interior_node_indices] = self.__interior_node_indices
        cols[self.__interior_node_indices] = reduced_interior_node_indices
        # free boundary nodes
        rows[self.__free_boundary_node_indices] = self.__free_boundary_node_indices
        cols[self.__free_boundary_node_indices] = reduced_free_boundary_indices
        # forced boundary nodes
        rows[self.__forced_boundary_node_indices] = self.__forced_boundary_node_indices
        cols[self.__forced_boundary_node_indices] = reduced_free_boundary_indices
        # create the matrix map
        self.__free_to_all_matrix_map = sp.sparse.coo_matrix((data, (rows, cols))).tocsr()
        self.__all_to_free_matrix_map = self.__free_to_all_matrix_map.transpose()
        # Nodal field to element quadrature point map
        element_indices = np.arange(self.__number_of_elements, dtype=int)
        row_indices_elementwise = np.zeros((self.__number_of_elements, self.__number_of_nodes_per_element), dtype=int)
        for i in range(self.__number_of_nodes_per_element):
            row_indices_elementwise[:, i] = element_indices
        rows = row_indices_elementwise.ravel()
        cols = self.__nodal_dof_indices_elementwise.ravel()
        map_data_elementwise = np.zeros(self.__nodal_dof_indices_elementwise.shape, dtype=float)
        map_data_elementwise[:, :] = self.__shape_function_values[None, :]
        map_data = map_data_elementwise.ravel()
        self.__nodal_field_to_element_quadrature_points_map = sp.sparse.coo_matrix((map_data, (rows, cols))).tocsr()
        self.__element_quadrature_points_to_nodal_field_map = \
            self.__nodal_field_to_element_quadrature_points_map.transpose()


    ###################################################################################################################
    ###################################################################################################################
    def __create_filter_matrix(self,
                               shape_function_gradients: np.ndarray,
                               shape_function_values: np.ndarray,
                               element_areas: np.ndarray):
        """Creation of the PDE filter matrix and prefactorization for later use.

        The PDE filter matrix is constructed and condensed in a manner consistent with the periodic boundary
        conditions. The matrix is also factored in this function a single time and the factorization is repetitively
        used throughout the optimization history.

        Args:
            shape_function_gradients: A numpy array containing the shape function gradients for every element.
            shape_function_values: A numpy array containing the shape function values for every element.
            element_areas: A numpy array containing the area of every element.

        Returns:
            None
        """
        self.logger.debug("Create filter matrix and factor start")
        r_squared = self.__filter_radius * self.__filter_radius / 12.0
        diffusion_term = np.einsum('...,emd,end,e->emn', r_squared,
                                   shape_function_gradients, shape_function_gradients, element_areas)
        mass_term = np.einsum('e,m,n->emn', element_areas, shape_function_values, shape_function_values)
        filter_matrix_elementwise = diffusion_term + mass_term
        data = filter_matrix_elementwise.ravel()

        number_of_dofs_per_element = self.__number_of_nodes_per_element
        column_indices = \
            np.tile(self.__nodal_dof_indices_elementwise, (number_of_dofs_per_element, 1, 1)).transpose((1, 0, 2))
        row_indices = column_indices.transpose((0, 2, 1))
        rows = row_indices.flatten()
        cols = column_indices.flatten()

        filter_matrix_uncondensed = sp.sparse.coo_matrix((data, (rows, cols))).tocsr()

        # Apply periodic boundary conditions
        filter_matrix_condensed = \
            ( self.__all_to_free_matrix_map @ (filter_matrix_uncondensed @ self.__free_to_all_matrix_map) )
        filter_matrix_condensed_csc = filter_matrix_condensed.tocsc()
        self.__factored_matrix_solve = utils.factorized(filter_matrix_condensed_csc)

        self.logger.debug("Create filter matrix and factor end")


    ###################################################################################################################
    ###################################################################################################################
    def apply_filter_to_nodal_design_variables_and_return_element_densities(self,
                                                                            free_design_variables: np.ndarray,
                                                                            optimization_iteration_number: int = 0):
        """Apply PDE filter to design variables and get element values at quadrature points.

        The PDE filter is applied to the nodal design variables, followed by the smooth heaviside projection. Then
        the filtered and projected nodal values are used in order to obtain the values of the field at every element
        quadrature point.

        Args:
            free_design_variables: A numpy array containing the free nodal design variables from the optimizer.
            optimization_iteration_number: An integer representating the current optimization iteration number.

        Returns:
            A tuple (a, b) where 'a' is a numpy array containing the element densities and 'b' contains the
            projected and filtered nodal densities which are used to visualize the RVE design in Paraview.
        """
        self.logger.debug("Apply filter to free design variables begin")

        all_design_variables = self.__free_to_all_matrix_map @ free_design_variables
        all_design_variables_at_quad_points = self.__nodal_field_to_element_quadrature_points_map @ all_design_variables
        filter_rhs_elementwise = \
            np.einsum('e,en->en', all_design_variables_at_quad_points, self.__element_areas_x_shape_function_values)

        data = filter_rhs_elementwise.ravel()
        rows = self.__nodal_dof_indices_elementwise.ravel()
        cols = np.zeros(rows.shape, dtype=int)
        filter_rhs_vector_uncondensed = sp.sparse.coo_matrix((data, (rows, cols))).toarray().ravel()

        filter_rhs_vector_condensed = self.__all_to_free_matrix_map @ filter_rhs_vector_uncondensed

        free_filtered_densities = self.__factored_matrix_solve(filter_rhs_vector_condensed)

        all_filtered_densities = self.__free_to_all_matrix_map @ (free_filtered_densities.ravel())

        if np.any(all_filtered_densities < 0.0):
            all_filtered_densities[all_filtered_densities < 0.0] = 0.0
        if np.any(all_filtered_densities > 1.0):
            all_filtered_densities[all_filtered_densities > 1.0] = 1.0

        projected_and_filtered_nodal_densities = self.smooth_heaviside_projection.apply(
            all_filtered_densities, optimization_iteration_number=optimization_iteration_number)

        filtered_densities_at_element_centroids = \
            self.__nodal_field_to_element_quadrature_points_map @ projected_and_filtered_nodal_densities

        self.logger.debug("Apply filter to free design variables end")
        return filtered_densities_at_element_centroids, projected_and_filtered_nodal_densities


    ###################################################################################################################
    ###################################################################################################################
    def apply_sensitivity_chain_rule_derivative(self,
                                                sensitivity_wrt_element_densities: np.ndarray):
        """Given the gradient wrt element variables, return gradient wrt free design variables.

        Performs the chain rule differentiation from the element centroid quantities back through the maps,
        smooth Heaviside projection, and PDE filter. The final value is the gradient with respect to the free
        nodal design variables that the optimizer needs in order to update the design efficiently.

        Args:
            sensitivity_wrt_element_densities: A numpy array containing the sensitivity wrt element variables.

        Returns:
            A numpy array containing the completed gradient calculation with respect to free nodal design variables.
        """
        self.logger.debug("Apply filter to all sensitivity variables begin")
        sensitivity_to_nodes = self.__element_quadrature_points_to_nodal_field_map @ sensitivity_wrt_element_densities
        smooth_heaviside_chain_rule_derivative = \
            self.smooth_heaviside_projection.apply_chain_rule(sensitivity_to_nodes)
        sensitivity_to_nodes_with_periodicity = self.__all_to_free_matrix_map @ smooth_heaviside_chain_rule_derivative
        adjoint_inverse_applied_to_nodal_sensitivity_with_periodicity = \
            self.__factored_matrix_solve(sensitivity_to_nodes_with_periodicity)
        adjoint_inverse_applied_to_nodal_sensitivity = \
            self.__free_to_all_matrix_map @ adjoint_inverse_applied_to_nodal_sensitivity_with_periodicity
        result_to_elements = \
            self.__nodal_field_to_element_quadrature_points_map @ adjoint_inverse_applied_to_nodal_sensitivity
        result_scaled_by_area = self.__element_areas.ravel() * result_to_elements.ravel()
        all_nodal_sensitivity = self.__element_quadrature_points_to_nodal_field_map @ result_scaled_by_area
        free_nodal_sensitivity = self.__all_to_free_matrix_map @ all_nodal_sensitivity
        return free_nodal_sensitivity


    ###################################################################################################################
    ###################################################################################################################
    def map_from_free_variables_to_all_variables(self,
                                                 free_design_variables: np.ndarray):
        """Map from free nodal variables to all nodal variables."""
        return self.__free_to_all_matrix_map @ free_design_variables


    ###################################################################################################################
    ###################################################################################################################
    def map_from_all_variables_to_free_variables(self,
                                                 all_design_variables: np.ndarray):
        """From all nodal variables return only the free nodal variables."""
        return all_design_variables[self.__bottom_left_corner_and_interior_and_free_dof_indices]


    ###################################################################################################################
    ###################################################################################################################
    def update_logged_values(self,
                             logged_values: dict) -> dict:
        """Update the values of anything worth writing to a logfile at each optimization iteration.

        Takes in a dictionary of values to log and updates it with additional quantities. The previous quantities are
        retained in the dictionary and the new dictionary is returned.

        Args:
            logged_values: A dictionary of useful quantities to log at each optimization iteration.

        Returns:
            An updated dictionary of logged values.
        """
        updated_logged_values = self.smooth_heaviside_projection.update_logged_values(logged_values)
        return updated_logged_values



#######################################################################################################################
#######################################################################################################################
class SmoothHeavisideProjection:
    """Smooth Heaviside projection class for filtered variables.

    A smooth Heaviside projection class which handles not only the projection but the chain rule derivatives
    required for consistent sensitivities. Continuation on the projection parameter is also enabled via the user
    specified continuation function.

    .. math::
        \\rho(\\hat{\\rho}) = \\frac{\\tanh{\\beta \\eta} + \\tanh{\\beta \\left( \\hat{\\rho} - \\eta \\right)}}
            {\\tanh{\\beta \\eta} + \\tanh{\\beta \\left( 1 - \\eta \\right)}}

    Attributes:
        None
    """

    ###################################################################################################################
    ###################################################################################################################
    def __init__(self,
                 beta_parameter_continuation_function: Callable,
                 maximum_heaviside_projection_parameter: float):
        """Initialization of the smooth Heaviside projection object.

        Takes a continuation function for the \\beta parameter and the maximum allowable beta parameter. Internal
        attributes are initialized and \\beta is attempted to be set using the user-specified continuation function
        for an optimization iteration number of 0.

        Args:
            beta_parameter_continuation_function: A callable function taking the iteration number and
                returning the current value of the smooth Heaviside projection parameter, \\beta.
            maximum_heaviside_projection_parameter: A float that is the upper bound on the smooth Heaviside projection
                parameter computed in the RVE using the filter radius and the element size.

        Returns:
            None
        """
        self.logger = logging.getLogger("MaterialTopOpt.SmoothHeavisideProjection")
        self.__beta_parameter_continuation_function = beta_parameter_continuation_function
        self.__maximum_beta_parameter = maximum_heaviside_projection_parameter
        self.__beta = 1.0
        self.__eta = 0.5
        self.__set_beta(optimization_iteration_number=0)


    ###################################################################################################################
    ###################################################################################################################
    def __set_beta(self,
                   optimization_iteration_number: int = 0) -> None:
        """Update the current value of the projection parameter \\beta.

        Uses the user-specified continuation function to try and update the projection parameter \\beta.

        Args:
            optimization_iteration_number: An integer containing the current optimization iteration number.

        Returns:
            None
        """
        try:
            self.__beta = self.__beta_parameter_continuation_function(optimization_iteration_number)
        except Exception as exc:
            raise ValueError(("The user provided 'smooth Heaviside projection parameter continuation function' was not"
                              " a function which accepts an integer and returns a floating point value")) from exc
        assert isinstance(self.__beta, (float, int)), \
            "The user provided 'smooth Heaviside projection parameter continuation function' did not return a" + \
            f" floating point value. Returned '{self.__beta}' with type '{type(self.__beta)}'"
        assert 0.0 < self.__beta, "The user provided 'smooth Heaviside projection parameter continuation function'" + \
            f" did not return a value greater than 0. Returned '{self.__beta}'"
        if self.__beta > self.__maximum_beta_parameter:
            message = "The user provided 'smooth Heaviside projection parameter continuation function'"
            message += " returned a value greater than the maximum allowed based on the mesh size and filter radius."
            message += f" Setting parameter equal to maximum value '{self.__maximum_beta_parameter:0.3e}' rather than"
            message += f" returned value '{self.__beta:0.3e}'."
            self.logger.warning(message)
            self.__beta = self.__maximum_beta_parameter


    ###################################################################################################################
    ###################################################################################################################
    def apply(self,
              filtered_density_vector: np.ndarray,
              optimization_iteration_number: int = 0) -> np.ndarray:
        """Apply the projection.

        Takes the filtered variables, updates the value of the projection parameter, \\beta, and applied the function.

        Args:
            filtered_density_vector: A numpy array containing the filtered design variables.
            optimization_iteration_number: An integer containing the current optimization iteration number.

        Returns:
            A numpy array containing the projected and filtered design variables.
        """
        self.__set_beta(optimization_iteration_number=optimization_iteration_number)
        denominator = (np.tanh(self.__beta * self.__eta) + np.tanh(self.__beta * (1.0 - self.__eta)))
        numerator = (np.tanh(self.__beta * self.__eta) + np.tanh(self.__beta * (filtered_density_vector - self.__eta)))
        projected_filtered_densities = numerator / denominator
        self.__chain_rule_sensitivity_vector = \
            (self.__beta / denominator) * ( 1.0 - np.tanh(self.__beta * (filtered_density_vector - self.__eta))**2 )
        return projected_filtered_densities


    ###################################################################################################################
    ###################################################################################################################
    def apply_chain_rule(self,
                         sensitivity_vector: np.ndarray) -> np.ndarray:
        """Apply the chain rule derivative through the projection.

        Takes the gradient with respect to the projected nodal variables and returns the gradient with respect to
        the filtered nodal variables.

        Args:
            sensitivity_vector: A numpy array containing the gradient with respect to the projected nodal variables.

        Returns:
            A numpy array containing the gradient with respect to the filtered nodal variables.
        """
        return self.__chain_rule_sensitivity_vector.ravel() * sensitivity_vector.ravel()


    ###################################################################################################################
    ###################################################################################################################
    def update_logged_values(self,
                             logged_values: dict) -> dict:
        """Update the values of anything worth writing to a logfile at each optimization iteration.

        Takes in a dictionary of values to log and updates it with additional quantities. The previous quantities are
        retained in the dictionary and the new dictionary is returned.

        Args:
            logged_values: A dictionary of useful quantities to log at each optimization iteration.

        Returns:
            An updated dictionary of logged values.
        """
        projection_logged_values = {"Heaviside Projection Parameter": self.__beta}
        logged_values.update(projection_logged_values)
        return logged_values
