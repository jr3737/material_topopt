"""A module that contains the class RepresentativeVolumeElement2D.

The representative volume element contained in this module is intended only for use in
2D with the assumption of linear elasticity and order-one homogenization. Additionally,
the RVE is restricted to a square domain that is discretized with 3-node triangular
elements for simplicity.
"""
import os
import shutil
import logging
import numpy as np
import scipy as sp
from scipy.spatial import Delaunay
import meshio
import matplotlib.pyplot as plt

import src.material_topopt.utilities as utils
from src.material_topopt.periodic_density_filter import PeriodicDensityFilter


#######################################################################################################################
#######################################################################################################################
class RepresentativeVolumeElement2D:
    """Two-dimensional Representative Volume Element using only 3-node linear triangular elements.

    Performs linear elastic order-one homogenization of a square representative volume element in 2D constructed with
    3-node tetrahedral elements using periodic boundary conditions. The microstructure is described by a scalar
    density field with values between 0 and 1 that allow interpolation between two different materials with differing
    elastic modulus. The class also performs the sensitivity analysis of the coarse scale homogenized constitutive
    tensor components with respect to the design variables for optimization.

    Attributes:
        periodic_density_filter: A PeriodicDensityFilter instance to perform filtering of design variables.
    """

    ###################################################################################################################
    ###################################################################################################################
    def __init__(self,
                 representative_volume_element_parameters: dict):
        """Initialization of a representative volume element.

        Takes in a dictionary of parameters which are parsed and stored as internal attributes. A two-dimensional
        square RVE is created and meshed using a grid of 3-node triangular finite elements. Multiple quantities that
        are repeatedly used are precomputed in addition to initialization of the PDE filter.

        Args:
            representative_volume_element_parameters: A dictionary of representative volume element parameters.

        Returns:
            None
        """
        self.logger = logging.getLogger("MaterialTopOpt.RepresentativeVolumeElement2D")
        self.logger.debug("Initialization begin")
        self.__set_user_parameters(representative_volume_element_parameters)
        self.__space_dimension = 2
        self.__simp_exponent = 3.0

        self.__create_representative_volume_element_triangle_mesh(
            number_of_elements_along_each_edge=self.__number_of_elements_along_each_edge)
        self.__base_linear_elastic_constitutive_tensor = \
            utils.get_2d_isotropic_linear_elastic_4th_order_constitutive_tensor(elastic_modulus=1.0,
                                                                                poissons_ratio=self.__poissons_ratio)
        self.__precompute_fixed_numerical_quantities()

        if self.__density_filter_radius < 1.5 * self.__element_size:
            message = "The filter radius is smaller than 1.5 times the element size. " + \
                f"Please set the density filter radius equal to at least {1.5 * self.__element_size:0.5e}."
            self.logger.warning(message)

        self.logger.debug("Initialization end")

        maximum_heaviside_projection_parameter = \
            (2.0 * self.__density_filter_radius) / ((3.0**0.5) * self.__element_size)
        pde_filter_parameters = dict(element_areas=self.__element_areas,
                                     shape_function_gradients=self.__shape_function_gradients,
                                     shape_function_values=self.__shape_function_values,
                                     filter_radius=self.__density_filter_radius,
                                     element_connectivity=self.__element_connectivity,
                                     nodal_coordinates=self.__nodal_coordinates,
                                     interior_node_indices=self.__interior_node_indices,
                                     free_boundary_node_indices=self.__face_node_pairs[:, 0].ravel(),
                                     forced_boundary_node_indices=self.__face_node_pairs[:, 1].ravel(),
                                     bottom_left_corner_node_index=self.__bottom_left_node_index,
                                     forced_corner_node_indices=self.__forced_corner_node_indices)
        self.periodic_density_filter = PeriodicDensityFilter(pde_filter_parameters,
                                                             self.__smooth_heaviside_projection_continuation_function,
                                                             maximum_heaviside_projection_parameter)


    ###################################################################################################################
    ###################################################################################################################
    def __set_user_parameters(self,
                              representative_volume_element_parameters: dict):
        """Checks and sets the representative volume element parameters.

        Takes the user-supplied dictionary of RVE parameters and checks them prior to setting
        the corresponding attributes of the class.

        Args:
            representative_volume_element_parameters: User-specified dictionary containing the RVE problem parameters.

        Returns:
            None

        Raises:
            AssertionError: An error is thrown when parameters are not the correct type,
                necessary parameters are not specified, or files dont exist.
        """
        self.__number_of_elements_along_each_edge = utils.get_parameter_or_default(
            representative_volume_element_parameters, "number of elements along each edge", int, bounds=(2, None),
            default_value=50, required=True)
        self.__soft_material_elastic_modulus = utils.get_parameter_or_default(
            representative_volume_element_parameters, "soft material elastic modulus", float, bounds=(1.0e-9, None),
            default_value=1.0, required=True)
        self.__stiff_material_elastic_modulus = utils.get_parameter_or_default(
            representative_volume_element_parameters, "stiff material elastic modulus", float,
            bounds=(self.__soft_material_elastic_modulus, None), default_value=1.0e3, required=True,
            additional_message="The stiff material elastic modulus cannot be smaller than the soft material.")
        self.__poissons_ratio = utils.get_parameter_or_default(
            representative_volume_element_parameters, "poissons ratio", float, bounds=(-0.5, 0.45),
            default_value=0.33, required=True)
        self.__density_filter_radius = utils.get_parameter_or_default(
            representative_volume_element_parameters, "density filter radius", float, bounds=(0.0, 1.0),
            default_value=0.05, required=True)
        self.__output_directory_path = utils.get_parameter_or_default(
            representative_volume_element_parameters, "output directory path", str, required=False,
            default_value=os.path.join(os.getcwd(), "output"))
        if not os.path.exists(self.__output_directory_path):
            os.mkdir(self.__output_directory_path)

        key = "SIMP exponent continuation function"
        self.__simp_exponent_continuation_function = \
            representative_volume_element_parameters[key] if key in representative_volume_element_parameters.keys() \
            else lambda iteration_number: 3.0 if iteration_number < 50 else 4.0

        key = "smooth Heaviside projection continuation function"
        self.__smooth_heaviside_projection_continuation_function = \
            representative_volume_element_parameters[key] if key in representative_volume_element_parameters.keys() \
            else lambda iteration_number: 1.0

        key = "design variable initialization function"
        self.__design_variable_initialization_function = \
            representative_volume_element_parameters[key] if key in representative_volume_element_parameters.keys() \
            else None


    ###################################################################################################################
    ###################################################################################################################
    def __create_representative_volume_element_triangle_mesh(self,
                                                             number_of_elements_along_each_edge: int = 10):
        """Create the representative volume element's 3-node triangle mesh.

        Creates a square RVE and computes a 3-node triangle discretization, including the nodal coordinates and
        element connectivity arrays.

        Args:
            number_of_elements_along_each_edge: An integer representing the number of elements along one RVE edge.

        Returns:
            None
        """
        self.logger.debug("Create RVE mesh begin")
        domain_width  = 1.0
        domain_height = 1.0
        x = np.linspace(0.0, domain_width,  number_of_elements_along_each_edge + 1)
        y = np.linspace(0.0, domain_height, number_of_elements_along_each_edge + 1)
        self.__element_size = max(x[1] - x[0], y[1] - y[0])
        x_grid, y_grid = np.meshgrid(x, y)
        square_points = np.zeros((x.size * y.size, self.__space_dimension))
        square_points[:, 0] = x_grid.ravel()
        square_points[:, 1] = y_grid.ravel()
        triangulation = Delaunay(square_points)
        nodal_coordinates = triangulation.points
        element_connectivity = triangulation.simplices
        smallest_node_index = np.amin(element_connectivity)
        assert smallest_node_index == 0
        self.__nodal_coordinates = nodal_coordinates
        self.__number_of_nodes = nodal_coordinates.shape[0]
        self.__element_connectivity = element_connectivity
        self.__number_of_elements, self.__number_of_nodes_per_element = element_connectivity.shape
        self.__rve_area = domain_width * domain_height


    ###################################################################################################################
    ###################################################################################################################
    def __compute_node_index_pairs_on_opposite_faces(self):
        """Computes the index pairs for applying periodic boundary conditions.

        Computes the index pairs of nodes on opposite faces in addition to storing the indices of the nodes
        at the corners of the RVE, all for imposing periodic boundary conditions later.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug("Compute node index pairs on opposite faces")
        smallest_x_coordinate, smallest_y_coordinate = np.amin(self.__nodal_coordinates, axis=0)
        largest_x_coordinate, largest_y_coordinate   = np.amax(self.__nodal_coordinates, axis=0)

        geometric_tolerance = 1.0e-5

        x_coordinates = self.__nodal_coordinates[:, 0].ravel()
        y_coordinates = self.__nodal_coordinates[:, 1].ravel()

        left_face_node_mask   = x_coordinates < (smallest_x_coordinate + geometric_tolerance)
        right_face_node_mask  = x_coordinates > (largest_x_coordinate  - geometric_tolerance)
        bottom_face_node_mask = y_coordinates < (smallest_y_coordinate + geometric_tolerance)
        top_face_node_mask    = y_coordinates > (largest_y_coordinate  - geometric_tolerance)

        left_face_node_indices   = \
            np.argwhere(  left_face_node_mask & ~bottom_face_node_mask &   ~top_face_node_mask).ravel()
        right_face_node_indices  = \
            np.argwhere( right_face_node_mask & ~bottom_face_node_mask &   ~top_face_node_mask).ravel()
        bottom_face_node_indices = \
            np.argwhere(bottom_face_node_mask &   ~left_face_node_mask & ~right_face_node_mask).ravel()
        top_face_node_indices    = \
            np.argwhere(   top_face_node_mask &   ~left_face_node_mask & ~right_face_node_mask).ravel()

        # if not np.allclose(y_coordinates[left_face_node_indices],
        #                    y_coordinates[right_face_node_indices], atol=geometric_tolerance):
        #     sorted_left_indices_ascending = np.argsort(y_coordinates[left_face_node_indices])
        #     left_face_node_indices = left_face_node_indices[sorted_left_indices_ascending]

        #     sorted_right_indices_ascending = np.argsort(y_coordinates[right_face_node_indices])
        #     right_face_node_indices = right_face_node_indices[sorted_right_indices_ascending]

        #     assert np.allclose(y_coordinates[left_face_node_indices],
        #                        y_coordinates[right_face_node_indices], atol=geometric_tolerance)

        # if not np.allclose(x_coordinates[bottom_face_node_indices],
        #                    x_coordinates[top_face_node_indices], atol=geometric_tolerance):
        #     sorted_bottom_indices_ascending = np.argsort(x_coordinates[bottom_face_node_indices])
        #     bottom_face_node_indices = bottom_face_node_indices[sorted_bottom_indices_ascending]

        #     sorted_top_indices_ascending = np.argsort(x_coordinates[top_face_node_indices])
        #     top_face_node_indices = top_face_node_indices[sorted_top_indices_ascending]

        #     assert np.allclose(x_coordinates[bottom_face_node_indices],
        #                        x_coordinates[top_face_node_indices], atol=geometric_tolerance)

        all_interior_nodes_mask = \
            ~(left_face_node_mask | right_face_node_mask | bottom_face_node_mask | top_face_node_mask)
        interior_node_indices   = np.argwhere(all_interior_nodes_mask).ravel()

        bottom_left_node_index  = np.argwhere( left_face_node_mask & bottom_face_node_mask).ravel()
        bottom_right_node_index = np.argwhere(right_face_node_mask & bottom_face_node_mask).ravel()
        top_left_node_index     = np.argwhere( left_face_node_mask &    top_face_node_mask).ravel()
        top_right_node_index    = np.argwhere(right_face_node_mask &    top_face_node_mask).ravel()
        corner_node_indices = np.concatenate(
            [bottom_left_node_index, bottom_right_node_index, top_left_node_index, top_right_node_index])

        number_of_pairs = left_face_node_indices.size + bottom_face_node_indices.size
        face_node_pairs = np.empty((number_of_pairs, 2), dtype=int)
        i = left_face_node_indices.size
        face_node_pairs[:i, 0] = left_face_node_indices
        face_node_pairs[:i, 1] = right_face_node_indices
        face_node_pairs[i:, 0] = bottom_face_node_indices
        face_node_pairs[i:, 1] = top_face_node_indices

        self.__face_node_pairs = face_node_pairs
        self.__interior_node_indices = interior_node_indices
        self.__corner_node_indices = corner_node_indices

        self.__bottom_left_node_index = bottom_left_node_index
        self.__forced_corner_node_indices = np.concatenate(
            [bottom_right_node_index, top_left_node_index, top_right_node_index])


    ###################################################################################################################
    ###################################################################################################################
    def __compute_dof_index_vectors_and_sparse_matrix_maps(self):
        """Compute vectors of indices and sparse matrix maps for condensing periodic boundary conditions.

        Computes numerous vectors of nodal dof indices that are useful for dealing with the periodic boundary
        conditions. Additionally, sparse matrix maps are constructed for efficient condensation of dependent
        degrees of freedom from vectors and matrices.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug("Compute dof index for periodic boundary conditions")
        interior_node_dof_indices = self.__nodal_dof_indices[self.__interior_node_indices, :].ravel()
        free_boundary_node_indices = self.__face_node_pairs[:, 0].ravel()
        forced_boundary_node_indices = self.__face_node_pairs[:, 1].ravel()
        free_boundary_dofs = self.__nodal_dof_indices[free_boundary_node_indices, :].ravel()
        forced_boundary_dofs = self.__nodal_dof_indices[forced_boundary_node_indices, :].ravel()
        self.__corner_node_dof_indices = self.__nodal_dof_indices[self.__corner_node_indices, :].ravel()
        self.__bottom_left_corner_node_dof_indices = self.__nodal_dof_indices[self.__bottom_left_node_index, :].ravel()

        # Create the dof map matrices
        number_of_global_dofs = self.__number_of_nodes * self.__space_dimension
        data = np.ones((number_of_global_dofs, ), dtype=float)
        cols = np.zeros(data.shape, dtype=int)
        rows = np.zeros(data.shape, dtype=int)
        k = self.__bottom_left_corner_node_dof_indices.size
        reduced_bottom_left_corner_dof_indices = np.arange(k, dtype=int)
        i = k + interior_node_dof_indices.size
        reduced_interior_dof_indices = np.arange(k, i, dtype=int)
        j = i + free_boundary_dofs.size
        reduced_free_boundary_dof_indices = np.arange(i, j, dtype=int)
        assert (self.__corner_node_dof_indices.size + interior_node_dof_indices.size +
                free_boundary_dofs.size + forced_boundary_dofs.size) == number_of_global_dofs
        # corner nodes
        rows[self.__corner_node_dof_indices] = self.__corner_node_dof_indices
        cols[self.__corner_node_dof_indices] = np.concatenate(
            [reduced_bottom_left_corner_dof_indices, reduced_bottom_left_corner_dof_indices,
             reduced_bottom_left_corner_dof_indices, reduced_bottom_left_corner_dof_indices])
        # interior nodes
        rows[interior_node_dof_indices] = interior_node_dof_indices
        cols[interior_node_dof_indices] = reduced_interior_dof_indices
        # free boundary nodes
        rows[free_boundary_dofs] = free_boundary_dofs
        cols[free_boundary_dofs] = reduced_free_boundary_dof_indices
        # forced boundary nodes
        rows[forced_boundary_dofs] = forced_boundary_dofs
        cols[forced_boundary_dofs] = reduced_free_boundary_dof_indices
        # create the matrix map
        self.__free_to_all_matrix_map = sp.sparse.coo_matrix((data, (rows, cols))).tocsr()
        self.__all_to_free_matrix_map = self.__free_to_all_matrix_map.transpose()


    ###################################################################################################################
    ###################################################################################################################
    def __precompute_fixed_numerical_quantities(self):
        """Precompute a number of numerical quantities for later use.

        Computes and stores a large quantity of arrays for repeated use throughout the optimization procedure.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug("Precompute fixed numerical quantities begin")
        self.__compute_node_index_pairs_on_opposite_faces()
        self.__shape_function_values, self.__shape_function_gradients, self.__element_areas = \
            utils.get_triangle_shape_functions_and_element_areas(self.__nodal_coordinates, self.__element_connectivity)
        self.__virtual_displacement_symmetric_gradients = \
            utils.get_virtual_displacement_symmetric_gradients(self.__shape_function_gradients)
        self.__base_stiffness_matrix_elementwise = \
            utils.get_2d_linear_elasticity_problem_stiffness_matrix_elementwise(
                self.__base_linear_elastic_constitutive_tensor,
                self.__virtual_displacement_symmetric_gradients,
                self.__element_areas)
        self.__nodal_dof_indices, matrix_row_indices, matrix_column_indices, \
            self.__residual_vector_indices_elementwise = \
            utils.get_2d_linear_elasticity_problem_dof_indices(self.__nodal_coordinates, self.__element_connectivity)
        self.__matrix_row_indices = matrix_row_indices
        self.__matrix_col_indices = matrix_column_indices
        self.__compute_dof_index_vectors_and_sparse_matrix_maps()
        self.logger.debug("Precompute fixed numerical quantities end")


    ###################################################################################################################
    ###################################################################################################################
    def __assemble_rve_stiffness_matrix_and_rhs_vectors_and_solve(self, density_variables_elementwise: np.ndarray):
        """Linear system assembly and solution for the influence functions.

        Given the current density variables, the linear systems are assembled and solved in order to obtain
        the influence functions required to compute the homogenized elastic properties. The solution is also written
        to Paraview VTK files in this method.

        Args:
            density_variables_elementwise: A numpy array of density values at each element's quadrature point.

        Returns:
            None
        """
        self.logger.debug("Assemble RVE matrix and RHS vectors")
        simp_penalization = density_variables_elementwise**self.__simp_exponent
        one_minus_simp_penalization = 1.0 - simp_penalization
        penalized_elastic_modulus = self.__stiff_material_elastic_modulus * simp_penalization + \
            self.__soft_material_elastic_modulus * one_minus_simp_penalization

        # Form the stiffness matrix
        penalized_stiffness_matrix_elementwise = \
            np.einsum('emn,e->emn', self.__base_stiffness_matrix_elementwise, penalized_elastic_modulus)
        data = penalized_stiffness_matrix_elementwise.ravel()
        stiffness_matrix_uncondensed = \
            sp.sparse.coo_matrix((data, (self.__matrix_row_indices, self.__matrix_col_indices))).tocsc()

        # Form the right hand side vectors
        row_indices = self.__residual_vector_indices_elementwise.ravel()
        col_indices = np.zeros(row_indices.shape, dtype=int)
        temporary_integral = np.einsum('ekij,e...,e...->ekij',
                                       self.__virtual_displacement_symmetric_gradients,
                                       penalized_elastic_modulus,
                                       self.__element_areas)
        # H11
        rhs_vector_for_H11_elementwise = np.einsum('ekij,ij->ek',
                                                   temporary_integral,
                                                   self.__base_linear_elastic_constitutive_tensor[:, :, 0, 0])
        rhs_vector_for_H11_uncondensed = \
            sp.sparse.coo_matrix((rhs_vector_for_H11_elementwise.ravel(), (row_indices, col_indices))).toarray().ravel()
        rhs_vector_for_H11_uncondensed *= -1.0
        # H22
        rhs_vector_for_H22_elementwise = np.einsum('ekij,ij->ek',
                                                   temporary_integral,
                                                   self.__base_linear_elastic_constitutive_tensor[:, :, 1, 1])
        rhs_vector_for_H22_uncondensed = \
            sp.sparse.coo_matrix((rhs_vector_for_H22_elementwise.ravel(), (row_indices, col_indices))).toarray().ravel()
        rhs_vector_for_H22_uncondensed *= -1.0
        # H12
        rhs_vector_for_H12_elementwise = np.einsum('ekij,ij->ek',
                                                   temporary_integral,
                                                   self.__base_linear_elastic_constitutive_tensor[:, :, 0, 1])
        rhs_vector_for_H12_uncondensed = \
            sp.sparse.coo_matrix((rhs_vector_for_H12_elementwise.ravel(), (row_indices, col_indices))).toarray().ravel()
        rhs_vector_for_H12_uncondensed *= -1.0

        # Make the RVE corner dofs equal to 0 prior to condensing the periodic BCs out of the equations
        for corner_dof_index in self.__corner_node_dof_indices:
            row = stiffness_matrix_uncondensed.getrow(corner_dof_index)
            col = stiffness_matrix_uncondensed.getcol(corner_dof_index)
            row.data[:] = 0.0
            col.data[:] = 0.0
            stiffness_matrix_uncondensed[corner_dof_index, :] = row
            stiffness_matrix_uncondensed[:, corner_dof_index] = col
        i = self.__bottom_left_corner_node_dof_indices[0]
        stiffness_matrix_uncondensed[i, i] = 1.0
        i = self.__bottom_left_corner_node_dof_indices[1]
        stiffness_matrix_uncondensed[i, i] = 1.0
        rhs_vector_for_H11_uncondensed[self.__corner_node_dof_indices] = 0.0
        rhs_vector_for_H22_uncondensed[self.__corner_node_dof_indices] = 0.0
        rhs_vector_for_H12_uncondensed[self.__corner_node_dof_indices] = 0.0

        stiffness_matrix_condensed = self.__all_to_free_matrix_map @ \
            (stiffness_matrix_uncondensed @ self.__free_to_all_matrix_map)
        rhs_vector_for_H11_condensed = self.__all_to_free_matrix_map @ rhs_vector_for_H11_uncondensed
        rhs_vector_for_H22_condensed = self.__all_to_free_matrix_map @ rhs_vector_for_H22_uncondensed
        rhs_vector_for_H12_condensed = self.__all_to_free_matrix_map @ rhs_vector_for_H12_uncondensed

        self.logger.debug("Solve systems of equations")

        # Solve the condensed systems of equations
        stiffness_matrix_condensed_csc = stiffness_matrix_condensed.tocsc()
        factored_matrix_solve = utils.factorized(stiffness_matrix_condensed_csc)
        H11_condensed = factored_matrix_solve(rhs_vector_for_H11_condensed)
        H22_condensed = factored_matrix_solve(rhs_vector_for_H22_condensed)
        H12_condensed = factored_matrix_solve(rhs_vector_for_H12_condensed)

        # Construct the uncondensed solution vectors
        H11_uncondensed = self.__free_to_all_matrix_map @ H11_condensed
        H22_uncondensed = self.__free_to_all_matrix_map @ H22_condensed
        H12_uncondensed = self.__free_to_all_matrix_map @ H12_condensed

        self.logger.debug("Paraview VTK output")

        new_shape = (self.__number_of_nodes, self.__space_dimension)
        self.__H11_nodewise = H11_uncondensed.reshape(new_shape)
        self.__H22_nodewise = H22_uncondensed.reshape(new_shape)
        self.__H12_nodewise = H12_uncondensed.reshape(new_shape)

        # VTK Output
        points_3dim = np.zeros((self.__number_of_nodes, 3))
        points_3dim[:, :2] = self.__nodal_coordinates
        H11_3dim = np.zeros((self.__number_of_nodes, 3))
        H11_3dim[:, :2] = self.__H11_nodewise
        H22_3dim = np.zeros((self.__number_of_nodes, 3))
        H22_3dim[:, :2] = self.__H22_nodewise
        H12_3dim = np.zeros((self.__number_of_nodes, 3))
        H12_3dim[:, :2] = self.__H12_nodewise
        point_data = {"H11": H11_3dim, "H22": H22_3dim, "H12": H12_3dim, "NodalDensity": self.filtered_nodal_densities}
        cell_data = {"ElementDensity": [density_variables_elementwise]}
        cell_blocks = [meshio.CellBlock('triangle', self.__element_connectivity)]
        my_mesh = meshio.Mesh(points_3dim, cell_blocks, point_data=point_data, cell_data=cell_data)
        rve_output_filepath = os.path.join(self.__output_directory_path,
                                           f"rve.{self.__optimization_iteration_number:04d}.vtk")
        my_mesh.write(rve_output_filepath)


    ###################################################################################################################
    ###################################################################################################################
    def __compute_coarse_scale_constitutive_tensor(self,
                                                   density_variables_elementwise: np.ndarray):
        """Computation of the coarse scale elastic properties.

        The homogenized constitutive tensor for use at the coarse scale is computed using the now known influence
        functions for the given microstructure, represented by the density field. The derivative of the 6 independent
        components (in 2D) of this constitutive tensor is also computed in this method and stored as an attribute.

        Args:
            density_variables_elementwise: A numpy array of density values at each element's quadrature point.

        Returns:
            None
        """
        self.logger.debug("Compute coarse scale constitutive tensor")
        simp_penalization = density_variables_elementwise**self.__simp_exponent
        one_minus_simp_penalization = 1.0 - simp_penalization
        penalized_elastic_modulus = self.__stiff_material_elastic_modulus * simp_penalization + \
            self.__soft_material_elastic_modulus * one_minus_simp_penalization

        number_of_dofs_per_element = self.__number_of_nodes_per_element * self.__space_dimension
        new_shape = (self.__number_of_elements, number_of_dofs_per_element)
        H11_elementwise = self.__H11_nodewise[self.__element_connectivity, :].reshape(new_shape)
        H22_elementwise = self.__H22_nodewise[self.__element_connectivity, :].reshape(new_shape)
        H12_elementwise = self.__H12_nodewise[self.__element_connectivity, :].reshape(new_shape)

        H11_symmetric_gradients = np.einsum('ekij,ek->eij',
            self.__virtual_displacement_symmetric_gradients, H11_elementwise)
        H22_symmetric_gradients = np.einsum('ekij,ek->eij',
            self.__virtual_displacement_symmetric_gradients, H22_elementwise)
        H12_symmetric_gradients = np.einsum('ekij,ek->eij',
            self.__virtual_displacement_symmetric_gradients, H12_elementwise)

        integrated_elastic_modulus = np.einsum('e,e->', penalized_elastic_modulus, self.__element_areas)

        integral_H11 = np.einsum('e,ijkl,ekl,e->ij', penalized_elastic_modulus,
            self.__base_linear_elastic_constitutive_tensor, H11_symmetric_gradients, self.__element_areas,
            optimize=True)
        integral_H22 = np.einsum('e,ijkl,ekl,e->ij', penalized_elastic_modulus,
            self.__base_linear_elastic_constitutive_tensor, H22_symmetric_gradients, self.__element_areas,
            optimize=True)
        integral_H12 = np.einsum('e,ijkl,ekl,e->ij', penalized_elastic_modulus,
            self.__base_linear_elastic_constitutive_tensor, H12_symmetric_gradients, self.__element_areas,
            optimize=True)

        self.__coarse_scale_constitutive_tensor = np.zeros((2, 2, 2, 2))
        H_integral_map = {(0, 0): integral_H11, (0, 1): integral_H12, (1, 1): integral_H22}
        independent_tensor_components, component_symmetries = \
            utils.get_list_of_2d_constitutive_tensor_independent_components_and_symmetries()
        for component_index, (i, j, m, n) in enumerate(independent_tensor_components):
            self.__coarse_scale_constitutive_tensor[i, j, m, n] = \
                integrated_elastic_modulus * self.__base_linear_elastic_constitutive_tensor[i, j, m, n] + \
                    H_integral_map[(m, n)][i, j]

            for component_symmetry in component_symmetries[component_index]:
                a, b, c, d = component_symmetry
                self.__coarse_scale_constitutive_tensor[a, b, c, d] = \
                    self.__coarse_scale_constitutive_tensor[i, j, m, n]

        self.__coarse_scale_constitutive_tensor /= self.__rve_area

        self.logger.debug("Compute coarse scale constitutive tensor derivatives")

        simp_penalization_derivative = \
            self.__simp_exponent * (density_variables_elementwise**(self.__simp_exponent - 1.0))
        penalized_elastic_modulus_derivative = \
            (self.__stiff_material_elastic_modulus - self.__soft_material_elastic_modulus)*simp_penalization_derivative

        d_f11_d_density_elementwise = np.einsum('eaij,e,ij,e->ea',
                                                self.__virtual_displacement_symmetric_gradients,
                                                penalized_elastic_modulus_derivative,
                                                self.__base_linear_elastic_constitutive_tensor[:, :, 0, 0],
                                                self.__element_areas, optimize=True)
        d_R11_d_density_elementwise = np.einsum('eaij,e,ijkl,ekl,e->ea',
                                                self.__virtual_displacement_symmetric_gradients,
                                                penalized_elastic_modulus_derivative,
                                                self.__base_linear_elastic_constitutive_tensor,
                                                H11_symmetric_gradients,
                                                self.__element_areas, optimize=True)
        d_R11_d_density_elementwise += d_f11_d_density_elementwise

        d_f22_d_density_elementwise = np.einsum('eaij,e,ij,e->ea',
                                                self.__virtual_displacement_symmetric_gradients,
                                                penalized_elastic_modulus_derivative,
                                                self.__base_linear_elastic_constitutive_tensor[:, :, 1, 1],
                                                self.__element_areas, optimize=True)
        d_R22_d_density_elementwise = np.einsum('eaij,e,ijkl,ekl,e->ea',
                                                self.__virtual_displacement_symmetric_gradients,
                                                penalized_elastic_modulus_derivative,
                                                self.__base_linear_elastic_constitutive_tensor,
                                                H22_symmetric_gradients,
                                                self.__element_areas, optimize=True)
        d_R22_d_density_elementwise += d_f22_d_density_elementwise

        d_f12_d_density_elementwise = np.einsum('eaij,e,ij,e->ea',
                                                self.__virtual_displacement_symmetric_gradients,
                                                penalized_elastic_modulus_derivative,
                                                self.__base_linear_elastic_constitutive_tensor[:, :, 0, 1],
                                                self.__element_areas, optimize=True)
        d_R12_d_density_elementwise = np.einsum('eaij,e,ijkl,ekl,e->ea',
                                                self.__virtual_displacement_symmetric_gradients,
                                                penalized_elastic_modulus_derivative,
                                                self.__base_linear_elastic_constitutive_tensor,
                                                H12_symmetric_gradients,
                                                self.__element_areas, optimize=True)
        d_R12_d_density_elementwise += d_f12_d_density_elementwise

        self.__d_L_coarse_1111_d_density_elementwise = \
            self.__assemble_coarse_scale_constitutive_tensor_sensitivity_elementwise(
                penalized_elastic_modulus_derivative, (0, 0, 0, 0), H11_symmetric_gradients,
                d_R11_d_density_elementwise, H11_elementwise)

        self.__d_L_coarse_1122_d_density_elementwise = \
            self.__assemble_coarse_scale_constitutive_tensor_sensitivity_elementwise(
                penalized_elastic_modulus_derivative, (0, 0, 1, 1), H22_symmetric_gradients,
                d_R22_d_density_elementwise, H11_elementwise)

        self.__d_L_coarse_1112_d_density_elementwise = \
            self.__assemble_coarse_scale_constitutive_tensor_sensitivity_elementwise(
                penalized_elastic_modulus_derivative, (0, 0, 0, 1), H12_symmetric_gradients,
                d_R12_d_density_elementwise, H11_elementwise)

        self.__d_L_coarse_2222_d_density_elementwise = \
            self.__assemble_coarse_scale_constitutive_tensor_sensitivity_elementwise(
                penalized_elastic_modulus_derivative, (1, 1, 1, 1), H22_symmetric_gradients,
                d_R22_d_density_elementwise, H22_elementwise)

        self.__d_L_coarse_2212_d_density_elementwise = \
            self.__assemble_coarse_scale_constitutive_tensor_sensitivity_elementwise(
                penalized_elastic_modulus_derivative, (1, 1, 0, 1), H12_symmetric_gradients,
                d_R12_d_density_elementwise, H22_elementwise)

        self.__d_L_coarse_1212_d_density_elementwise = \
            self.__assemble_coarse_scale_constitutive_tensor_sensitivity_elementwise(
                penalized_elastic_modulus_derivative, (0, 1, 0, 1), H12_symmetric_gradients,
                d_R12_d_density_elementwise, H12_elementwise)


    ###################################################################################################################
    ###################################################################################################################
    def __assemble_coarse_scale_constitutive_tensor_sensitivity_elementwise(self,
                                                                            d_elastic_modulus_d_density: np.ndarray,
                                                                            L_coarse_components: tuple,
                                                                            H_symmetric_gradients: np.ndarray,
                                                                            d_R_d_density_elementwise: np.ndarray,
                                                                            H_elementwise: np.ndarray):
        """Helper function for assembling the homogenized constitutive tensor derivative wrt the density variables.

        Args:
            d_elastic_modulus_d_density: A numpy array containing the derivative of each element's elastic modulus
                with respect to the corresponding element's density.
            L_coarse_components: A tuple containing the relevant integer components of the constitutive tensor.
            H_symmetric_gradients: A numpy array containing the symmetric gradients of the relevant influence function.
            d_R_d_density_elementwise: A numpy array containing the derivative of the residual vector with respect
                to each element's density.
            H_elementwise: A numpy array containing the values of the relevant influence function, elementwise.

        Returns:
            A numpy array containing the sensitivity of the constitutive tensor component with respect to the element
                density variables.
        """
        i, j, k, l = L_coarse_components
        constitutive_tensor_component = self.__base_linear_elastic_constitutive_tensor[i, j, k, l]
        partial_wrt_density_1 = np.einsum('e,e->e', d_elastic_modulus_d_density, self.__element_areas)
        partial_wrt_density_1 *= constitutive_tensor_component
        partial_wrt_density_2 = np.einsum('pq,epq,e,e->e', self.__base_linear_elastic_constitutive_tensor[i, j, :, :],
                                          H_symmetric_gradients, d_elastic_modulus_d_density, self.__element_areas,
                                          optimize=True)
        partial_wrt_density = partial_wrt_density_1 + partial_wrt_density_2
        #
        adjoint_contribution = np.einsum('ei,ei->e', H_elementwise, d_R_d_density_elementwise)
        #
        sensitivity_wrt_element_density = partial_wrt_density + adjoint_contribution
        sensitivity_wrt_element_density *= (1.0 / self.__rve_area)
        return sensitivity_wrt_element_density


    ###################################################################################################################
    ###################################################################################################################
    def get_initial_free_unfiltered_nodal_design_variables(self):
        """Initialize the nodal design variables.

        Ideally, a user-defined function is used to initialize the design variables based on the coordinates of the
        nodes. The design variables must be between 0 and 1. If no function is provided, then the design variables are
        randomly initialized to values between 0.1 and 0.9.

        Args:
            None

        Returns:
            A numpy array of the free unfiltered nodal design variables for the optimizer to begin with.
        """
        x_coordinates = self.__nodal_coordinates[:, 0].ravel()
        y_coordinates = self.__nodal_coordinates[:, 1].ravel()

        if self.__design_variable_initialization_function is not None:
            initial_design_variables = None
            try:
                initial_design_variables = self.__design_variable_initialization_function(x_coordinates, y_coordinates)
            except Exception as exc:
                raise ValueError("The user supplied function for initializing the design variables threw an error " + \
                    "when passed the arrays of x and y coordinates.") from exc
            assert isinstance(initial_design_variables, np.ndarray), \
                "The user supplied function for initializing the design variables did not return a numpy array"
            assert initial_design_variables.size == x_coordinates.size, \
                "The user supplied function for initializing the design variables did not return a numpy array " + \
                    "of the same size as the input coordinates"
            assert np.all((0.0 <= initial_design_variables) & (initial_design_variables <= 1.0)), \
                "The user supplied function for initializing the design variables did not return an array of " + \
                    "values between 0 and 1"
            free_unfiltered_nodal_design_variables = \
                self.periodic_density_filter.map_from_all_variables_to_free_variables(initial_design_variables)
            return free_unfiltered_nodal_design_variables

        message = ("User did not specify the 'design variable initialization function' so the design variables "
            "will be randomly initialized. Analysis may not produce expected results.")
        self.logger.warning(message)

        all_unfiltered_nodal_design_variables = np.random.uniform(0.1, 0.9, self.__number_of_nodes)
        free_unfiltered_nodal_design_variables = \
            self.periodic_density_filter.map_from_all_variables_to_free_variables(all_unfiltered_nodal_design_variables)
        return free_unfiltered_nodal_design_variables


    ###################################################################################################################
    ###################################################################################################################
    def run_forward_analysis(self,
                             free_nodal_design_variables: np.ndarray,
                             optimization_iteration_number: int = 0):
        """Run the forward analysis given the current design variables.

        The free nodal design variables are first filtered, projected, and then evaluated at the quadrature point of
        each element. Using this array of element densities RVE subproblem is solved and the coarse scale constitutive
        properties are computed.

        Args:
            free_nodal_design_variables: A numpy array of free nodal design variables.
            optimization_iteration_number: An integer containing the current optimization iteration number.

        Returns:
            None
        """
        candidate_simp_exponent = self.__simp_exponent_continuation_function(optimization_iteration_number)
        assert isinstance(candidate_simp_exponent, (float, int)), \
            "User supplied SIMP exponent continuation function did not return a floating point number or " + \
                f"an integer. Returned value = {candidate_simp_exponent}"
        assert 1.0 <= candidate_simp_exponent < 5.0, \
            "User supplied SIMP exponent continuation function did not return a value, x, with 1 <= x < 5.\n" + \
            f"Returned value = '{candidate_simp_exponent}'. Please correct your simp_exponent_continuation_function."
        self.__simp_exponent = candidate_simp_exponent
        #
        self.__optimization_iteration_number = optimization_iteration_number
        density_variables_elementwise, self.filtered_nodal_densities = \
            self.periodic_density_filter.apply_filter_to_nodal_design_variables_and_return_element_densities(
                free_nodal_design_variables,
                optimization_iteration_number=optimization_iteration_number
                )
        self.__assemble_rve_stiffness_matrix_and_rhs_vectors_and_solve(density_variables_elementwise)
        self.__compute_coarse_scale_constitutive_tensor(density_variables_elementwise)
        self.__volume_fraction = np.inner(density_variables_elementwise, self.__element_areas) / self.__rve_area


    ###################################################################################################################
    ###################################################################################################################
    def get_coarse_scale_constitutive_tensor(self):
        """Returns the previously computed coarse scale constitutive tensor."""
        return self.__coarse_scale_constitutive_tensor


    ###################################################################################################################
    ###################################################################################################################
    def get_macroscale_compliance_sensitivity(self,
                                              d_macroscale_compliance_d_coarse_scale_constitutive_tensor: np.ndarray):
        """Computed the macroscale compliance sensitivity.

        Given the derivative of the macroscale compliance with respect to the independent components of the coarse
        scale constitutive tensor, this function completes the chain rule derivative to obtain the gradient of the
        macroscale compliance with respect to the nodal design variables of the RVE.

        Args:
            d_macroscale_compliance_d_coarse_scale_constitutive_tensor: A numpy array containing the derivatives of
                the macroscale compliance with respect to the 6 (in 2D) independent components of the coarse scale
                constitutive tensor.

        Returns:
            A numpy array containing the completed gradient calculation.
        """
        macroscale_compliance_sensitivity  = d_macroscale_compliance_d_coarse_scale_constitutive_tensor[0] * \
            self.__d_L_coarse_1111_d_density_elementwise
        macroscale_compliance_sensitivity += d_macroscale_compliance_d_coarse_scale_constitutive_tensor[1] * \
            self.__d_L_coarse_1122_d_density_elementwise
        macroscale_compliance_sensitivity += d_macroscale_compliance_d_coarse_scale_constitutive_tensor[2] * \
            self.__d_L_coarse_1112_d_density_elementwise
        macroscale_compliance_sensitivity += d_macroscale_compliance_d_coarse_scale_constitutive_tensor[3] * \
            self.__d_L_coarse_2222_d_density_elementwise
        macroscale_compliance_sensitivity += d_macroscale_compliance_d_coarse_scale_constitutive_tensor[4] * \
            self.__d_L_coarse_2212_d_density_elementwise
        macroscale_compliance_sensitivity += d_macroscale_compliance_d_coarse_scale_constitutive_tensor[5] * \
            self.__d_L_coarse_1212_d_density_elementwise
        macroscale_compliance_sensitivity_wrt_design_variables = \
            self.periodic_density_filter.apply_sensitivity_chain_rule_derivative(macroscale_compliance_sensitivity)
        return macroscale_compliance_sensitivity_wrt_design_variables


    ###################################################################################################################
    ###################################################################################################################
    def get_volume_fraction_value_and_sensitivity(self):
        """Computes and returns the volume fraction of stiff material and its gradient.

        Args:
            None

        Returns:
            A tuple (a, b) where 'a' is the volume fraction of stiff material as a float, and 'b' is a numpy array
                containing the gradient of the volume fraction with respect to the nodal design variables.
        """
        volume_fraction_sensitivity_elementwise = self.__element_areas / self.__rve_area
        volume_fraction_sensitivity = \
            self.periodic_density_filter.apply_sensitivity_chain_rule_derivative(
                volume_fraction_sensitivity_elementwise
                )
        return self.__volume_fraction, volume_fraction_sensitivity


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
        updated_logged_values = self.periodic_density_filter.update_logged_values(logged_values)
        rve_problem_logged_values = {"SIMP Exponent": self.__simp_exponent}
        updated_logged_values.update(rve_problem_logged_values)
        return updated_logged_values


    ###################################################################################################################
    ###################################################################################################################
    def plot_mesh(self, plot_periodic_boundary_conditions: bool = True, save_figure: bool = True):
        """Create a simple plot of the RVE mesh.

        Created a matplotlib plot illustrating the RVE mesh and the nodesets used for imposing periodicity.

        Args:
            plot_periodic_boundary_conditions: If True the boundary conditions are plotted, otherwise not.
            save_figure: If True, the figure is saved as a PDF in the output directory.

        Returns:
            None
        """
        if shutil.which("tex") and shutil.which("dvipng") and shutil.which("gs"):
            plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        figure_size_scale_factor = 1.2
        fig = plt.figure(figsize=(figure_size_scale_factor*4.8, figure_size_scale_factor*4.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.triplot(self.__nodal_coordinates[:, 0], self.__nodal_coordinates[:, 1], self.__element_connectivity,
                   color='k', linewidth=1.0, alpha=0.4)
        if plot_periodic_boundary_conditions:
            #
            face_node_pair_x_coordinates = self.__nodal_coordinates[self.__face_node_pairs, 0]
            face_node_pair_y_coordinates = self.__nodal_coordinates[self.__face_node_pairs, 1]
            ax.plot(face_node_pair_x_coordinates[:, 0], face_node_pair_y_coordinates[:, 0], 'go',
                    label='independent boundary nodes')
            ax.plot(face_node_pair_x_coordinates[:, 1], face_node_pair_y_coordinates[:, 1], 'gx',
                    label='dependent boundary nodes')
            #
            interior_node_x_coordinates = self.__nodal_coordinates[self.__interior_node_indices, 0].ravel()
            interior_node_y_coordinates = self.__nodal_coordinates[self.__interior_node_indices, 1].ravel()
            ax.plot(interior_node_x_coordinates, interior_node_y_coordinates, 'bo', label='interior nodes')
            #
            corner_node_x_coordinates = self.__nodal_coordinates[self.__corner_node_indices, 0].ravel()
            corner_node_y_coordinates = self.__nodal_coordinates[self.__corner_node_indices, 1].ravel()
            ax.plot(corner_node_x_coordinates, corner_node_y_coordinates, 'ro', label='corner nodes')
            #
            ax.legend(bbox_to_anchor=(0.5, 1.25), loc='upper center')
        ax.set_xlabel('RVE X coordinate')
        ax.set_ylabel('RVE Y coordinate')
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        if save_figure:
            rve_figure_filepath = os.path.join(self.__output_directory_path, "RepresentativeVolumeElementMesh.pdf")
            fig.savefig(rve_figure_filepath)
