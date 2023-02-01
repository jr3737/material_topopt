"""A module containing numerous utility functions.

Numerous finite element utility functions for 3-node triangles, including isotropic constitutive tensors,
finite element matrix assembly, and generation of element stiffness matrices, shape functions, and areas.
Additionally, this module provides a function for returning the independent components of the coarse
scale constitutive tensor, along with the relevant symmetries, and a function for checking user-supplied
parameters against multiple criteria.
"""
import os
import logging
import numpy as np
from typing import Any
import scipy
try:
    from pypardiso import factorized as _factorized
    linear_algebra_package = "pypardiso"
except ImportError:
    from scipy.sparse.linalg import factorized as _factorized
    linear_algebra_package = "scipy.sparse.linalg"


#######################################################################################################################
#######################################################################################################################
def setup_logging(logging_level: int, logfile_path: str = ""):
    """Set up the optional logger at the specified level.

    Args:
        logging_level: An integer representing the logging level. Options are logging.DEBUG, logging.INFO,
            logging.WARNING, logging.ERROR, and logging.CRITICAL in order of decreasing output
        logfile_path: A string representing the filepath including the filename into which the logged output will
            be placed, in addition to console output. If this remains an empty string then no file logging will occur.

    Returns:
        None
    """
    logger = logging.getLogger("MaterialTopOpt")
    logger.setLevel(logging_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    first_line = "Logged Columns: Time (Hour:Minute:Second) , Logger Name , Logging Level , Message"
    formatter = logging.Formatter('%(asctime)s , %(name)s , %(levelname)s , %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info(first_line)
    if logfile_path != "":
        absolute_logfile_path = os.path.abspath(logfile_path)
        logfile_directory = os.path.dirname(absolute_logfile_path)
        if not os.path.exists(logfile_directory):
            os.mkdir(logfile_directory)
        with open(absolute_logfile_path, mode='w', encoding="UTF-8") as fid:
            fid.write(first_line + '\n')
        file_handler = logging.FileHandler(absolute_logfile_path)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


#######################################################################################################################
#######################################################################################################################
def factorized(sparse_matrix: scipy.sparse.spmatrix):
    """Wrapper function for matrix factorization based on currently installed linear algebra packages.

    Args:
        sparse_matrix: A scipy sparse matrix to be factored for reuse with multiple right hand side vectors.

    Returns:
        A callable solve function which uses the factored matrix.
    """
    logger = logging.getLogger("MaterialTopOpt.Utilities")
    logger.debug("matrix factorization with %s begin", linear_algebra_package)
    solve_function = _factorized(sparse_matrix)
    logger.debug("matrix factorization with %s end", linear_algebra_package)
    return solve_function


#######################################################################################################################
#######################################################################################################################
def get_2d_isotropic_linear_elastic_4th_order_constitutive_tensor(elastic_modulus: float = 1.0,
                                                                  poissons_ratio: float = 0.25):
    """Get a 2D isotropic, linear elastic, plane stress constitutive tensor.

    Args:
        elastic_modulus: The elastic modulus as a float.
        poissons_ratio: Poisson's ratio as a float.

    Returns:
        A 4-dimensional numpy array containing the 4th order constitutive tensor with shape (2, 2, 2, 2)
    """
    assert elastic_modulus > 0.0, f"The elastic modulus must be positive. Specified '{elastic_modulus}'"
    assert -1.0 < poissons_ratio < 0.5, f"Poissons ration must be > -1 and < 0.5. Specified '{poissons_ratio}'"
    space_dimension = 2
    shear_modulus = elastic_modulus / (2.0 * (1.0 + poissons_ratio))
    # plane stress
    lame_constant = elastic_modulus * poissons_ratio / (1.0 - poissons_ratio**2)
    identity_tensor = np.eye(space_dimension)
    identity_x_identity = np.einsum('ij,kl->ijkl', identity_tensor, identity_tensor)
    temp1 = np.einsum('ik,jl->ijkl', identity_tensor, identity_tensor)
    temp2 = np.einsum('il,jk->ijkl', identity_tensor, identity_tensor)
    fourth_order_symmetric_identity_tensor = 0.5 * (temp1 + temp2)
    linear_elastic_constitutive_tensor = lame_constant * identity_x_identity + \
        (2.0 * shear_modulus) * fourth_order_symmetric_identity_tensor
    return linear_elastic_constitutive_tensor


#######################################################################################################################
#######################################################################################################################
def get_triangle_shape_functions_and_element_areas(nodal_coordinates: np.ndarray,
                                                   element_connectivity: np.ndarray):
    '''Computes the shape function values, gradients, and element areas for linear triangular finite elements.

    Args:
        nodal_coordinates: np.ndarray of nodal coordinates with shape (number of nodes, 2)
        element_connectivity: np.ndarray of nodal indices for each element with shape (number of elements, 3)

    Returns:
        A tuple (a, b, c) where 'a' is a numpy array of shape function values in natural coordinates,
        'b' is a numpy array of shape function gradients in physical coordinates for every element, and
        'c' is a numpy array of element areas
    '''
    logger = logging.getLogger("MaterialTopOpt.Utilities")
    logger.debug("get_triangle_shape_functions_and_element_areas begin")
    space_dimension = nodal_coordinates.shape[1]
    assert space_dimension == 2, "The shape of the nodal_coordinates array should be (number of nodes, 2)"
    number_of_nodes_per_element = element_connectivity.shape[1]
    assert number_of_nodes_per_element == 3, \
        "The shape of the element_connectivity array should be (number of elements, 3). " + \
            "This function only works for 3 node triangular elements"

    nodal_coordinates_elementwise = nodal_coordinates[element_connectivity, :]
    quadrature_point_x_coordinate = 1.0 / 3.0
    quadrature_point_y_coordinate = 1.0 / 3.0
    quadrature_point_weight = 0.5
    shape_function_values_in_natural_coordinates = \
        np.array([(1.0 - quadrature_point_x_coordinate - quadrature_point_y_coordinate),
                  quadrature_point_x_coordinate,
                  quadrature_point_y_coordinate])
    shape_function_gradients_in_natural_coordinates = np.array([[-1.0, -1.0],
                                                                [ 1.0,  0.0],
                                                                [ 0.0,  1.0]])
    jacobian_per_element = \
        np.einsum('eni,nj->eij', nodal_coordinates_elementwise, shape_function_gradients_in_natural_coordinates)
    element_areas = quadrature_point_weight * np.linalg.det(jacobian_per_element)
    inverse_jacobian_per_element = np.linalg.inv(jacobian_per_element)
    shape_function_gradients_in_physical_coordinates = \
        np.einsum('eji,nj->eni', inverse_jacobian_per_element, shape_function_gradients_in_natural_coordinates)

    assert np.all(element_areas > 0.0), \
        "Element areas cannot be zero or negative. Ensure correct node ordering in the provided finite element mesh."
    logger.debug("get_triangle_shape_functions_and_element_areas end")
    return shape_function_values_in_natural_coordinates, shape_function_gradients_in_physical_coordinates, element_areas


#######################################################################################################################
#######################################################################################################################
def get_virtual_displacement_symmetric_gradients(shape_function_gradients: np.ndarray):
    '''Computes virtual displacement symmetric gradients for every element.

    Args:
        shape_function_gradients: A numpy array containing the shape function gradients for every element.

    Returns:
        A numpy array containing the virtual displacement symmetric gradients for every element.
    '''
    logger = logging.getLogger("MaterialTopOpt.Utilities")
    logger.debug("get_virtual_displacement_symmetric_gradients begin")
    _, number_of_nodes_per_element, number_of_dofs_per_node = shape_function_gradients.shape
    number_of_dofs_per_element = number_of_dofs_per_node * number_of_nodes_per_element
    du = np.zeros((number_of_dofs_per_element, number_of_nodes_per_element, number_of_dofs_per_node))
    for element_node_index in range(number_of_nodes_per_element):
        for nodal_dof_index in range(number_of_dofs_per_node):
            element_dof_index = element_node_index * number_of_dofs_per_node + nodal_dof_index
            du[element_dof_index, element_node_index, nodal_dof_index] = 1.0
    single_element_test_function_vector_fields = np.asarray(du)
    virtual_displacement_gradients = np.einsum('inx,end->eixd',
                                                single_element_test_function_vector_fields,
                                                shape_function_gradients)
    virtual_displacement_gradients_transpose = virtual_displacement_gradients.transpose((0, 1, 3, 2))
    virtual_displacement_symmetric_gradients = \
        0.5 * (virtual_displacement_gradients + virtual_displacement_gradients_transpose)
    logger.debug("get_virtual_displacement_symmetric_gradients end")
    return virtual_displacement_symmetric_gradients


#######################################################################################################################
#######################################################################################################################
def get_2d_linear_elasticity_problem_dof_indices(nodal_coordinates: np.ndarray,
                                                 element_connectivity: np.ndarray):
    '''Computes the indices of degrees of freedom, in addition to the corresponding matrix and vector indices.

    Args:
        nodal_coordinates: np.ndarray of nodal coordinates with shape (number of nodes, 2)
        element_connectivity: np.ndarray of nodal indices for each element with shape (number of elements, 3)

    Returns:
        A tuple (a, b, c, d) where 'a' is a numpy array of nodal dof indices, 'b' is a numpy array containing the
        row indices for a sparse matrix in COO format, 'c' is a numpy array containing the column indices for a
        sparse matrix in COO format, and 'd' is a numpy array containing the indices needed to efficiently assemble
        the residual vector in COO format.
    '''
    logger = logging.getLogger("MaterialTopOpt.Utilities")
    logger.debug("get_2D_linear_elasticity_problem_dof_indices begin")
    number_of_nodes, space_dimension = nodal_coordinates.shape
    assert space_dimension == 2, "Only implemented for space dimenion equal to 2"
    number_of_elements, number_of_nodes_per_element = element_connectivity.shape
    number_of_dofs_per_node = space_dimension
    total_number_of_nodal_dofs = number_of_nodes * number_of_dofs_per_node
    nodal_dof_indices = \
        np.arange(total_number_of_nodal_dofs, dtype=int).reshape((number_of_nodes, number_of_dofs_per_node))
    number_of_dofs_per_element = number_of_nodes_per_element * number_of_dofs_per_node
    dof_indices_elementwise = nodal_dof_indices[element_connectivity, :].reshape(
        (number_of_elements, number_of_dofs_per_element))
    column_indices = np.tile(dof_indices_elementwise, (number_of_dofs_per_element, 1, 1)).transpose((1, 0, 2))
    row_indices = column_indices.transpose((0, 2, 1))

    matrix_row_indices = row_indices.flatten()
    matrix_column_indices = column_indices.flatten()
    residual_vector_indices_elementwise = dof_indices_elementwise
    logger.debug("get_2D_linear_elasticity_problem_dof_indices end")
    return nodal_dof_indices, matrix_row_indices, matrix_column_indices, residual_vector_indices_elementwise


#######################################################################################################################
#######################################################################################################################
def get_2d_linear_elasticity_problem_stiffness_matrix_elementwise(linear_elastic_constitutive_tensor: np.ndarray,
                                                                  virtual_displacement_symmetric_gradients: np.ndarray,
                                                                  element_areas: np.ndarray) -> np.ndarray:
    '''Computes the 2D linear elastic stiffness matrix for every element in the mesh.

    Args:
        linear_elastic_constitutive_tensor: A numpy array containing a 4th order linear elastic constitutive tensor.
        virtual_displacement_symmetric_gradients: A numpy array with the virtual displacement symmetric gradients.
        element_areas: A numpy array containing the area of each element.

    Returns:
        A numpy array containing the stiffness matrix for every element in the finite element mesh. The array has
        shape (number of elements, number of element dofs, number of element dofs) where number of element dofs is 6
        for 2D linear elasticity with 3-node triangular finite elements.
    '''
    logger = logging.getLogger("MaterialTopOpt.Utilities")
    logger.debug("get_2D_linear_elasticity_problem_stiffness_matrix_elementwise begin")
    stiffness_matrix_elementwise = np.einsum('emij,ijkl,enkl,e->emn',
                                             virtual_displacement_symmetric_gradients,
                                             linear_elastic_constitutive_tensor,
                                             virtual_displacement_symmetric_gradients,
                                             element_areas,
                                             optimize=True)
    logger.debug("get_2D_linear_elasticity_problem_stiffness_matrix_elementwise end")
    return stiffness_matrix_elementwise


#######################################################################################################################
#######################################################################################################################
def get_list_of_2d_constitutive_tensor_independent_components_and_symmetries():
    '''Returns the 6 independent components of the coarse scale constitutive tensor and their symmetries.

    Args:
        None

    Returns:
        A tuple (a, b) where 'a' is a list of tuples of integers representing the 6 independent components of the
        coarse scale constitutive tensor and 'b' is a list of lists of tuples of integers, corresponding to the
        symmetries of each of the 6 independent components.
    '''
    independent_components = [(0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 1, 1, 1), (1, 1, 0, 1), (0, 1, 0, 1)]
    component_symmetries = [[], [(1, 1, 0, 0)], [(0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)], [],
                            [(1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)], [(1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0)]]
    return independent_components, component_symmetries



#######################################################################################################################
#######################################################################################################################
def get_parameter_or_default(user_parameters: dict,
                             param_key: str,
                             param_type: type,
                             bounds: tuple = (None, None),
                             default_value: Any = None,
                             required: bool = False,
                             additional_message: str = None):
    '''User parameter parsing function.

    Multiple checks can be performed on user parameters including whether it exists in the dictionary object, and if
    it doesn't, whether a default parameter should be returned. Additionally, the parameter type is checked and if the
    type is numeric, it can also be checked to ensure it is within allowable bounds. If the parameter is required and
    a user-specified value and the default value are not provided, then an exception is thrown.

    Args:
        user_parameters: A dictionary containing user-specified parameters.
        param_key: A string key for the user_parameters dictionary.
        param_type: The required type of the parameter.
        bounds: A tuple including both a lower and upper bound for the parameter. 'None' may be used as a place holder.
        default_value: The default value if the parameter was not specified by the user.
        required: A bool indicating whether the parameter is required.
        additional_message: An optional string to be displayed with any potential error message.

    Returns:
        The parameter value.
    '''
    def message(err: str):
        return "\n" + err + f"\n Additional information: {additional_message}"
    assert isinstance(user_parameters, dict), \
        message(f"Searching for parameter key '{param_key}' in a user parameters object that is not a dictionary." + \
            f" The object provide is of type {type(user_parameters)}.")
    assert len(bounds) == 2, message(f"Bounds specified for default parameter {param_key} is not a tuple " + \
            "of length 2. If there is no upper bound or lower bound, specify as (None, None).")
    lower_bound, upper_bound = bounds
    if lower_bound is not None:
        if not isinstance(lower_bound, param_type):
            my_message = message(f"The specified lower bound '{lower_bound}' for parameter '{param_key}'" + \
                f" is not of type '{param_type}'")
            raise TypeError(my_message)
    if upper_bound is not None:
        if not isinstance(upper_bound, param_type):
            my_message = message(f"The specified upper bound '{upper_bound}' for parameter '{param_key}'" + \
                f" is not of type '{param_type}'")
            raise TypeError(my_message)
    if (lower_bound is not None) and (upper_bound is not None):
        if lower_bound > upper_bound:
            my_message = message(f"The specified lower bound '{lower_bound}' for parameter '{param_key}'" + \
                f" is larger than the upper bound '{upper_bound}'")
            raise ValueError(my_message)
    if param_key in user_parameters.keys():
        param_value = user_parameters[param_key]
        if not isinstance(param_value, param_type):
            my_message = message(f"User param '{param_key}' is type '{type(param_value)}' rather than '{param_type}'")
            raise TypeError(my_message)
        assert len(bounds) == 2, \
            message(f"Bounds specified for user parameter {param_key} is not a tuple of length 2. " + \
                "If there is no upper bound or lower bound, specify as (None, None)")
        if lower_bound is not None:
            assert param_value >= lower_bound, \
                message(f"User parameter '{param_key}' is required to be greater than or equal to {lower_bound}. " + \
                    f"User specified a value of {param_value}.")
        if upper_bound is not None:
            assert param_value <= upper_bound, \
                message(f"User parameter '{param_key}' is required to be less than or equal to {upper_bound}. " + \
                    f"User specified a value of {param_value}.")
        return param_value
    else:
        if required and (default_value is None):
            raise ValueError(message(f"User must specify parameter '{param_key}' of type '{param_type}' " + \
                "since it is required with no valid default."))
        if default_value is None:
            return None
        param_value = default_value
        if not isinstance(param_value, param_type):
            raise TypeError(message(f"Default parameter '{param_key}' is type " + \
                f"'{type(param_value)}' but it must be of type {param_type}"))
        if lower_bound is not None:
            assert param_value >= lower_bound, message(f"Default parameter '{param_key}' is required to be " + \
                f"greater than or equal to {lower_bound}. User specified a value of {param_value}.")
        if upper_bound is not None:
            assert param_value <= upper_bound, message(f"Default parameter '{param_key}' is required to be " + \
                f"less than or equal to {upper_bound}. User specified a value of {param_value}.")
        return param_value
