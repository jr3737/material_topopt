'''Script to create a simple 3-node triangle mesh of a general polygon.
conda install -c conda-forge trimesh shapely
pip install triangle
'''
import numpy as np
import trimesh_utilities

mesh_filename = "lbracket.inp"
average_finite_element_size = 0.02
# Create a list of tuples representing the bounding (x, y) points of the polygon
#   Note: This list must start and end with the same (x, y) point
L = 1.0 # The length of the L-bracket's longest edge
W = 0.4 # The length of the L-bracket's shortest edge
my_bounding_points = [(0.0, 0.0), (L, 0.0), (L, W), (W, W), (W, L), (0.0, L), (0.0, 0.0)]

nodal_coordinates, element_connectivity = trimesh_utilities.get_nodal_coordinates_and_element_connectivity(
        my_bounding_points,
        average_finite_element_size
    )

nodal_x_coordinates = nodal_coordinates[:, 0].ravel()
nodal_y_coordinates = nodal_coordinates[:, 1].ravel()

geometric_tolerance = average_finite_element_size / 100.0

# Create the index sets of nodes to specify boundary conditions later
indices_of_nodes_on_the_top_edge   = np.argwhere(np.abs(nodal_y_coordinates - L) < geometric_tolerance).ravel()
indices_of_nodes_on_the_right_edge = np.argwhere(np.abs(nodal_x_coordinates - L) < geometric_tolerance).ravel()

# nodesets is a dictionary which maps the nodeset name as a string to a numpy array
#   of integers representing the indices of the nodes in the nodeset
nodesets = {"top_edge_nodeset":   indices_of_nodes_on_the_top_edge,
            "right_edge_nodeset": indices_of_nodes_on_the_right_edge}

# Write the meshfile to the current directory
trimesh_utilities.write_the_mesh_to_file(mesh_filename, nodal_coordinates, element_connectivity, nodesets)

# Plot the mesh and the nodesets just as a quick visual check
trimesh_utilities.plot_mesh_and_nodesets(nodal_coordinates, element_connectivity, nodesets)
