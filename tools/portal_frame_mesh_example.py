'''Script to create a simple 3-node triangle mesh of a general polygon.
conda install -c conda-forge trimesh shapely
pip install triangle
'''
import numpy as np
import trimesh_utilities

mesh_filename = "portal_frame.inp"
average_finite_element_size = 2.0 / 3.0
# Create a list of tuples representing the bounding (x, y) points of the polygon
#   Note: This list must start and end with the same (x, y) point
height = 30.0
length = 2.0 * height
port_height = 7.0 * height / 12.0
base_width = 11.0 * height / 120.0
distributed_load_length = height / 6.0

my_bounding_points = [(0.0, 0.0),
                      (base_width, 0.0),
                      (height, port_height),
                      (length - base_width, 0.0),
                      (length, 0.0),
                      (length, height),
                      (0.0, height),
                      (0.0, 0.0)]

nodal_coordinates, element_connectivity = trimesh_utilities.get_nodal_coordinates_and_element_connectivity(
        my_bounding_points,
        average_finite_element_size
    )

nodal_x_coordinates = nodal_coordinates[:, 0].ravel()
nodal_y_coordinates = nodal_coordinates[:, 1].ravel()

geometric_tolerance = average_finite_element_size / 100.0

# Create the index sets of nodes to specify boundary conditions later
center_x_coordinate = length / 2.0
nodes_on_bottom_edges_mask = np.abs(nodal_y_coordinates - 0.0)    < geometric_tolerance
nodes_on_top_edge_mask     = np.abs(nodal_y_coordinates - height) < geometric_tolerance
nodes_on_bottom_left_edge_mask  = nodes_on_bottom_edges_mask & (nodal_x_coordinates < center_x_coordinate)
nodes_on_bottom_right_edge_mask = nodes_on_bottom_edges_mask & (nodal_x_coordinates > center_x_coordinate)
nodes_on_top_edge_center_mask = nodes_on_top_edge_mask & \
    (nodal_x_coordinates > (center_x_coordinate - distributed_load_length / 2.0)) & \
    (nodal_x_coordinates < (center_x_coordinate + distributed_load_length / 2.0))
indices_of_nodes_on_the_bottom_left_edge  = np.argwhere(nodes_on_bottom_left_edge_mask).ravel()
indices_of_nodes_on_the_bottom_right_edge = np.argwhere(nodes_on_bottom_right_edge_mask).ravel()
indices_of_nodes_in_center_of_top_edge    = np.argwhere(nodes_on_top_edge_center_mask).ravel()
indices_of_node_at_bottom_left_vertex = np.argwhere((nodal_x_coordinates < geometric_tolerance) & \
                                                    (nodal_y_coordinates < geometric_tolerance)).ravel()

# nodesets is a dictionary which maps the nodeset name as a string to a numpy array
#   of integers representing the indices of the nodes in the nodeset
nodesets = {"bottom_left_nodeset":  indices_of_nodes_on_the_bottom_left_edge,
            "bottom_right_nodeset": indices_of_nodes_on_the_bottom_right_edge,
            "center_of_top_edge_nodeset": indices_of_nodes_in_center_of_top_edge,
            "bottom_left_vertex_nodeset": indices_of_node_at_bottom_left_vertex}

# Write the meshfile to the current directory
trimesh_utilities.write_the_mesh_to_file(mesh_filename, nodal_coordinates, element_connectivity, nodesets)

# Plot the mesh and the nodesets quickly just as a quick visual check
trimesh_utilities.plot_mesh_and_nodesets(nodal_coordinates, element_connectivity, nodesets)
