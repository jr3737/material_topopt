'''Script to create a simple 3-node triangle mesh of a general polygon.
'''
import numpy as np
import trimesh_utilities

mesh_filename = "letterG.inp"
average_finite_element_size = 0.25
# Create a list of tuples representing the bounding (x, y) points of the polygon
#   Note: This list must start and end with the same (x, y) point
t = 1.0 # The thickness of the letter
my_bounding_points = [(0.0, 0.0),
                      (5*t, 0.0),
                      (5*t, 3*t),
                      (3.5*t, 3*t),
                      (3.5*t, 2*t),
                      (4*t, 2*t),
                      (4*t, t),
                      (t, t),
                      (t, 5*t),
                      (3*t, 5*t),
                      (3*t, 4.5*t),
                      (4*t, 4.5*t),
                      (4*t, 6*t),
                      (0.0, 6*t),
                      (0.0, 0.0)]

nodal_coordinates, element_connectivity = trimesh_utilities.get_nodal_coordinates_and_element_connectivity(
        my_bounding_points,
        average_finite_element_size
    )

nodal_x_coordinates = nodal_coordinates[:, 0].ravel()
nodal_y_coordinates = nodal_coordinates[:, 1].ravel()

geometric_tolerance = average_finite_element_size / 100.0

# Create the index sets of nodes to specify boundary conditions later
mask = (np.abs(nodal_y_coordinates - 4.5*t) < geometric_tolerance) & (nodal_x_coordinates > (3*t - geometric_tolerance))
indices_of_nodes_at_G_start  = np.argwhere(mask)
mask = (np.abs(nodal_x_coordinates - 3.5*t) < geometric_tolerance) & \
       (nodal_y_coordinates > (2*t - geometric_tolerance)) & \
       (nodal_y_coordinates < (3*t + geometric_tolerance))
indices_of_nodes_at_G_finish = np.argwhere(mask)

# nodesets is a dictionary which maps the nodeset name as a string to a numpy array
#   of integers representing the indices of the nodes in the nodeset
nodesets = {"G_start_nodeset":  indices_of_nodes_at_G_start,
            "G_finish_nodeset": indices_of_nodes_at_G_finish}

# Write the meshfile to the current directory
trimesh_utilities.write_the_mesh_to_file(mesh_filename, nodal_coordinates, element_connectivity, nodesets)

# Plot the mesh and the nodesets just as a quick visual check
trimesh_utilities.plot_mesh_and_nodesets(nodal_coordinates, element_connectivity, nodesets)
