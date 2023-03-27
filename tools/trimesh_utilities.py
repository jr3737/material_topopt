'''Utility functions used to create a simple 3-node triangle mesh of a general two-dimensional polygon.'''
from shapely.geometry import Polygon
import trimesh
import numpy as np
import meshio
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def get_points_between_two_vertices(vertex1, vertex2, approximate_element_size):
    x1, y1 = vertex1
    x2, y2 = vertex2
    segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    assert approximate_element_size <= segment_length, \
        f"The approximate element size of '{approximate_element_size}' cannot be larger " + \
            "than the segment length of '{segment_length}'"
    number_of_intervals = np.ceil(segment_length / approximate_element_size)
    number_of_points_to_add = int(number_of_intervals) - 1
    x_increment = (x2 - x1) / number_of_intervals
    y_increment = (y2 - y1) / number_of_intervals
    additional_points = []
    x = x1
    y = y1
    for _ in range(number_of_points_to_add):
        x += x_increment
        y += y_increment
        additional_points.append((x, y))
    return additional_points


def add_additional_points_to_polygon(bounding_points, approximate_element_size):
    number_of_edges = len(bounding_points) - 1
    new_points_list = []
    for edge_index in range(number_of_edges):
        vertex1 = bounding_points[edge_index]
        vertex2 = bounding_points[edge_index + 1]
        points_between_vertex1_and_vertex2 = get_points_between_two_vertices(vertex1, vertex2, approximate_element_size)
        new_points_list.append(vertex1)
        new_points_list.extend(points_between_vertex1_and_vertex2)
    new_points_list.append(vertex2)
    return new_points_list


def get_nodal_coordinates_and_element_connectivity(my_bounding_points: List[Tuple[float, float]],
                                                   approximate_element_size: float,
                                                   perform_polygon_check: bool = True):
    all_points = add_additional_points_to_polygon(my_bounding_points, approximate_element_size)
    my_polygon = Polygon(all_points)
    maximum_triangle_area = 0.5 * approximate_element_size**2
    if perform_polygon_check:
        geometric_tolerance = approximate_element_size / 100.0
        points = np.array(all_points)
        for i in range(points.shape[0]):
            current_point = points[i, :]
            distances_to_point = np.linalg.norm(points - current_point)
            potentially_identical_point_indices = np.argwhere(distances_to_point < geometric_tolerance)
            if potentially_identical_point_indices.size > 1:
                print(f"Warning: Point '{current_point}' may be the same as these points " + \
                    f"'{points[potentially_identical_point_indices, :]}' which includes the original point")
    my_triangle_args = f'pq30a{maximum_triangle_area:0.6f}'
    nodal_coordinates, element_connectivity = trimesh.creation.triangulate_polygon(
            my_polygon,
            triangle_args=my_triangle_args,
            engine='triangle'
        )
    return nodal_coordinates, element_connectivity


def write_the_mesh_to_file(mesh_filename: str,
                           nodal_coordinates: np.ndarray,
                           element_connectivity: np.ndarray,
                           nodesets: Dict[str, np.ndarray]):
    number_of_nodes = nodal_coordinates.shape[0]
    number_of_elements = element_connectivity.shape[0]
    for nodeset_name, nodeset_array in nodesets.items():
        if nodeset_array.ndim > 1:
            nodesets[nodeset_name] = nodeset_array.flatten()
    cell_block = meshio.CellBlock('triangle', element_connectivity)
    meshio_object = meshio.Mesh(nodal_coordinates, [cell_block], point_sets=nodesets)
    meshio_object.write(mesh_filename)
    print(f"Mesh written to file '{mesh_filename}' with {number_of_nodes} nodes and {number_of_elements} elements.")
    for nodeset_name, nodeset_indices in nodesets.items():
        print(f"\tNodeset '{nodeset_name}' contains {nodeset_indices.size} nodes.")


def plot_mesh_and_nodesets(nodal_coordinates: np.ndarray,
                           element_connectivity: np.ndarray,
                           nodesets: Dict[str, np.ndarray]):
    nodal_x_coordinates = nodal_coordinates[:, 0].ravel()
    nodal_y_coordinates = nodal_coordinates[:, 1].ravel()
    my_marker_specs = ['bo','rx','gv','m^','c*','b>', 'r<']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.triplot(nodal_x_coordinates, nodal_y_coordinates, element_connectivity, color='k', linewidth=1.0)
    for nodeset_index, (nodeset_name, nodeset_indices) in enumerate(nodesets.items()):
        nodeset_x_coordinates = nodal_x_coordinates[nodeset_indices]
        nodeset_y_coordinates = nodal_y_coordinates[nodeset_indices]
        nodeset_marker_spec = my_marker_specs[nodeset_index]
        ax.plot(nodeset_x_coordinates, nodeset_y_coordinates, nodeset_marker_spec, label=nodeset_name)
    ax.legend(loc='best')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    fig.tight_layout()
    plt.show()

