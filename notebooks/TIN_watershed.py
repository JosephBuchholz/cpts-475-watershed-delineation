from TIN_drainage import *

def delineate_watershed_from_drainage_node(drainage_node: Node):
    watershed_triangles = set()
    nodes_to_visit = [drainage_node]

    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        for triangle in current_node.upstream_triangles:
            watershed_triangles.add(triangle)
        for upstream_node in current_node.upstream_nodes:
            nodes_to_visit.append(upstream_node)

    return watershed_triangles

def delineate_watershed_from_nearest_outlet(drainage_network: list[Node], pour_point):
    closest_node = None
    for node in drainage_network:
        if closest_node is None:
            closest_node = node
        else:
            dist_current = np.linalg.norm(np.array(node.point[0:2]) - np.array(pour_point))
            dist_closest = np.linalg.norm(np.array(closest_node.point[0:2]) - np.array(pour_point))
            if dist_current < dist_closest:
                closest_node = node

    return delineate_watershed_from_drainage_node(closest_node), closest_node

def delineate_random_watershed(drainage_network: list[Node]):
    random_index = np.random.randint(0, len(drainage_network))
    start_node = drainage_network[random_index]
    return delineate_watershed_from_drainage_node(start_node), start_node

def delineate_largest_watershed(drainage_network: list[Node]):
    largest_watershed = set()
    start_node = None
    for node in drainage_network:
        current_watershed = delineate_watershed_from_drainage_node(node)
        if len(current_watershed) > len(largest_watershed):
            largest_watershed = current_watershed
            start_node = node

    return largest_watershed, start_node
