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

def delineate_watershed(drainage_network: list[Node], pour_point):
    # TODO: find closest drainage node to pour_point
    random_index = np.random.randint(0, len(drainage_network))
    start_node = drainage_network[random_index]
    return delineate_watershed_from_drainage_node(start_node), start_node
