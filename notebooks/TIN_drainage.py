from TIN_engine import *
import uuid

class Node:
    def __init__(self, point):
        self.point: tuple = point
        self.upstream_nodes: list[Node] = [] # parents
        self.downstream_node: Node = None # child
        self.id = str(uuid.uuid4())

from TIN_draw import *

def calculate_steepest_descent_line2(ax, start_triangle: Triangle, triangles: list[Triangle], triangle_to_outlet_node: dict[Triangle, Node], max_steps: int = 1000):
    if start_triangle in triangle_to_outlet_node: # TODO: check if this is correct
        return triangle_to_outlet_node[start_triangle]

    current_triangle = start_triangle
    current_point = current_triangle.get_centroid().coord()
    current_node = Node(current_point)
    previous_z = current_point[2]
    onEdge = False
    current_edge = None
    current_vertex = None
    onVertex = False

    print("current_triangle", current_triangle)
    print("current_point", current_point)
    #draw_point(ax, current_point, markersize=5)

    for i in range(1, max_steps + 1):

        if not onEdge and not onVertex:
            print("At triangle id:", current_triangle.id, " at point:", current_point)

            if current_triangle in triangle_to_outlet_node:
                # connect to existing outlet node
                existing_outlet_node = triangle_to_outlet_node[current_triangle]
                current_node.downstream_node = existing_outlet_node
                existing_outlet_node.upstream_nodes.append(current_node)
                return None

            descent = calculate_steepest_descent2(current_triangle)
            descent = descent / np.linalg.norm(descent)
            descent *= 0.001

            next_point, adj_tri, v1, v2 = get_point_and_adj_triangle_from_descent2(current_triangle, current_point[0:2], descent)
            if next_point is None:
                print("next_point should not be None, stopping at iteration ", i)
                break

            new_node = Node((next_point[0], next_point[1], previous_z))
            new_node.upstream_nodes.append(current_node)
            current_node.downstream_node = new_node
            triangle_to_outlet_node[current_triangle] = new_node

            current_node = new_node
            current_point = next_point
            previous_z = new_node.point[2]

            if adj_tri is None:
                print("No adjacent triangle found, stopping at iteration ", i)
                break
        
            #draw_point(ax, current_point, markersize=4)

            current_triangle = adj_tri 
            current_edge = (v1, v2)
            onEdge = True
            onVertex = False
        elif onEdge:
            print("At edge, tri.id", current_triangle.id)

            if current_triangle in triangle_to_outlet_node:
                # connect to existing outlet node
                existing_outlet_node = triangle_to_outlet_node[current_triangle]
                current_node.downstream_node = existing_outlet_node
                existing_outlet_node.upstream_nodes.append(current_node)
                return None

            descent_current = calculate_steepest_descent2(current_triangle) # gh from p. 1239, Jones et al.

            if current_edge is None:
                TIN_log("ERROR: current_edge is None while onEdge is True")
                break

            v1, v2 = current_edge

            # TODO: make sure no refernce errors here
            current_triangle = make_triangle_counterclockwise2(current_triangle)

            if (current_triangle.vertices().index(v1) + 1) % 3 == current_triangle.vertices().index(v2): # if v1 comes before v2 
                ij = v2.coord() - v1.coord()
            else: # v2 comes before v1
                # swap
                temp = v1
                v1 = v2
                v2 = temp

                ij = v2.coord() - v1.coord()

            # direction of the current triangle: if positive then current triangle slopes toward the current edge
            direction = utils.cross_2D(np.array(descent_current), ij[0:2])

            if direction > 0: #TODO
                print("slopes away from edge edge")

                # basically do the same as when not an edge and not a vertex 
                descent = descent_current / np.linalg.norm(descent_current)
                descent *= 0.001

                next_point, adj_tri, v1, v2 = get_point_and_adj_triangle_from_descent2(current_triangle, current_point[0:2], descent)
                if next_point is None:
                    print("next_point should not be None, stopping at iteration ", i)
                    break

                new_node = Node((next_point[0], next_point[1], previous_z))
                new_node.upstream_nodes.append(current_node)
                current_node.downstream_node = new_node
                triangle_to_outlet_node[current_triangle] = new_node

                current_node = new_node
                current_point = next_point
                previous_z = new_node.point[2]

                if adj_tri is None:
                    print("No adjacent triangle found, stopping at iteration ", i)
                    break

                draw_point(ax, current_point, markersize=3)

                current_triangle = adj_tri 
                current_edge = (v1, v2)
                onEdge = True
                onVertex = False
            else:
                print("slopes toward edge")

                # go down to lowest vertex
                lowest = None
                if v1.z > v2.z:
                    lowest = v2
                else:
                    lowest = v1
                
                new_node = Node(lowest.coord())
                new_node.upstream_nodes.append(current_node)
                current_node.downstream_node = new_node
                triangle_to_outlet_node[current_triangle] = new_node

                current_node = new_node
                current_point = lowest.coord()
                previous_z = lowest.z

                current_vertex = lowest

                onEdge = False
                onVertex = True
        elif onVertex:
            print("At vertex")
            #draw_vertex(ax, current_vertex, markersize=2)

            ordered = get_all_triangles_and_edges_at_point(current_vertex, triangles) # TODO: inefficent, need to optimize
            previous_item = None
            next_item = None
            j = 0
            next_point = None
            next_triangle = None
            adj_triangles = []
            for item in ordered:
                previous_item = ordered[(j - 1) % len(ordered)]
                next_item = ordered[(j + 1) % len(ordered)]

                if isinstance(item, Triangle):
                    if test_triangle(current_vertex, item):
                        next_triangle = item
                else: # must be an edge
                    # next_item and previous_item should be triangles adjacent to this edge
                    if test_edge(current_vertex, item, next_item, previous_item):
                        next_point = get_other_vertex_from_edge(current_vertex, item)
                        adj_triangles.append(next_item)
                        adj_triangles.append(previous_item)
                        #result.append(next_point.coord())
                        #previous_z = next_point.z
                
                j += 1
            
            if next_point is not None:
                if adj_triangles[0] in triangle_to_outlet_node:
                    # connect to existing outlet node
                    existing_outlet_node = triangle_to_outlet_node[adj_triangles[0]]
                    current_node.downstream_node = existing_outlet_node
                    existing_outlet_node.upstream_nodes.append(current_node)
                    return None
                elif adj_triangles[1] in triangle_to_outlet_node:
                    # connect to existing outlet node
                    existing_outlet_node = triangle_to_outlet_node[adj_triangles[1]]
                    current_node.downstream_node = existing_outlet_node
                    existing_outlet_node.upstream_nodes.append(current_node)
                    return None

                new_node = Node(next_point.coord())
                new_node.upstream_nodes.append(current_node)
                current_node.downstream_node = new_node

                # add the two adjacent triangles that share this channel edge
                triangle_to_outlet_node[adj_triangles[0]] = new_node
                triangle_to_outlet_node[adj_triangles[1]] = new_node

                current_node = new_node
                current_point = next_point.coord()
                previous_z = next_point.z

                current_vertex = next_point

                onEdge = False
                onVertex = True
            elif next_triangle is not None:
                current_triangle = next_triangle

                onEdge = False
                onVertex = False
            else:
                # need to stop here, no where to go
                break

    return current_node


# See Freitas et al. 2016
def create_drainage_network(ax, triangles: list[Triangle]) -> list[Node]:
    triangle_to_outlet_node = {}  # mapping from triangle id to outlet node
    outlet_nodes = []

    # sort triangles by elevation
    sorted_triangles = sorted(triangles, key=lambda t: t.get_centroid().z, reverse=True)

    for triangle in sorted_triangles:
        outlet = calculate_steepest_descent_line2(ax, triangle, sorted_triangles, triangle_to_outlet_node)

        if outlet is not None:
            outlet_nodes.append(outlet) # new outlet node created
        # else: already connected to existing outlet node
    
    return outlet_nodes
