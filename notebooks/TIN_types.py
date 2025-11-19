import numpy as np
from settings import DEBUG_TIN
import utils

def TIN_log(msg):
    if DEBUG_TIN:
        print("[TIN]: {}".format(msg))

class Vertex:
    def __init__(self, x, y, z, id):
        self.x = x
        self.y = y
        self.z = z
        self.id = id
        self.connected_vertices = []
        self.connected_triangles = []
    
    def get_vertex_ids(self):
        return [v.id for v in self.connected_vertices]

    def is_vertex_connected(self, vertex):
        return vertex.id in self.get_vertex_ids()

    def coord(self):
        return np.array([self.x, self.y, self.z])
    
    def coord2D(self):
        return np.array([self.x, self.y])
    
    def __str__(self):
        return "Vertex(id: {}, x: {}, y: {}, z: {})".format(self.id, self.x, self.y, self.z)
    
    def __eq__(self, other):
        return self.id == other.id

class Triangle:
    # v1, v2, v3 are vertices (Vertex objects)
    # a1, a2, a3 are adjacent triangles (Triangle objects)
    def __init__(self, v1, v2, v3, a1, a2, a3, id):
        self.v1: Vertex = v1
        self.v2: Vertex = v2
        self.v3: Vertex = v3
        self.a1: Triangle = a1
        self.a2: Triangle = a2
        self.a3: Triangle = a3
        self.id = id
    
    def vertices(self):
        return [self.v1, self.v2, self.v3]
    
    def adjacent_triangles(self):
        return [self.a1, self.a2, self.a3]
    
    def triangle_has_vertex(self, vertex):
        return self.v1.id == vertex.id or self.v3.id == vertex.id or self.v2.id == vertex.id
    
    def get_other_vertex(self, v1, v2):
        if not self.triangle_has_vertex(v1) or not self.triangle_has_vertex(v2):
            TIN_log("ERROR: one or both vertices not in triangle")
            return None
        
        for v in self.vertices():
            if v.id != v1.id and v.id != v2.id:
                return v
    
    # This function was generated with AI (Copilot Code Completion)
    # Returns the shared edge (as a tuple of two Vertex objects) between this triangle
    # and an (hopefully adjacent) triangle
    def get_shared_edge(self, other_triangle):
        shared_vertices = []
        for v in self.vertices():
            if other_triangle.triangle_has_vertex(v):
                shared_vertices.append(v)
        
        if len(shared_vertices) != 2:
            TIN_log("Error: triangles do not share an edge (self.id: {}, other_triangle.id: {}, shared_vertices length: {}, are adjacent: {})".format(self.id, other_triangle.id, len(shared_vertices), self.is_adjacent(other_triangle)))
            return None
        elif self.is_adjacent(other_triangle) == False:
            TIN_log("Warning: triangles share an edge but are not marked as adjacent (self.id: {}, other_triangle.id: {}, shared_vertices length: {})".format(self.id, other_triangle.id, len(shared_vertices)))
        
        return (shared_vertices[0], shared_vertices[1])
    
    def is_adjacent(self, other_triangle):
        if other_triangle is None:
            return False
        
        return (self.a1 is not None and self.a1.id == other_triangle.id) or (self.a2 is not None and self.a2.id == other_triangle.id) or (self.a3 is not None and self.a3.id == other_triangle.id)

    def convert_to_array_triangle(self):
        return np.array([self.v1.coord(), self.v2.coord(), self.v3.coord()])
    
    def convert_to_indices(self):
        return np.array([self.v1.id, self.v2.id, self.v3.id])
    
    def get_edges_with_vertex(self, vertex):
        if vertex not in self.vertices():
            print("Error: vertex does not exist in this triangle")
            return (None, None)
        
        other = self.vertices()
        other.remove(vertex)
        edges = ([vertex, other[0]], [vertex, other[1]])
        edge_vecs = (np.array([edges[0][1].x - edges[0][0].x, edges[0][1].y - edges[0][0].y]),
                     np.array([edges[1][1].x - edges[1][0].x, edges[1][1].y - edges[1][0].y]))

        side = utils.cross_2D(edge_vecs[0], edge_vecs[1])
        if side < 0:
            # swap, since they are on the wrong side (i.e. the cross product should be positive)
            edges = (edges[1], edges[0])
        
        return edges
    
    # This function was generated with AI (Copilot Code Completion)
    # Returns the centroid of the triangle as a Vertex object
    def get_centroid(self):
        x = (self.v1.x + self.v2.x + self.v3.x) / 3.0
        y = (self.v1.y + self.v2.y + self.v3.y) / 3.0
        z = (self.v1.z + self.v2.z + self.v3.z) / 3.0
        return Vertex(x, y, z, -1) # id of -1 since it's not a real vertex
    
    def __str__(self):
        return "Triangle(id: {}, v1: {}, v2: {}, v3: {})".format(self.id, self.v1, self.v2, self.v3)