import numpy as np

class Vertex:
    def __init__(self, x, y, z, id):
        self.x = x
        self.y = y
        self.z = z
        self.id = id
    
    def coord(self):
        return np.array([self.x, self.y, self.z])

# "sign" and "point_in_triangle" see https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

# returns True if the point is inside the given triangle, otherwise False
def point_in_triangle(point, triangle):
    d1 = sign(point, triangle[0], triangle[1])
    d2 = sign(point, triangle[1], triangle[2])
    d3 = sign(point, triangle[2], triangle[0])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

# AI (Google AI Overview) wrote the code for this function:
def line_segment_line_intersection(p1, p2, q1, v):
    """
    Finds the intersection point of a line segment and a line in 2D.

    Args:
        p1 (np.array): Start point of the line segment (e.g., np.array([x1, y1])).
        p2 (np.array): End point of the line segment (e.g., np.array([x2, y2])).
        q1 (np.array): A point on the line (e.g., np.array([qx, qy])).
        v (np.array): Direction vector of the line (e.g., np.array([vx, vy])).

    Returns:
        np.array or None: The intersection point if it exists on the segment, otherwise None.
    """
    
    A = np.array([p2 - p1, -v]).T  # Matrix for coefficients of t and s
    b = q1 - p1                   # Right-hand side vector

    # Check for parallel lines (determinant of A close to zero)
    if np.isclose(np.linalg.det(A), 0):
        # Lines are parallel, no unique intersection or infinite intersections
        return None 

    try:
        ts = np.linalg.solve(A, b)
        t, s = ts[0], ts[1]

        if 0 <= t <= 1 and s >= 0:  # Check if intersection is on the line segment
            intersection_point = p1 + t * (p2 - p1)
            return intersection_point
        else:
            return None  # Intersection is outside the line segment
    except np.linalg.LinAlgError:
        return None # No solution (e.g., singular matrix if lines are parallel)

def get_real_vertex_3D(vertex, xs, ys, zs):
    return np.array([xs[vertex], ys[vertex], zs[vertex]])

def get_real_vertex_2D(vertex, xs, ys, zs):
    return np.array([xs[vertex], ys[vertex]])

# translates the given triangle that contains only the indicies of each vertex into a triangle that
# contains the actual values for each vertex (ignores z)
def get_full_2D_triangle(triangle, xs, ys, zs):
    return np.array([[xs[triangle[0]], ys[triangle[0]]], [xs[triangle[1]], ys[triangle[1]]], [xs[triangle[2]], ys[triangle[2]]]])

def get_full_3D_triangle(triangle, xs, ys, zs):
    return np.array([[xs[triangle[0]], ys[triangle[0]], zs[triangle[0]]],
            [xs[triangle[1]], ys[triangle[1]], zs[triangle[1]]],
            [xs[triangle[2]], ys[triangle[2]], zs[triangle[2]]]])

def get_triangle_at(point, triangles, xs, ys, zs):
    for triangle in triangles:
        full = get_full_2D_triangle(triangle, xs, ys, zs)
        if point_in_triangle(point, full):
            return triangle

    print("Failed to get triangle for point")
    return False

def get_triangles_with_edge(p1, p2, triangles):
    triangle_list = [] # there should only ever be two triangles that share the same edge

    for triangle in triangles:
        # this is not the best way to do it, but it was quick and easy
        if triangle[0] == p1 and triangle[1] == p2:
            triangle_list.append(triangle)
        elif triangle[1] == p1 and triangle[2] == p2:
            triangle_list.append(triangle)
        elif triangle[2] == p1 and triangle[0] == p2:
            triangle_list.append(triangle)
        elif triangle[0] == p2 and triangle[1] == p1:
            triangle_list.append(triangle)
        elif triangle[1] == p2 and triangle[2] == p1:
            triangle_list.append(triangle)
        elif triangle[2] == p2 and triangle[0] == p1:
            triangle_list.append(triangle)

    if len(triangle_list) != 2:
        print("Failed getting 2 triangles with the same specified edge: ", p1, ", ", p2)
    
    return triangle_list

# see Jones et al. (p. 1237)
def calculate_steepest_descent(triangle):
    x = (triangle[0][0], triangle[1][0], triangle[2][0])
    y = (triangle[0][1], triangle[1][1], triangle[2][1])
    z = (triangle[0][2], triangle[1][2], triangle[2][2])

    A = y[0] * (z[1] - z[2]) + y[1] * (z[2] - z[0]) + y[2] * (z[0] - z[1])
    B = z[0] * (x[1] - x[2]) + z[1] * (x[2] - x[0]) + z[2] * (x[0] - x[1])
    C = x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])
    #D = -(A * x[0]) - (B * y[0]) - (C * z[1])

    descent = [(A / C), (B / C)]
    return descent

def get_point_from_descent(triangle, start, descent, xs, ys, zs):
    full = get_full_2D_triangle(triangle, xs, ys, zs)

    intersection = line_segment_line_intersection(full[0], full[1], start, descent)
    if intersection is not None:
        return (intersection, triangle[0], triangle[1])
    intersection = line_segment_line_intersection(full[1], full[2], start, descent)
    if intersection is not None:
        return (intersection, triangle[1], triangle[2])
        
    intersection = line_segment_line_intersection(full[2], full[0], start, descent)

    if intersection is None:
        print("ERROR: next_point should never be None")

    return (intersection, triangle[2], triangle[0])

# really long name, but does the same as "get_point_from_descent" except it
# also gets the other, adjacent triangle that touches the intersection point
def get_point_and_adj_triangle_from_descent(triangle, start, descent, xs, ys, zs, triangles):
    intersection, v1, v2 = get_point_from_descent(triangle, start, descent, xs, ys, zs)

    adj_triangles = get_triangles_with_edge(v1, v2, triangles)
    if len(adj_triangles) != 2:
        print("Error: not enough (or too many) triangles")
    
    current_tri = None
    adj_tri = None
    if np.array_equal(adj_triangles[0], np.array(triangle)):
        current_tri = adj_triangles[0]
        adj_tri = adj_triangles[1]
    elif np.array_equal(adj_triangles[1], np.array(triangle)):
        current_tri = adj_triangles[1]
        adj_tri = adj_triangles[0]
    else:
        print("Error: this should not happen, one of the triangles must be equal")
    
    return (intersection, adj_tri, v1, v2)

# see: https://math.stackexchange.com/questions/1324179/how-to-tell-if-3-connected-points-are-connected-clockwise-or-counter-clockwise
# triangle should be an array of three points (e.g. [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
def is_clockwise(triangle):
    matrix = np.array([[triangle[0][0], triangle[0][1], 1],
                       [triangle[1][0], triangle[1][1], 1],
                       [triangle[2][0], triangle[2][1], 1]])
    
    det = np.linalg.det(matrix)

    if det < 0:
        return True # clockwise
    else:
        return False # counterclockwise

def make_triangle_counterclockwise(triangle):
    if is_clockwise(triangle):
        return triangle[::-1] # reverse the array
    
    return triangle # already counterclockwise

def find_row_index(array, row):
    index = 0
    for r in array:
        if np.array_equal(r, row):
            return index
        
        index += 1
    
    print("Failed to find row")
    return None

def get_all_triangles_with_point(vertex, triangles):
    result = []
    for triangle in triangles:
        if vertex.id in triangle:
            result.append(triangle)
    
    return result

def test_triangle(point, triangle):
    return True

def test_edge(point, end_point):
    return True

def draw_triangle(ax, triangle, color, xs, ys):
    ax.plot([xs[triangle[0]], xs[triangle[1]]], [ys[triangle[0]], ys[triangle[1]]], "-", color=color, linewidth=1)
    ax.plot([xs[triangle[1]], xs[triangle[2]]], [ys[triangle[1]], ys[triangle[2]]], "-", color=color, linewidth=1)
    ax.plot([xs[triangle[2]], xs[triangle[0]]], [ys[triangle[2]], ys[triangle[0]]], "-", color=color, linewidth=1)

def draw_vertex(ax, point):
    ax.plot(point.x, point.y, 'o')

def draw_point(ax, point):
    ax.plot(point[0], point[1], 'o')