import numpy as np
import pyvista as pv
from TIN_types import Triangle

def draw_triangle(ax, triangle, color, xs, ys): 
    ax.plot([xs[triangle[0]], xs[triangle[1]]], [ys[triangle[0]], ys[triangle[1]]], "-", color=color, linewidth=1)
    ax.plot([xs[triangle[1]], xs[triangle[2]]], [ys[triangle[1]], ys[triangle[2]]], "-", color=color, linewidth=1)
    ax.plot([xs[triangle[2]], xs[triangle[0]]], [ys[triangle[2]], ys[triangle[0]]], "-", color=color, linewidth=1)

# This function was generated with AI (Copilot Code Completion)
# The function draws a triangle given a Triangle object
def draw_triangle_object(ax, triangle: Triangle, color, linewidth=1, filled=False):
    if filled:
        ax.fill([triangle.v1.x, triangle.v2.x, triangle.v3.x],
                [triangle.v1.y, triangle.v2.y, triangle.v3.y],
                color=color, alpha=0.3)
    ax.plot([triangle.v1.x, triangle.v2.x], [triangle.v1.y, triangle.v2.y], "-", color=color, linewidth=linewidth)
    ax.plot([triangle.v2.x, triangle.v3.x], [triangle.v2.y, triangle.v3.y], "-", color=color, linewidth=linewidth)
    ax.plot([triangle.v3.x, triangle.v1.x], [triangle.v3.y, triangle.v1.y], "-", color=color, linewidth=linewidth)

def draw_vertex(ax, point, markersize=1):
    ax.plot(point.x, point.y, 'o', markersize=markersize)

def draw_point(ax, point, markersize=1):
    ax.plot(point[0], point[1], 'o', markersize=markersize)

# where edge is a tuple of two vertices
def draw_line(ax, edge, color, linewidth=2):
    ax.plot([edge[0].x, edge[1].x], [edge[0].y, edge[1].y], "-", color=color, linewidth=linewidth)

def draw_line_points(ax, p1, p2, color, linewidth=2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "-", color=color, linewidth=linewidth)


# triangles: is a n by 3 array of triangle vertex indices
def get_mesh_from_triangles(triangles, vertices_3d):
    faces = np.hstack([np.full((triangles.shape[0], 1), 3, dtype=np.int32), triangles]).flatten()
    mesh = pv.PolyData(vertices_3d, faces)

    return mesh