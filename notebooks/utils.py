
def cross_2D(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

def square_meters_to_square_miles(area_m2: float) -> float:
    return area_m2 / 1609.34**2