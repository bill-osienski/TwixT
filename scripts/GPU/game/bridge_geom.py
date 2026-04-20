"""
Geometry primitives for TwixT knight-edge intersection.

Separated to avoid circular imports between bridge.py and edge_index.py.
This module has NO dependencies on other game/ modules.
"""


def _orient(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> int:
    """
    Orientation test for three points.

    Returns:
        1 if CCW (counter-clockwise)
       -1 if CW (clockwise)
        0 if collinear
    """
    v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return 1 if v > 0 else (-1 if v < 0 else 0)


def _proper_intersect_knight(
    x1: int, y1: int, x2: int, y2: int,
    x3: int, y3: int, x4: int, y4: int
) -> bool:
    """
    Fast proper intersection test for TwixT knight-edges.

    For knight-move segments (delta +/-1,+/-2 or +/-2,+/-1):
    - Collinear overlaps cannot happen between distinct knight-edges
    - No interior lattice points exist on knight-edge segments (gcd(1,2)=1)
    - So we only need the pure orientation test

    Args:
        x1, y1, x2, y2: First segment endpoints (x=col, y=row)
        x3, y3, x4, y4: Second segment endpoints (x=col, y=row)

    Returns:
        True if segments properly cross (not just touch at endpoints).
    """
    o1 = _orient(x1, y1, x2, y2, x3, y3)
    o2 = _orient(x1, y1, x2, y2, x4, y4)
    # If either endpoint of seg2 is collinear with seg1, or both on same side: no crossing
    if o1 == 0 or o2 == 0 or o1 == o2:
        return False

    o3 = _orient(x3, y3, x4, y4, x1, y1)
    o4 = _orient(x3, y3, x4, y4, x2, y2)
    # Same check for seg1 endpoints relative to seg2
    if o3 == 0 or o4 == 0 or o3 == o4:
        return False

    return True
