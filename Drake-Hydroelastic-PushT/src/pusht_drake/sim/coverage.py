"""Exact T-on-goal coverage: the Push-T reward, in world meters.

    coverage(pose, goal_pose) = area( T @ pose  intersect  T @ goal_pose ) / area(T)

This reproduces the semantics of the gym-pusht reward exactly. The name "pixel
coverage" is misleading: gym_pusht never rasterized anything. Its _get_coverage
built shapely polygons from the block's two rectangles and took the exact
intersection area over the goal area (gym_pusht/envs/pusht.py:232-238). Same
ratio here, congruent T against congruent T, so the normalizer is
simultaneously the goal area and the block area. It is not IoU.

Implementation: the T is exactly two interior-disjoint rectangles (crossbar +
stem, pusht_drake.sim.tblock), so with current rects R_i and goal rects G_j

    area(T_now intersect T_goal) = sum_{i,j} area(R_i intersect G_j)   (4 convex pairs)

each an exact convex-convex intersection: Sutherland-Hodgman clip + shoelace.
Pure numpy, exact to float precision, microseconds per call, with no shapely,
no pymunk and no rendering. Import pulls neither pydrake nor torch (tblock is a
numpy-only module).

Measured sensitivity near the goal, for sanity checks (the T is a
millimetre-for-pixel replica of the canonical 512-px Push-T task, so the
numbers transfer directly): about 1.9 coverage-points per mm of translation and
3.0 points per degree of rotation.
"""

from __future__ import annotations

import numpy as np

from pusht_drake.sim.tblock import PUSHT_T

__all__ = ["COVERAGE_DENOMINATOR_M2", "coverage", "t_rectangles"]


def _body_frame_rectangles() -> tuple[np.ndarray, np.ndarray]:
    """The two rectangles as CCW (4, 2) corner arrays in the body frame.

    primitive_boxes() is in the pickle frame (origin = crossbar centre);
    the slider pose is reported in the body frame (origin = area centroid), so
    shift by -com_offset. Derived from TBlock fields, never re-typed numbers.
    """
    corners = []
    for box in PUSHT_T.primitive_boxes():
        half_w = box["size"][0] / 2.0
        half_h = box["size"][1] / 2.0
        centre_y = box["transform"][1, 3] - PUSHT_T.com_offset_y
        corners.append(
            np.array(
                [
                    [-half_w, centre_y - half_h],
                    [half_w, centre_y - half_h],
                    [half_w, centre_y + half_h],
                    [-half_w, centre_y + half_h],
                ]
            )
        )
    return corners[0], corners[1]


_CROSSBAR, _STEM = _body_frame_rectangles()

#: area(T) in m^2: the constant denominator (6300 px^2 in gym-pusht).
COVERAGE_DENOMINATOR_M2 = float(PUSHT_T.area)


def t_rectangles(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The T's two rectangles as world-frame (4, 2) corner arrays at pose."""
    x, y, theta = np.asarray(pose, dtype=np.float64).reshape(3)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    shift = np.array([x, y])
    return _CROSSBAR @ rot.T + shift, _STEM @ rot.T + shift


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip convex polygon subject by convex CCW clip."""
    output = subject
    n = len(clip)
    for i in range(n):
        if len(output) == 0:
            return output
        a = clip[i]
        b = clip[(i + 1) % n]
        edge = b - a
        # inside = left of the directed edge (CCW polygon interior)
        offsets = (output - a) @ np.array([-edge[1], edge[0]])
        keep = offsets >= 0.0
        clipped = []
        m = len(output)
        for j in range(m):
            k = (j + 1) % m
            p, q = output[j], output[k]
            if keep[j]:
                clipped.append(p)
                if not keep[k]:
                    t = offsets[j] / (offsets[j] - offsets[k])
                    clipped.append(p + t * (q - p))
            elif keep[k]:
                t = offsets[j] / (offsets[j] - offsets[k])
                clipped.append(p + t * (q - p))
        output = np.array(clipped) if clipped else np.empty((0, 2))
    return output


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace area of a CCW polygon (0.0 for fewer than 3 vertices)."""
    if len(vertices) < 3:
        return 0.0
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def coverage(pose: np.ndarray, goal_pose: np.ndarray) -> float:
    """Fraction of the goal-posed T covered by the T at pose, in [0, 1].

    Both poses are (x, y, theta) about the T's area centroid, world meters.
    1.0 exactly iff the poses coincide mod 2pi (the T has no rotational
    symmetry, so the maximizer is unique).
    """
    now = t_rectangles(pose)
    goal = t_rectangles(goal_pose)
    intersection = 0.0
    for rect in now:
        for goal_rect in goal:
            intersection += _polygon_area(_clip_convex(rect, goal_rect))
    return intersection / COVERAGE_DENOMINATOR_M2
