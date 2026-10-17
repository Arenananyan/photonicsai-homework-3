"""Host-side motion checks independent of the model's instructions."""

import math


def validate_preview(preview):
    """Accept vertical moves and level XY travel above the clearance plane.

    This is a geometric guard, not evidence that a physical path is unobstructed.
    The operator must still review the full preview before execution.
    """
    points = [preview["start"], *preview["targets"]]
    if len(points) < 2:
        raise ValueError("Preview must contain a path")
    for point in points:
        if len(point) != 3 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in point
        ):
            raise ValueError("Preview coordinates must be three finite numbers")
    clearance = max(100.0, points[0][2], points[-1][2])
    for start, end in zip(points, points[1:]):
        xy_changed = any(abs(a - b) > 1e-6 for a, b in zip(start[:2], end[:2]))
        if xy_changed and (
            abs(start[2] - end[2]) > 1e-6 or min(start[2], end[2]) < clearance - 1e-6
        ):
            raise ValueError(
                "Lift vertically to clearance before level XY travel; lower Z afterwards"
            )
    for speed in [preview["speed_mm_s"], *preview.get("step_speeds_mm_s", [])]:
        if (
            type(speed) not in (float, int)
            or not math.isfinite(speed)
            or not 0 < speed <= 30
        ):
            raise ValueError("Motion speed must be finite and within (0, 30] mm/s")
