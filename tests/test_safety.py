import unittest

from hw3.safety import validate_preview


class PreviewSafety(unittest.TestCase):
    def test_lift_traverse_lower(self):
        validate_preview(
            dict(
                start=[0, 0, 5],
                targets=[[0, 0, 100], [10, 10, 100], [10, 10, 20]],
                speed_mm_s=20,
            )
        )

    def test_unsafe_paths_and_speeds(self):
        for targets, speed in [
            ([[10, 10, 100]], 20),
            ([[10, 10, 5]], 20),
            ([[0, 0, 100], [10, 10, 100]], 31),
            ([[0, 0, float("nan")]], 20),
        ]:
            with (
                self.subTest(targets=targets, speed=speed),
                self.assertRaises(ValueError),
            ):
                validate_preview(
                    dict(start=[0, 0, 5], targets=targets, speed_mm_s=speed)
                )

    def test_high_start_requires_high_traverse(self):
        with self.assertRaises(ValueError):
            validate_preview(
                dict(
                    start=[0, 0, 150],
                    targets=[[0, 0, 100], [10, 10, 100]],
                    speed_mm_s=20,
                )
            )
