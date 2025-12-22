import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.convert_obstacle_to_meter import (  # noqa: E402
    apply_homography,
    meter_to_pixel_factory,
    pixel_to_meter_factory,
)


def test_pixel_to_meter_origin_maps_to_zero():
    H = np.eye(3)
    origin_px = (0.0, 100.0)
    px_to_m = pixel_to_meter_factory(H, origin_px)
    origin_m = px_to_m(np.array([origin_px], dtype=float))[0]
    np.testing.assert_allclose(origin_m, np.zeros(2), atol=1e-9)


def test_roundtrip_identity_homography():
    H = np.eye(3)
    origin_px = (0.0, 100.0)
    px_to_m = pixel_to_meter_factory(H, origin_px)
    m_to_px = meter_to_pixel_factory(H, origin_px)

    points_px = np.array(
        [
            [0.0, 0.0],
            [10.5, 20.25],
            [50.0, 90.0],
            [123.45, 67.89],
        ]
    )
    meters = px_to_m(points_px)
    roundtrip_px = m_to_px(meters)
    np.testing.assert_allclose(roundtrip_px, points_px, atol=1e-6)


def test_roundtrip_affine_homography_with_translation():
    H = np.array([[2.0, 0.0, 5.0], [0.0, 1.5, -3.0], [0.0, 0.0, 1.0]])
    origin_px = (10.0, 50.0)
    px_to_m = pixel_to_meter_factory(H, origin_px)
    m_to_px = meter_to_pixel_factory(H, origin_px)

    points_px = np.array(
        [
            [0.0, 0.0],
            [25.0, 10.0],
            [80.5, 120.25],
        ]
    )
    meters = px_to_m(points_px)
    roundtrip_px = m_to_px(meters)
    np.testing.assert_allclose(roundtrip_px, points_px, atol=1e-6)
