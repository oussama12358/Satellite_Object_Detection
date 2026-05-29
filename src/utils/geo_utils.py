"""
Geospatial Utilities for Satellite Imagery
============================================
Handles coordinate transformations between pixel space, normalized space,
and real-world geographic coordinates (WGS-84, UTM).

Used for:
    - Georeferencing detections (lat/lon output for GIS tools)
    - Scaling detections from tile coordinates back to full image
    - Computing real-world object dimensions from GSD (Ground Sampling Distance)
"""

import math
from typing import Tuple, List, Dict
import numpy as np


# Earth radius in meters
EARTH_RADIUS_M = 6_371_000.0


def pixel_to_latlon(
    px: float, py: float,
    img_width: int, img_height: int,
    top_left_lon: float, top_left_lat: float,
    bottom_right_lon: float, bottom_right_lat: float,
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to WGS-84 lat/lon using linear interpolation.

    Assumes the image has been orthorectified (no significant distortion).

    Args:
        px, py: Pixel coordinates (x=col, y=row)
        img_width, img_height: Image dimensions
        top_left_lon/lat: Top-left corner geographic coords
        bottom_right_lon/lat: Bottom-right corner geographic coords

    Returns:
        (longitude, latitude) in decimal degrees
    """
    lon = top_left_lon + (px / img_width) * (bottom_right_lon - top_left_lon)
    lat = top_left_lat + (py / img_height) * (bottom_right_lat - top_left_lat)
    return lon, lat


def latlon_to_pixel(
    lon: float, lat: float,
    img_width: int, img_height: int,
    top_left_lon: float, top_left_lat: float,
    bottom_right_lon: float, bottom_right_lat: float,
) -> Tuple[float, float]:
    """
    Convert WGS-84 lat/lon to pixel coordinates.

    Returns:
        (px, py) pixel coordinates
    """
    px = (lon - top_left_lon) / (bottom_right_lon - top_left_lon) * img_width
    py = (lat - top_left_lat) / (bottom_right_lat - top_left_lat) * img_height
    return px, py


def haversine_distance(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
) -> float:
    """
    Compute great-circle distance between two geographic points (meters).

    Args:
        lon1, lat1: First point in decimal degrees
        lon2, lat2: Second point in decimal degrees

    Returns:
        Distance in meters
    """
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def compute_gsd(
    altitude_m: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
) -> float:
    """
    Compute Ground Sampling Distance (GSD) in meters/pixel.

    GSD determines the real-world size of each pixel.
    Critical for sizing analysis: a 10px ship at 0.5m/px GSD = 5m width.

    Args:
        altitude_m: Platform altitude above ground (meters)
        focal_length_mm: Camera focal length (mm)
        sensor_width_mm: Camera sensor width (mm)
        image_width_px: Image width in pixels

    Returns:
        GSD in meters per pixel
    """
    gsd = (altitude_m * sensor_width_mm) / (focal_length_mm * image_width_px)
    return gsd


def pixel_bbox_to_realworld(
    x1: float, y1: float, x2: float, y2: float,
    gsd_m_per_px: float,
) -> Dict[str, float]:
    """
    Convert pixel bounding box dimensions to real-world measurements.

    Args:
        x1, y1, x2, y2: Pixel coordinates
        gsd_m_per_px: Ground Sampling Distance

    Returns:
        Dict with width_m, height_m, area_m2, length_m (max dim)
    """
    width_px = x2 - x1
    height_px = y2 - y1
    width_m = width_px * gsd_m_per_px
    height_m = height_px * gsd_m_per_px
    return {
        "width_m": round(width_m, 2),
        "height_m": round(height_m, 2),
        "area_m2": round(width_m * height_m, 2),
        "length_m": round(max(width_m, height_m), 2),  # longest dimension
    }


def classify_ship_size(length_m: float) -> str:
    """
    Classify ship size category by length (meters).
    Based on Lloyd's Register maritime classification.
    """
    if length_m < 20:
        return "small-craft"
    elif length_m < 50:
        return "small-vessel"
    elif length_m < 100:
        return "medium-vessel"
    elif length_m < 200:
        return "large-vessel"
    elif length_m < 300:
        return "very-large-vessel"
    else:
        return "ultra-large-vessel"


def classify_aircraft_size(length_m: float) -> str:
    """Classify aircraft by estimated length."""
    if length_m < 10:
        return "UAV/small-aircraft"
    elif length_m < 20:
        return "light-aircraft"
    elif length_m < 40:
        return "regional-jet"
    elif length_m < 65:
        return "narrow-body"
    else:
        return "wide-body"


def tile_pixel_to_full_image(
    tile_x1: float, tile_y1: float, tile_x2: float, tile_y2: float,
    tile_origin_x: int, tile_origin_y: int,
) -> Tuple[float, float, float, float]:
    """
    Translate bounding box from tile-local pixel coords to full-image coords.

    Args:
        tile_x1/y1/x2/y2: Box in tile pixel space
        tile_origin_x/y: Top-left corner of tile in full image

    Returns:
        (x1, y1, x2, y2) in full image pixel coordinates
    """
    return (
        tile_x1 + tile_origin_x,
        tile_y1 + tile_origin_y,
        tile_x2 + tile_origin_x,
        tile_y2 + tile_origin_y,
    )


def compute_detection_density(
    detections: List[Dict],
    img_width: int,
    img_height: int,
    grid_size: int = 10,
) -> np.ndarray:
    """
    Compute spatial detection density heatmap (useful for threat analysis).

    Args:
        detections: List of dicts with x1, y1, x2, y2 keys
        img_width, img_height: Image dimensions
        grid_size: Number of grid cells per dimension

    Returns:
        (grid_size × grid_size) density array (normalized 0-1)
    """
    density = np.zeros((grid_size, grid_size), dtype=np.float32)

    for det in detections:
        cx = (det["x1"] + det["x2"]) / 2
        cy = (det["y1"] + det["y2"]) / 2
        gx = int(cx / img_width * grid_size)
        gy = int(cy / img_height * grid_size)
        gx = min(gx, grid_size - 1)
        gy = min(gy, grid_size - 1)
        density[gy, gx] += 1

    if density.max() > 0:
        density /= density.max()

    return density
