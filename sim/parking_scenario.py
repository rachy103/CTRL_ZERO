"""Perpendicular-parking scenario geometry for the contest parking mission.

All coordinates are world meters in the rescaled map frame
(map 15.085 x 10.444 m, origin at the center). Bay geometry was measured
on the generated track texture and scaled by the calibrated map factor;
the derivation is recorded in ``config/sim_config.json`` under
``provenance.parking_scenario``.

Contest rules (from the user-provided specification):
* Obstacle cars occupy either bays 2 and 4 (case 1) or bays 1 and 3
  (case 2), numbered west to east; the vehicle must park between them.
* The start position is one of four 3 cm markers in a 30 cm grid near
  the entry line; the chosen marker sits under the right front wheel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from .lidar_sim import ObstacleRect
except ImportError:
    from lidar_sim import ObstacleRect


# Texture-derived geometry, already scaled to the rescaled world frame.
BAY_DIVIDER_X_M = (-0.8727, 0.0685, 1.0095, 1.9541, 2.8975)
BAY_TOP_Y_M = 1.5317
BAY_MOUTH_Y_M = 0.2205
AISLE_SOUTH_Y_M = -1.4360
ENTRY_LINE_X_M = 3.8793

# Start markers measured from the map artwork itself: four ~3 cm white
# squares are drawn beside the IN text. Numbered per the contest sheet
# with the vehicle heading west: 1/2 = row closer to the entry line,
# 1/3 = vehicle-left (south) column, 2/4 = vehicle-right (north) column.
MARKER_POSITIONS_M = {
    1: (4.121, -0.749),
    2: (4.121, -0.461),
    3: (4.433, -0.749),
    4: (4.433, -0.461),
}

# Obstacle cars use the own-vehicle footprint (user authorization).
OBSTACLE_LENGTH_M = 0.99
OBSTACLE_WIDTH_M = 0.55

OBSTACLE_BAYS_BY_CASE = {1: (2, 4), 2: (1, 3)}
TARGET_BAY_BY_CASE = {1: 3, 2: 2}


@dataclass(frozen=True)
class StartPose:
    """Rear-axle world pose derived from a start marker."""

    rear_axle_x_m: float
    rear_axle_y_m: float
    yaw_rad: float


def bay_center_x_m(bay_number: int) -> float:
    if not 1 <= bay_number <= 4:
        raise ValueError(f"Bay number must be 1..4, got {bay_number}")
    return (
        BAY_DIVIDER_X_M[bay_number - 1] + BAY_DIVIDER_X_M[bay_number]
    ) / 2.0


def bay_center_y_m() -> float:
    return (BAY_TOP_Y_M + BAY_MOUTH_Y_M) / 2.0


def obstacle_rects(case: int) -> list[ObstacleRect]:
    """Obstacle car footprints parked nose-north, centered in their bays."""
    if case not in OBSTACLE_BAYS_BY_CASE:
        raise ValueError(f"Obstacle case must be 1 or 2, got {case}")
    rects = []
    for bay in OBSTACLE_BAYS_BY_CASE[case]:
        rects.append(
            ObstacleRect(
                center_x=bay_center_x_m(bay),
                center_y=bay_center_y_m(),
                length_m=OBSTACLE_LENGTH_M,
                width_m=OBSTACLE_WIDTH_M,
                yaw_rad=math.pi / 2.0,
            )
        )
    return rects


def marker_position_m(marker: int) -> tuple[float, float]:
    """World position of start marker 1..4 as drawn on the map."""
    if marker not in MARKER_POSITIONS_M:
        raise ValueError(f"Start marker must be 1..4, got {marker}")
    return MARKER_POSITIONS_M[marker]


def start_pose(
    marker: int,
    wheelbase_m: float,
    front_track_m: float,
) -> StartPose:
    """Rear-axle pose with the right front wheel over the marker, heading west."""
    marker_x, marker_y = marker_position_m(marker)
    yaw = math.pi  # heading west (-x)
    # Right front wheel in the vehicle frame: (wheelbase, -front_track/2).
    # heading west: forward = (-1, 0), left = (0, -1).
    wheel_offset_x = -wheelbase_m
    wheel_offset_y = front_track_m / 2.0
    return StartPose(
        rear_axle_x_m=marker_x - wheel_offset_x,
        rear_axle_y_m=marker_y - wheel_offset_y,
        yaw_rad=yaw,
    )


def timed_plan(
    marker: int,
    *,
    wheelbase_m: float,
    front_track_m: float,
    max_steer_deg: float,
    v_forward_mps: float,
    v_reverse_mps: float,
    lidar_forward_m: float,
    arc_end_rear_axle_y_m: float = 0.7,
    sector_half_angle_deg: float = 5.0,
    trigger_latency_s: float = 0.0,
    approach_trim_s: float = 0.0,
) -> dict[str, float]:
    """Closed-form two-arc plan: swing out left, reverse right into the gap.

    Both arcs use full steering lock (radius R = wheelbase / tan(max
    steer)). The rotation split follows from requiring the reverse arc to
    end heading south at ``arc_end_rear_axle_y_m`` with theta1 + theta2 =
    90 deg; the approach distance places the maneuver so the arc ends on
    the centerline of the bay one pitch west of the first detected car.
    """
    marker_x, marker_y = marker_position_m(marker)
    start_y = marker_y - front_track_m / 2.0  # rear-axle centerline
    radius = wheelbase_m / math.tan(math.radians(max_steer_deg))

    cos_theta1 = ((arc_end_rear_axle_y_m - start_y) / radius + 1.0) / 2.0
    if not -1.0 < cos_theta1 < 1.0:
        raise ValueError(
            f"Arc end y={arc_end_rear_axle_y_m} is unreachable from start "
            f"y={start_y:.3f} at radius {radius:.3f}"
        )
    theta1 = math.acos(cos_theta1)
    theta2 = math.pi / 2.0 - theta1

    pitch = (BAY_DIVIDER_X_M[-1] - BAY_DIVIDER_X_M[0]) / (
        len(BAY_DIVIDER_X_M) - 1
    )
    obstacle_south_face_y = bay_center_y_m() - OBSTACLE_LENGTH_M / 2.0
    lateral_to_face = obstacle_south_face_y - start_y
    # SEARCH counts both parked cars; the maneuver is timed from the moment
    # the SECOND car's east edge enters the +/-5 deg right abeam cone of the
    # front-mounted lidar. The target bay lies one pitch east of that car,
    # which keeps the dead-reckoned approach segment short.
    trigger_rear_axle_ahead_of_second_car = (
        OBSTACLE_WIDTH_M / 2.0
        + math.tan(math.radians(sector_half_angle_deg)) * lateral_to_face
        + lidar_forward_m
    )
    swing_start_offset_from_target = radius * (2.0 * math.sin(theta1) - 1.0)
    approach_m = (
        trigger_rear_axle_ahead_of_second_car
        - pitch
        - swing_start_offset_from_target
    )
    if approach_m < 0.0:
        raise ValueError(
            f"Second-car trigger leaves a negative approach ({approach_m:.3f} m); "
            "the swing would have to start before the trigger fires."
        )

    return {
        "theta1_deg": math.degrees(theta1),
        "theta2_deg": math.degrees(theta2),
        "radius_m": radius,
        "approach_s": (
            approach_m / v_forward_mps - trigger_latency_s + approach_trim_s
        ),
        "swing_s": radius * theta1 / v_forward_mps,
        "arc_s": radius * theta2 / v_reverse_mps,
        "start_rear_axle_y_m": start_y,
    }


def target_bay_rect(case: int) -> tuple[float, float, float, float]:
    """Target bay as (x_min, x_max, y_min, y_max)."""
    bay = TARGET_BAY_BY_CASE[case]
    return (
        BAY_DIVIDER_X_M[bay - 1],
        BAY_DIVIDER_X_M[bay],
        BAY_MOUTH_Y_M,
        BAY_TOP_Y_M,
    )
