"""
Sanding an even (flat) bar with a virtual KUKA KR16 R2010 robot.

The motion imitates hand sanding:
- Approach the bar perpendicular to its surface and touch down gently
- Stroke back and forth along the bar in straight lines
- Shift sideways and stroke the next lane, so the whole bar surface is
  covered with overlap
- Lift off and return home

The start (home) position is the taught "home" pose of the flange in world
coordinates; the TCP "1" is the Sander, as on the real robot.

Geometry (world coordinates, robot base on the table at the origin):
- The center of the sanding area is at (`surface_x`, `surface_y`, `surface_z`)
- The bar axis is turned by `bar_yaw_deg` about the world z-axis
  (0 = along the world x-axis, away from the robot; 90 = along the world y-axis)
- The face is tilted by `tilt_deg` about the bar axis (0 = horizontal, facing up)
- The sanding disc is the robot's TCP "1" (the Sander) and is tilted by
  `pad_angle_deg` against the surface across the bar
  (like a hand-held disc sander; this also keeps the robot wrist out of
  singularities)
"""

import asyncio
import math

import nova
from nova import api, run_program
from nova.actions import cartesian_ptp, linear
from nova.cell import virtual_controller
from nova.types import MotionSettings, Pose
from pydantic import Field

# Taught "home" pose of the flange (TCP 2) in world: x, y, z in mm, rotation vector in rad
HOME_FLANGE_POSE = (477.7, 12.8, 1360.4, 2.8166, -0.0555, 1.2229)
# Offset of the sanding disc from the flange (TCP 1 on the real robot)
SANDER_TCP_POSITION = [-147.1, 0.0, 190.7]
SANDER_TCP_ORIENTATION = [0.0, -0.7854, 0.0]


# Configure the robot program
@nova.program(
    id="start_here",  # Unique identifier of the program. If not provided, the function name will be used.
    name="Sand Bar",  # Readable name of the program
    viewer=nova.viewers.Rerun(),  # add this line for a 3D visualization
    preconditions=nova.ProgramPreconditions(
        controllers=[
            virtual_controller(
                name="hansi2",
                manufacturer=api.models.Manufacturer.KUKA,
                type="kuka-kr16_r2010_2",
            )
        ],
        cleanup_controllers=False,
    ),
)
async def start(
    ctx: nova.ProgramContext,
    count: int = Field(
        default=1, ge=1, le=10, description="Number of sanding passes over the bar"
    ),
    bar_length: float = Field(default=456, gt=0, description="Length of the bar in mm"),
    bar_width: float = Field(
        default=5.2, gt=0, description="Width of the surface to sand in mm"
    ),
    surface_x: float = Field(
        default=874,
        description="Distance of the bar center in front of the robot base in mm",
    ),
    surface_y: float = Field(
        default=186.7, description="Sideways offset of the bar center in mm"
    ),
    surface_z: float = Field(
        default=137.9,
        description="Height of the sanding surface above the robot base in mm",
    ),
    bar_yaw_deg: float = Field(
        default=0,
        description="Direction of the bar axis about the world z-axis in degrees",
    ),
    tilt_deg: float = Field(
        default=0, ge=-90, le=90, description="Tilt of the bar surface in degrees"
    ),
    sanding_pad_width: float = Field(
        default=125, gt=0, description="Diameter of the sanding disc in mm"
    ),
    pad_angle_deg: float = Field(
        default=10,
        ge=-30,
        le=30,
        description="Tilt of the sanding disc against the surface in degrees",
    ),
    strokes_per_lane: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of strokes (forth, back, ...) per lane",
    ),
):
    """Sand the surface of a tilted flat bar."""
    cell = ctx.cell
    controller = await cell.controller("hansi2")
    cycle = ctx.cycle(extra={"app": "visual-studio-code"})

    travel = MotionSettings(tcp_velocity_limit=250)
    approach = MotionSettings(tcp_velocity_limit=50)
    sanding = MotionSettings(tcp_velocity_limit=120)

    motion_group = controller[0]
    tcp = "1"  # Sander, TCP 1 on the real robot
    await motion_group.ensure_virtual_tcp(
        api.models.RobotTcp(
            id=tcp,
            name=tcp,
            position=SANDER_TCP_POSITION,
            orientation=SANDER_TCP_ORIENTATION,
        )
    )
    # The home pose is taught for the flange, express it for the Sander TCP "1"
    home = Pose(HOME_FLANGE_POSE) @ Pose(
        (*SANDER_TCP_POSITION, *SANDER_TCP_ORIENTATION)
    )

    # Surface frame: tool z points into the surface, x runs along the bar, y across it
    surface = (
        Pose((surface_x, surface_y, surface_z, 0, 0, math.radians(bar_yaw_deg)))
        @ Pose((0, 0, 0, math.pi, 0, 0))
        @ Pose((0, 0, 0, math.radians(tilt_deg), 0, 0))
    )
    # Tilt the disc about its contact point
    disc_tilt = Pose((0, 0, 0, math.radians(pad_angle_deg), 0, 0))

    def disc_at(x: float, y: float) -> Pose:
        return surface @ Pose((x, y, 0, 0, 0, 0)) @ disc_tilt

    lift = Pose((0, 0, -50, 0, 0, 0))  # 50 mm off the surface (against tool z)

    # Lanes across the bar width with ~30% overlap of the sanding pad
    usable_width = max(0.0, bar_width - sanding_pad_width)
    lanes = math.ceil(usable_width / (sanding_pad_width * 0.7)) + 1
    lane_ys = [
        -usable_width / 2 + usable_width * i / (lanes - 1) if lanes > 1 else 0.0
        for i in range(lanes)
    ]
    bar_start, bar_end = -bar_length / 2, bar_length / 2

    first_contact = disc_at(bar_start, lane_ys[0])
    actions = [
        cartesian_ptp(home, settings=travel),
        cartesian_ptp(first_contact @ lift, settings=travel),  # Hover over bar start
        linear(first_contact, settings=approach),  # Touch down gently
    ]

    x = bar_start
    for y in lane_ys:
        actions.append(linear(disc_at(x, y), settings=approach))  # Shift sideways
        for _ in range(strokes_per_lane):
            x = bar_end if x == bar_start else bar_start
            actions.append(linear(disc_at(x, y), settings=sanding))

    last_contact = disc_at(x, lane_ys[-1])
    actions += [
        linear(last_contact @ lift, settings=approach),  # Lift off
        cartesian_ptp(home, settings=travel),
    ]

    # Start the cycle
    await cycle.start()

    # Plan the movements (shows in 3D viewer or creates an rrd file)
    joint_trajectory = await motion_group.plan(actions, tcp)

    print(f"Sanding {lanes} lanes over a {bar_length} x {bar_width} mm bar...")
    for i in range(count):
        print(f"Sanding pass {i + 1} of {count}")
        await motion_group.execute(joint_trajectory, tcp, actions=actions)
        print(f"Pass {i + 1} completed")
        await asyncio.sleep(1)

    # Finish the cycle
    await cycle.finish()
    print("Sanding completed!")


if __name__ == "__main__":
    run_program(start)
