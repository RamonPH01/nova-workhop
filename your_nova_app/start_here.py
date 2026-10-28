"""
Sanding an even (flat) bar with a virtual KUKA KR16 R2010 robot.

The motion imitates hand sanding:
- Approach the bar perpendicular to its surface and touch down gently
- Stroke back and forth along the bar in straight lines
- Shift sideways and stroke the next lane, so the whole bar surface is
  covered with overlap
- Lift off and return home

Geometry (world coordinates, robot base at the origin):
- The center of the sanding area is at (`surface_x`, 0, `surface_z`)
- The bar runs along the world x-axis, tilted by `tilt_deg` so that it rises
  away from the robot and its surface faces the robot
- The tool always stays perpendicular to the surface
"""

import asyncio
import math

import nova
from nova import api, run_program
from nova.actions import cartesian_ptp, joint_ptp, linear
from nova.cell import virtual_controller
from nova.types import MotionSettings, Pose
from pydantic import Field


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
    count: int = Field(default=1, ge=1, le=10, description="Number of sanding passes over the bar"),
    bar_length: float = Field(default=400, gt=0, description="Length of the bar in mm"),
    bar_width: float = Field(default=60, gt=0, description="Width of the bar in mm"),
    surface_x: float = Field(default=1000, description="Distance of the bar center from the robot base in mm"),
    surface_z: float = Field(default=600, description="Height of the bar center in mm"),
    tilt_deg: float = Field(default=45, ge=-90, le=90, description="Tilt of the bar surface in degrees"),
    sanding_pad_width: float = Field(default=30, gt=0, description="Width of the sanding pad in mm"),
    strokes_per_lane: int = Field(
        default=2, ge=1, le=20, description="Number of strokes (forth, back, ...) per lane"
    ),
):
    """Sand the surface of a tilted flat bar."""
    cell = ctx.cell
    controller = await cell.controller("kuka-kr16-r2010")
    cycle = ctx.cycle(extra={"app": "visual-studio-code"})

    travel = MotionSettings(tcp_velocity_limit=250)
    approach = MotionSettings(tcp_velocity_limit=50)
    sanding = MotionSettings(tcp_velocity_limit=120)

    motion_group = controller[0]
    home_joints = await motion_group.joints()
    tcp_names = await motion_group.tcp_names()
    tcp = tcp_names[0]

    # Surface frame: tool z points into the surface, x runs along the bar, y across it
    surface = Pose((surface_x, 0, surface_z, math.pi, 0, 0)) @ Pose(
        (0, 0, 0, 0, math.radians(tilt_deg), 0)
    )
    lift = Pose((0, 0, -50, 0, 0, 0))  # 50 mm off the surface (against tool z)

    # Lanes across the bar width with ~30% overlap of the sanding pad
    usable_width = max(0.0, bar_width - sanding_pad_width)
    lanes = math.ceil(usable_width / (sanding_pad_width * 0.7)) + 1
    lane_ys = [
        -usable_width / 2 + usable_width * i / (lanes - 1) if lanes > 1 else 0.0
        for i in range(lanes)
    ]
    bar_start, bar_end = -bar_length / 2, bar_length / 2

    first_contact = surface @ Pose((bar_start, lane_ys[0], 0, 0, 0, 0))
    actions = [
        joint_ptp(home_joints, settings=travel),
        cartesian_ptp(first_contact @ lift, settings=travel),  # Hover over bar start
        linear(first_contact, settings=approach),  # Touch down gently
    ]

    x = bar_start
    for y in lane_ys:
        actions.append(linear(surface @ Pose((x, y, 0, 0, 0, 0)), settings=approach))  # Shift sideways
        for _ in range(strokes_per_lane):
            x = bar_end if x == bar_start else bar_start
            actions.append(linear(surface @ Pose((x, y, 0, 0, 0, 0)), settings=sanding))

    last_contact = actions[-1].target
    actions += [
        linear(last_contact @ lift, settings=approach),  # Lift off
        joint_ptp(home_joints, settings=travel),
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
