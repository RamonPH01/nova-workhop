"""
Sanding an even (flat) bar with a virtual KUKA KR16 R1610 robot.

The motion imitates hand sanding:
- Approach the bar perpendicular to its surface and touch down gently
- Stroke back and forth along the bar in straight lines
- Shift sideways and stroke the next lane, so the whole bar surface is
  covered with overlap
- Lift off, swing the disc over to the other side of the bar, touch down
  again and sand that side the same way
- Lift off and return home

The start (home) position is the taught "home" pose of the flange in world
coordinates. The robot moves there in a straight joint move from wherever it
is, and returns there at the end. Home and the hover pose over the bar are
fixed joint positions with the wrist not flipped (A5 > 0, A4 and A6 as close
to 0 as possible), so the wrist cannot pick a twisted configuration. Before
the sanding cycle runs, the program checks that A4 and A6 turn by no more
than `max_wrist_turn_deg` and stops otherwise.

Geometry (world coordinates):
- The robot base sits at (`base_x`, `base_y`, `base_z`), turned by `base_yaw_deg`
  about the world z-axis. All 0 = robot base at the world origin; replace with
  the base position measured on the real robot
- The center of the sanding area is at (`surface_x`, `surface_y`, `surface_z`)
- The bar axis is turned by `bar_yaw_deg` about the world z-axis
  (0 = along the world x-axis, away from the robot; 90 = along the world y-axis)
- The face is tilted by `tilt_deg` about the bar axis (0 = horizontal, facing up)
- The sanding disc sits `tool_length` mm in front of the flange on the flange
  axis and is tilted by `pad_angle_deg` against the surface across the bar,
  to one side first and mirrored to the other side after that
  (like a hand-held disc sander)
- The disc also leans by `lean_deg` along the bar. Without it the disc axis is
  almost in line with the forearm (wrist singularity), and tilting the disc
  over to the second side would make A4 and A6 spin about 180°
"""

import asyncio
import math

import nova
from nova import api, run_program
from nova.actions import joint_ptp, linear
from nova.cell import virtual_controller
from nova.types import MotionSettings, Pose
from pydantic import Field

# Taught "home" pose of the flange in world: x, y, z in mm, rotation vector in rad
HOME_FLANGE_POSE = (477.7, 12.8, 1360.4, 2.8166, -0.0555, 1.2229)


# Configure the robot program
@nova.program(
    id="start_here",  # Unique identifier of the program. If not provided, the function name will be used.
    name="Sand Bar",  # Readable name of the program
    viewer=nova.viewers.Rerun(),  # add this line for a 3D visualization
    preconditions=nova.ProgramPreconditions(
        controllers=[
            virtual_controller(
                name="kukar16",
                manufacturer=api.models.Manufacturer.KUKA,
                type="kuka-kr16_r1610_2",
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
    base_x: float = Field(
        default=0, description="Robot base position in the world, x in mm"
    ),
    base_y: float = Field(
        default=0, description="Robot base position in the world, y in mm"
    ),
    base_z: float = Field(
        default=0, description="Robot base position in the world, z in mm"
    ),
    base_yaw_deg: float = Field(
        default=0,
        description="Rotation of the robot base about the world z-axis in degrees",
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
    tool_length: float = Field(
        default=220,
        ge=0,
        description="Distance from flange to sanding disc surface in mm",
    ),
    pad_angle_deg: float = Field(
        default=10,
        ge=-30,
        le=30,
        description="Tilt of the sanding disc against the surface in degrees, mirrored for the second side",
    ),
    lean_deg: float = Field(
        default=-20,
        ge=-45,
        le=45,
        description="Lean of the sanding disc along the bar in degrees, keeps the wrist out of singularities",
    ),
    strokes_per_lane: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of strokes (forth, back, ...) per lane",
    ),
    max_wrist_turn_deg: float = Field(
        default=120,
        gt=0,
        le=360,
        description="Maximum range A4 and A6 may turn during the sanding cycle in degrees",
    ),
):
    """Sand both sides of a tilted flat bar."""
    cell = ctx.cell
    controller = await cell.controller("kukar16")
    cycle = ctx.cycle(extra={"app": "visual-studio-code"})

    travel = MotionSettings(tcp_velocity_limit=80)
    approach = MotionSettings(tcp_velocity_limit=80)
    sanding = MotionSettings(tcp_velocity_limit=80)

    motion_group = controller[0]
    tcp_names = await motion_group.tcp_names()
    tcp = tcp_names[0]

    setup = await motion_group.get_setup(tcp)
    limits = [(j.position.lower_limit, j.position.upper_limit) for j in setup.global_limits.joints]

    async def joint_solutions(pose: Pose) -> list[tuple[float, ...]]:
        # Every joint solution for the pose, including A4/A6 turned by ±360° within the joint limits
        (solutions,) = await motion_group._inverse_kinematics([pose], tcp)
        turns = (-2 * math.pi, 0, 2 * math.pi)
        candidates = [
            tuple(j + (d4 if i == 3 else d6 if i == 5 else 0) for i, j in enumerate(s))
            for s in solutions
            for d4 in turns
            for d6 in turns
        ]
        candidates = [c for c in candidates if all(lo <= j <= hi for j, (lo, hi) in zip(c, limits))]
        if not candidates:
            raise ValueError(f"No joint solution for {pose}")
        return candidates

    robot_base = Pose((base_x, base_y, base_z, 0, 0, math.radians(base_yaw_deg)))
    # Home with the wrist not flipped and A4/A6 as untwisted as possible
    home_joints = min(
        await joint_solutions(~robot_base @ Pose(HOME_FLANGE_POSE)),
        key=lambda j: (j[4] <= 0, abs(j[3]) + abs(j[5])),
    )
    # Surface frame in robot base coordinates: tool z points into the surface, x runs along the bar, y across it
    surface = (
        ~robot_base
        @ Pose((surface_x, surface_y, surface_z, 0, 0, math.radians(bar_yaw_deg)))
        @ Pose((0, 0, 0, math.pi, 0, 0))
        @ Pose((0, 0, 0, math.radians(tilt_deg), 0, 0))
    )

    def flange_at(x: float, y: float, pad_deg: float) -> Pose:
        # Tilt the disc about its contact point; the robot moves the flange, so keep it tool_length behind the disc
        return (
            surface
            @ Pose((x, y, 0, 0, 0, 0))
            @ Pose((0, 0, 0, math.radians(pad_deg), 0, 0))
            @ Pose((0, 0, 0, 0, math.radians(lean_deg), 0))
            @ Pose((0, 0, -tool_length, 0, 0, 0))
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

    # Hover over the bar in the wrist configuration closest to home
    hover_joints = min(
        await joint_solutions(flange_at(bar_start, lane_ys[0], pad_angle_deg) @ lift),
        key=lambda j: (j[4] <= 0, sum(abs(a - b) for a, b in zip(j, home_joints))),
    )

    actions = [joint_ptp(home_joints, settings=travel)]
    x = bar_start
    for side, (pad_deg, ys) in enumerate(
        ((pad_angle_deg, lane_ys), (-pad_angle_deg, lane_ys[::-1]))
    ):
        contact = flange_at(x, ys[0], pad_deg)
        if side == 0:
            actions.append(joint_ptp(hover_joints, settings=travel))  # Hover over the bar
        else:
            actions.append(linear(contact @ lift, settings=approach))  # Just tilt the disc over
        actions.append(linear(contact, settings=approach))  # Touch down gently
        for y in ys:
            actions.append(
                linear(flange_at(x, y, pad_deg), settings=approach)
            )  # Shift sideways
            for _ in range(strokes_per_lane):
                x = bar_end if x == bar_start else bar_start
                actions.append(linear(flange_at(x, y, pad_deg), settings=sanding))
        actions.append(linear(actions[-1].target @ lift, settings=approach))  # Lift off

    actions.append(joint_ptp(home_joints, settings=travel))

    # Plan the sanding cycle from home, so every pass starts and ends at the same joints
    joint_trajectory = await motion_group.plan(actions, tcp, start_joint_position=home_joints)

    # Stop before moving if the wrist would twist too much
    for axis in (3, 5):
        turn = math.degrees(
            max(j[axis] for j in joint_trajectory.joint_positions)
            - min(j[axis] for j in joint_trajectory.joint_positions)
        )
        if turn > max_wrist_turn_deg:
            raise RuntimeError(
                f"A{axis + 1} would turn by {turn:.0f}° (limit {max_wrist_turn_deg:.0f}°), not moving"
            )

    # Start the cycle
    await cycle.start()

    # Move to home from wherever the robot is
    to_home = [joint_ptp(home_joints, settings=travel)]
    await motion_group.execute(await motion_group.plan(to_home, tcp), tcp, actions=to_home)

    print(
        f"Sanding {lanes} lanes on both sides of a {bar_length} x {bar_width} mm bar..."
    )
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
