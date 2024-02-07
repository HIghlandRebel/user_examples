# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.examples.base_sample import BaseSample
# Can be used to create a new cube or to point to an already existing cube in stage.

import asyncio
import numpy as np

# import omni.physx as _physx
import omni
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import VisualCylinder
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper_Properties
from omni.isaac.dynamic_control import _dynamic_control

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class NexeraGripperTest(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        # self.ur10s_view = ArticulationView(prim_paths_expr="/World/ur10e_1", name="self.ur10s_view")
        return

    def setup_scene(self):

        # Colors to represent when gripper is open or closed
        self.color_closed = Gf.Vec3f(1.0, 0.2, 0.2)
        self.color_open = Gf.Vec3f(0.2, 1.0, 0.2)
        self.count = 0

        world = self.get_world()

        world.scene.add_default_ground_plane()

        self.box_item = world.scene.add(

            DynamicCuboid(

                prim_path="/World/box_item",

                name="box_item",

                position=np.array([0.7, 0.0, 0.05]),

                scale=np.array([.1, .1, .1]),

                color=np.array([0, 0, 1.0]),

                )
            )
        
        
        
        asset_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"

        add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur10e_1")

        # batch process articulations via an ArticulationView

        self.ur10s_view = ArticulationView(prim_paths_expr="/World/ur10e_1", name="self.ur10s_view")

        world.scene.add(self.ur10s_view)

        # set robot position
        new_positions = np.array([[0, 0, 0.5]])

        # set the joint positions for each articulation

        self.ur10s_view.set_world_poses(positions=new_positions)

        self.ur10s_view.set_joint_positions(np.array([[3.379,-2.0,-2.142,-0.5753, 1.566, 0.0]]))


        # add gripper shell to the scene
        self.gripperGeom = world.scene.add(

            VisualCylinder(prim_path="/World/ur10e_1/tool0/gripper",

                translation=np.array([0.0, 0.0, 0.25]),
                radius = 0.05,
                height = 0.5,

                color=np.array(self.color_open))
            )

        # setup dynamic control
        self.dc = _dynamic_control.acquire_dynamic_control_interface()

        # Gripper properties
        self.sgp = Surface_Gripper_Properties()
        self.sgp.d6JointPath = "/World/ur10e_1/tool0/surfaceGripper"

        self.sgp.parentPath = "/surfaceGripper"                # Set the Cone as the parent object
        # self.sgp.offset = dc.Transform()
        self.sgp.offset.p.x = 0
        self.sgp.offset.p.z = -0.1001
        self.sgp.offset.r = [0.7071, 0, 0.7071, 0]  # Rotate to point gripper in Z direction
        self.sgp.gripThreshold = 0.02
        self.sgp.forceLimit = 1.0e2
        self.sgp.torqueLimit = 1.0e3
        self.sgp.bendAngle = np.pi / 4
        self.sgp.stiffness = 1.0e4
        self.sgp.damping = 1.0e3
        self.sgp.retryClose = False                         # If set to True, surface gripper will keep trying to close until it picks up an object

        self.surface_gripper = Surface_Gripper(self.dc)
        self.surface_gripper.initialize(self.sgp)
        
        omni.timeline.get_timeline_interface().play()
        task = asyncio.ensure_future(omni.kit.app.get_app().next_update_async())
        # print("Pause simulation...")
        # asyncio.ensure_future(pause_sim(task))

        # self.surface_gripper.update()                       # On every sim step, update the gripper status

        # self._physx_subs = _physx.get_physx_interface().subscribe_physics_step_events(self._on_simulation_step)
        # self._timeline.play()
        return


    async def setup_post_load(self):

        self._world = self.get_world()

        self._cube = self._world.scene.get_object("box_item")



        self._world.add_physics_callback("sim_step", callback_fn=self.print_cube_info) #callback names have to be unique
        # self._world.add_physics_callback("sim_step2", callback_fn=self.gripper_status) #callback names have to be unique
        # self._world.add_physics_callback("sim_step3", callback_fn=self.robot_mover) #callback names have to be unique

        return


    # step_size as an argument

    def print_cube_info(self, step_size):

        position, orientation = self._cube.get_world_pose()

        linear_velocity = self._cube.get_linear_velocity()

        # will be shown on terminal

        print("Cube position is : " + str(position))

        print("Cube's orientation is : " + str(orientation))

        print("Cube's linear velocity is : " + str(linear_velocity))

    def gripper_status(self, step_size):
        

        self.surface_gripper.update()                       # On every sim step, update the gripper status

        if self.surface_gripper.is_closed():                # Assign color to Cone based on current state

            self.gripperGeom.color=np.array(self.color_closed)
            print("\nCLOSED\n")

        else:

            self.gripperGeom.color=np.array(self.color_open)
            print("\nOPEN\n", self.count)

    def robot_mover(self, step_size):
        if self.count >= 100:
            self.count = 0
        else:
            self.count +=1
        
        if self.count == 50:
            self.ur10s_view.set_joint_positions(np.array([[3.379,-2.0,-2.142,-0.5753, 1.566, 0.0]]))
        if self.count == 100:
            self.ur10s_view.set_joint_positions(np.array([[3.379,-1.95,-2.16,-0.5753, 1.566, 0.0]]))

    # Async task that pauses simulation once the incoming task is complete

    async def pause_sim(task):

        done, pending = await asyncio.wait({task})

        if task in done:

            print("Waited until next frame, pausing")

            omni.timeline.get_timeline_interface().pause()
    
    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
