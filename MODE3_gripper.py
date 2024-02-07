# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import weakref

import numpy as np
import math
import cmath
import copy

import omni
import omni.ext
import omni.kit.commands
import omni.kit.usd
import omni.physx as _physx
import omni.ui as ui
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.dynamic_control import _dynamic_control as dc

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.robot_assembler import RobotAssembler,AssembledBodies 
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.objects import VisualCylinder

from omni.isaac.core import World

# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper, Surface_Gripper_Properties
from omni.isaac.ui.menu import make_menu_item_description
from omni.isaac.ui.ui_utils import (
    add_separator,
    btn_builder,
    combo_floatfield_slider_builder,
    get_style,
    setup_ui_headers,
    state_btn_builder,
)
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

# from transforms import *

EXTENSION_NAME = "MODE3 gripper"


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        """Initialize extension and UI elements"""

        self._ext_id = ext_id
        self.count = 0
        self.robotStep = 0
        # Loads interfaces
        self._timeline = omni.timeline.get_timeline_interface()
        self._dc = dc.acquire_dynamic_control_interface()
        self._usd_context = omni.usd.get_context()
        self._window = None
        self._models = {}
        
        # Creates UI window with default size of 600x300
        # self._window = omni.ui.Window(
        #     title=EXTENSION_NAME, width=300, height=200, visible=False, dockPreference=ui.DockPreference.LEFT_BOTTOM
        # )
        menu_items = [
            make_menu_item_description(ext_id, EXTENSION_NAME, lambda a=weakref.proxy(self): a._menu_callback())
        ]
        self._menu_items = [MenuItemDescription(name="User", sub_menu=menu_items)]
        add_menu_items(self._menu_items, "Isaac Examples")

        self._build_ui()

        self.TCPoffset = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]

        self.startPose = [0.7, 0, -0.3, 2.1451689320097858, -2.295180745437043, -1.3137268122743516e-16]
        self.graspPose = [0.7, 0, -0.4, 2.1451689320097858, -2.295180745437043, -1.3137268122743516e-16]
        self.liftPose =  [0.7, 0, -0.2, 2.1451689320097858, -2.295180745437043, -1.3137268122743516e-16]
        self.farPose =   [0, 0.7, -0.2, 2.1451689320097858, -2.295180745437043, -1.3137268122743516e-16]

        self.surface_gripper = None
        self.cone = None
        self.box = None
        self._stage_id = -1

    def _build_ui(self):
        if not self._window:
            self._window = ui.Window(
                title=EXTENSION_NAME, width=0, height=0, visible=False, dockPreference=ui.DockPreference.LEFT_BOTTOM
            )
            self._window.set_visibility_changed_fn(self._on_window)
            with self._window.frame:
                with ui.VStack(spacing=5, height=0):
                    title = "MODE3 gripper"
                    doc_link = "https://docs.omniverse.nvidia.com/isaacsim/latest/features/robots_simulation/ext_omni_isaac_surface_gripper.html"

                    overview = "This Example shows how to simulate a suction-cup gripper in Isaac Sim. "
                    overview += "It simulates suction by creating a Joint between two bodies when the parent and child bodies are close at the gripper's point of contact."
                    overview += "\n\nPress the 'Open in IDE' button to view the source code."

                    setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

                    frame = ui.CollapsableFrame(
                        title="Command Panel",
                        height=0,
                        collapsed=False,
                        style=get_style(),
                        style_type_name_override="CollapsableFrame",
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                    )
                    with frame:
                        with ui.VStack(style=get_style(), spacing=5):

                            args = {
                                "label": "Load Scene",
                                "type": "button",
                                "text": "Load",
                                "tooltip": "Load a gripper into the Scene",
                                "on_clicked_fn": self._on_create_scenario_button_clicked,
                            }
                            self._models["create_button"] = btn_builder(**args)

                            args = {
                                "label": "Gripper State",
                                "type": "button",
                                "a_text": "Close",
                                "b_text": "Open",
                                "tooltip": "Open and Close the Gripper",
                                "on_clicked_fn": self._on_toggle_gripper_button_clicked,
                            }
                            self._models["toggle_button"] = state_btn_builder(**args)

                            add_separator()

                            args = {
                                "label": "Gripper Force (UP)",
                                "default_val": 0,
                                "min": 0,
                                "max": 1.0e2,
                                "step": 1,
                                "tooltip": ["Force in ()", "Force in ()"],
                            }
                            self._models["force_slider"], slider = combo_floatfield_slider_builder(**args)

                            args = {
                                "label": "Set Force",
                                "type": "button",
                                "text": "APPLY",
                                "tooltip": "Apply the Gripper Force to the Z-Axis of the Cone",
                                "on_clicked_fn": self._on_force_button_clicked,
                            }
                            self._models["force_button"] = btn_builder(**args)

                            args = {
                                "label": "Gripper Speed (UP)",
                                "default_val": 0,
                                "min": 0,
                                "max": 5.0e1,
                                "step": 1,
                                "tooltip": ["Speed in ()", "Speed in ()"],
                            }

                            add_separator()

                            self._models["speed_slider"], slider = combo_floatfield_slider_builder(**args)

                            args = {
                                "label": "Set Speed",
                                "type": "button",
                                "text": "APPLY",
                                "tooltip": "Apply Cone Velocity in the Z-Axis",
                                "on_clicked_fn": self._on_speed_button_clicked,
                            }
                            self._models["speed_button"] = btn_builder(**args)

                            ui.Spacer()

    def on_shutdown(self):
        remove_menu_items(self._menu_items, "Isaac Examples")
        self._physx_subs = None
        self._window = None

    def _on_window(self, status):
        if status:
            self._usd_context = omni.usd.get_context()
            if self._usd_context is not None:
                self._stage_event_sub = (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update_ui)
                )
        else:
            self._stage_event_sub = None
            self._physx_subs = None

    def _menu_callback(self):
        self._window.visible = not self._window.visible

    def _on_update_ui(self, widget):
        self._models["create_button"].enabled = self._timeline.is_playing()
        self._models["toggle_button"].enabled = self._timeline.is_playing()
        self._models["force_button"].enabled = self._timeline.is_playing()
        self._models["speed_button"].enabled = self._timeline.is_playing()
        # If the scene has been reloaded, reset UI to create Scenario
        if self._usd_context.get_stage_id() != self._stage_id:
            self._models["create_button"].enabled = True
            # self._models["create_button"].text = "Create Scenario"
            self._models["create_button"].set_tooltip("Creates a new scenario with the cone on top of the Cube")
            self._models["create_button"].set_clicked_fn(self._on_create_scenario_button_clicked)
            self.cone = None
            self.box = None
            self._stage_id = -1

    def _toggle_gripper_button_ui(self):
        # Checks if the surface gripper has been created
        if self.surface_gripper is not None:
            if self.surface_gripper.is_closed():
                self._models["toggle_button"].text = "OPEN"
            else:
                self._models["toggle_button"].text = "CLOSE"
        pass

    # actions that occur each time step    
    def _on_simulation_step(self, step):
        # Checks if the simulation is playing, and if the stage has been loaded
        if self._timeline.is_playing() and self._stage_id != -1:
            self.count += 1
            if self.count >= 50: 
                self.count = 0
                # statemachine to control robot motion & grasping of item
                if self.robotStep == 0:
                    # starting position with robot above item
                    self.calcJointAngles = self.calc_joint_angles( desired_TCP_pose = self.liftPose, TCP_offset = self.TCPoffset)
                    self.calcJointAngles[0] += 3.14
                    # self.ur10s_view.set_joint_positions(np.array([3.3914,-1.9001,-2.1309,-0.68138, 1.570796, 0.318968]))
                    # self.ur10s_view.set_joint_velocities([0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001])
                    angVel = self.ur10s_view.get_joint_velocities()
                    print("Angular velocities:", angVel)
                    gains = self.ur10s_view.get_gains()
                    print("\nGains:", gains)
                    self.ur10s_view.set_joint_positions(np.array(self.calcJointAngles))
                    print("Above item")
                    self.robotStep = 1

                elif self.robotStep == 1:
                    # gripper contacts item
                    self.calcJointAngles = self.calc_joint_angles( desired_TCP_pose = self.graspPose, TCP_offset = self.TCPoffset)
                    self.calcJointAngles[0] += 3.14
                    angVel = self.ur10s_view.get_joint_velocities()
                    print("Angular velocities:", angVel)
                    self.ur10s_view.set_joint_positions(np.array(self.calcJointAngles))
                    # self.ur10s_view.set_joint_positions(np.array([3.3914,-2.06232,-2.160112,-0.4899456, 1.570796, 0.318968]))
                    print("Contact item")
                    self.robotStep = 2

                elif self.robotStep == 2:
                    # grasp item
                    self.surface_gripper.close()
                    print("Gripper closing")
                    # self.surface_gripper.open()
                    self.robotStep = 3

                elif self.robotStep == 3:
                    # lift item
                    self.calcJointAngles = self.calc_joint_angles( desired_TCP_pose = self.liftPose, TCP_offset = self.TCPoffset)
                    self.calcJointAngles[0] += 3.14
                    self.ur10s_view.set_joint_positions(np.array(self.calcJointAngles))
                    # self.ur10s_view.set_joint_positions(np.array([3.3914,-1.9001,-2.1309,-0.68138, 1.570796, 0.318968]))
                    print("Lifting item")
                    self.robotStep = 4
                
                elif self.robotStep == 4:
                    print("Item lifted")
                    self.robotStep = 5
                
                elif self.robotStep == 5:
                    self.x = 0.71
                    # lower item
                    # self.calcJointAngles = self.calc_joint_angles( desired_TCP_pose = self.farPose, TCP_offset = self.TCPoffset)
                    # self.calcJointAngles[0] += 3.14
                    # self.ur10s_view.set_joint_positions(np.array(self.calcJointAngles))
                    # self.ur10s_view.set_joint_positions(np.array([3.3914,-2.06232,-2.160112,-0.4899456, 1.570796, 0.318968]))  
                    print("Far pose")
                    self.robotStep = 6

                elif self.robotStep == 6:
                    # lift item
                    self.calcJointAngles = self.calc_joint_angles( desired_TCP_pose = self.liftPose, TCP_offset = self.TCPoffset)
                    self.calcJointAngles[0] += 3.14
                    self.ur10s_view.set_joint_positions(np.array(self.calcJointAngles))
                    # self.ur10s_view.set_joint_positions(np.array([3.3914,-1.9001,-2.1309,-0.68138, 1.570796, 0.318968]))
                    print("Back to above item")
                    self.robotStep = 7

                elif self.robotStep == 7:
                    # lower item
                    self.calcJointAngles = self.calc_joint_angles( desired_TCP_pose = self.graspPose, TCP_offset = self.TCPoffset)
                    self.calcJointAngles[0] += 3.14
                    self.ur10s_view.set_joint_positions(np.array(self.calcJointAngles))
                    # self.ur10s_view.set_joint_positions(np.array([3.3914,-2.06232,-2.160112,-0.4899456, 1.570796, 0.318968]))  
                    print("Lowering item")
                    self.robotStep = 8

                elif self.robotStep == 8:
                    # release item
                    self.surface_gripper.open()  
                    print("Releasing gripper") 
                    self.robotStep = 0 

            # step control of robot motion
            if self.robotStep == 6:
                self.x -= 0.01
                if self.x < 0:
                    self.x = 0
                y = self.x*-1 + 0.7
                print("X=", self.x, ", Y = ", y)
                pose = [self.x, y, self.farPose[2], self.farPose[3], self.farPose[4], self.farPose[5]]
                self.calcJointAngles = self.calc_joint_angles( desired_TCP_pose = pose, TCP_offset = self.TCPoffset)
                self.calcJointAngles[0] += 3.14
                self.ur10s_view.set_joint_positions(np.array(self.calcJointAngles))
            elif self.robotStep == 7:
                self.x += 0.01
                if self.x > 0.7:
                    self.x = 0.7
                y = self.x*-1 + 0.7
                print("X=", self.x, ", Y = ", y)
                pose = [self.x, y, self.farPose[2], self.farPose[3], self.farPose[4], self.farPose[5]]
                self.calcJointAngles = self.calc_joint_angles( desired_TCP_pose = pose, TCP_offset = self.TCPoffset)
                self.calcJointAngles[0] += 3.14
                self.ur10s_view.set_joint_positions(np.array(self.calcJointAngles))

            self.surface_gripper.update()
            if self.surface_gripper.is_closed():
                # print("\nCLOSED!")
                # self.cylGeom.GetDisplayColorAttr().Set([self.color_closed])
                self.boxGeom.GetDisplayColorAttr().Set([self.color_closed])
            else:
                # print("\nOPEN!")
                # self.cylGeom.GetDisplayColorAttr().Set([self.color_open])
                self.boxGeom.GetDisplayColorAttr().Set([self.color_open])
            self._toggle_gripper_button_ui()
            
            # # pos = self.ur10s_view.get_joint_positions()
            # print("\nJoint positions:")
            # print(pos)
            # print("")
            # Check if the handles for cone and box have been loaded
            if self.cone is None:
                # self.cone = self._dc.get_rigid_body("/World/GripperCyl")
                self.cone = self._dc.get_rigid_body("/World/MODE3_Gripper")
                self.box = self._dc.get_rigid_body("/World/Box")
            # If the surface Gripper has been created, update wheter it has been broken or not
            if self.surface_gripper is not None:
                self.surface_gripper.update()
                if self.surface_gripper.is_closed():
                    # self.cylGeom.GetDisplayColorAttr().Set([self.color_closed])
                    self.boxGeom.GetDisplayColorAttr().Set([self.color_closed])
                else:
                    # self.cylGeom.GetDisplayColorAttr().Set([self.color_open])
                    self.boxGeom.GetDisplayColorAttr().Set([self.color_open])
                self._toggle_gripper_button_ui()

    def _on_reset_scenario_button_clicked(self):
        if self._timeline.is_playing() and self._stage_id != -1:
            if self.surface_gripper is not None:
                self.surface_gripper.open()
            self._dc.set_rigid_body_linear_velocity(self.cone, [0, 0, 0])
            self._dc.set_rigid_body_linear_velocity(self.box, [0, 0, 0])
            self._dc.set_rigid_body_angular_velocity(self.cone, [0, 0, 0])
            self._dc.set_rigid_body_angular_velocity(self.box, [0, 0, 0])

            self._dc.set_rigid_body_pose(self.cone, self.gripper_start_pose)
            self._dc.set_rigid_body_pose(self.box, self.box_start_pose)

    async def _create_scenario(self, task):
        done, pending = await asyncio.wait({task})
        if task in done:
            
            # Repurpose button to reset Scene
            # self._models["create_button"].text = "Reset Scene"
            self._models["create_button"].set_tooltip("Resets scenario with the cone on top of the Cube")

            # Get Handle for stage and stage ID to check if stage was reloaded
            self._stage = self._usd_context.get_stage()
            self._stage_id = self._usd_context.get_stage_id()
            self._timeline.stop()
            self._models["create_button"].set_clicked_fn(self._on_reset_scenario_button_clicked)

            # Adds a light to the scene
            distantLight = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/World/DistantLight"))
            distantLight.CreateIntensityAttr(500)
            distantLight.AddOrientOp().Set(Gf.Quatf(-0.3748, -0.42060, -0.0716, 0.823))

            # Set up stage with Z up, treat units as cm, set up gravity and ground plane
            UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)
            self.scene = UsdPhysics.Scene.Define(self._stage, Sdf.Path("/World/physicsScene"))
            self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            self.scene.CreateGravityMagnitudeAttr().Set(9.81)
            omni.kit.commands.execute(
                "AddGroundPlaneCommand",
                stage=self._stage,
                planePath="/World/groundPlane",
                axis="Z",
                size=10.000,
                position=Gf.Vec3f(0),
                color=Gf.Vec3f(0.5),
            )
            # Colors to represent when gripper is open or closed
            self.color_closed = Gf.Vec3f(1.0, 0.2, 0.2)
            self.color_open = Gf.Vec3f(0.2, 1.0, 0.2)

            # Cone that will represent the gripper
            self.gripper_start_pose = dc.Transform([0, 0, 0.301], [1, 0, 0, 0])
            # self.gripper_start_pose = dc.Transform([0, 0, 0.401], [0, 1, 0, 0])
            # self.gripper_start_pose = dc.Transform([1.18308, 0.3907, 0.05073], [0.00332, 0.00332, 0.7071, -0.7071])
            # self.coneGeom = self.createRigidBody(
            #     UsdGeom.Cone,
            #     "/GripperCone",
            #     0.100,
            #     [0.10, 0.10, 0.10],
            #     self.gripper_start_pose.p,
            #     self.gripper_start_pose.r,
            #     self.color_open,
            # )

            self.gripper_start_pose = dc.Transform([0, 0, 0.2501], [0, 1, 0, 0])
            # self.cylGeom = self.createRigidBody(
            #     UsdGeom.Cylinder,               # body type
            #     "/World/GripperCyl",   # prim path
            #     0.100,                          # mass
            #     [0.05, 0.05, 0.25],             # scale
            #     self.gripper_start_pose.p,      # position
            #     self.gripper_start_pose.r,      # rotation
            #     self.color_open,                # color
            # )

            stl_path = "C:/Users/jeffk/Documents/Omniverse/MODE3 - setm312.usd"
            # asset_path = get_assets_root_path() + "C:/Users/jeffk/Documents/Omniverse/GripperOnUR10.usd"
            

            add_reference_to_stage(usd_path=stl_path, prim_path="/World/MODE3_Gripper")

            # # batch process articulations via an ArticulationView

            self.gripper_view = ArticulationView(prim_paths_expr="/World/MODE3_Gripper", name="gripper_view")
            

            # # set robot base position
            new_positions = np.array([[2, 1, 0]])

            # # set the joint positions for each articulation
            self.gripper_view.set_world_poses(positions=new_positions)

            self.base_pose = dc.Transform([0, 0, 0.2501], [0, 0, 0, 0])
            self.baseGeom = self.createRigidBody(
                UsdGeom.Cylinder,               # body type
                "/World/RobotBase",   # prim path
                0.100,                          # mass
                [0.1, 0.1, 0.25],             # scale
                self.base_pose.p,      # position
                self.base_pose.r,      # rotation
                self.color_open,                # color
            )

            # Box to be picked
            # self.box_start_pose = dc.Transform([0, 0, 0.10], [1, 0, 0, 0])
            self.box_start_pose = dc.Transform([0.7, 0, 0.05], [1, 0, 0, 0])
            self.boxGeom = self.createRigidBody(
                UsdGeom.Cube, "/World/Box", 0.10, [0.05, 0.05, 0.05], self.box_start_pose.p, self.box_start_pose.r, [0.2, 0.2, 1]
            )

            # Reordering the quaternion to follow DC convention for later use.
            self.gripper_start_pose = dc.Transform([0, 0, 0.301], [0, 0, 0, 1])
            # self.gripper_start_pose = dc.Transform([1.18308, 0.3907, 0.05073], [0.00332, 0.00332, 0.7071, -0.7071])
            # self.box_start_pose = dc.Transform([0, 0, 0.10], [0, 0, 0, 1])
            self.box_start_pose = dc.Transform([0.7, 0, 0.10], [0, 0, 0, 1])

            # Gripper properties
            self.sgp = Surface_Gripper_Properties()
            # self.sgp.d6JointPath = "/World/GripperCyl/SurfaceGripper"
            # self.sgp.parentPath = "/World/GripperCyl"
            self.sgp.d6JointPath = "/World/MODE3_Gripper/SurfaceGripper"
            self.sgp.parentPath = "/World/MODE3_Gripper"
            self.sgp.offset = dc.Transform()
            self.sgp.offset.p.x = 0#-0.349999
            self.sgp.offset.p.z = -0.1001
            self.sgp.offset.r = [0.7071, 0, 0.7071, 0]  # Rotate to point gripper in Z direction
            self.sgp.gripThreshold = 0.02
            self.sgp.forceLimit = 10000
            self.sgp.torqueLimit = 100000
            self.sgp.bendAngle = np.pi / 4
            self.sgp.stiffness = 1.0e4
            self.sgp.damping = 1.0e3

            self.surface_gripper = Surface_Gripper(self._dc)
            self.surface_gripper.initialize(self.sgp)
            # Set camera to a nearby pose and looking directly at the Gripper cone
            set_camera_view(
                eye=[4.00, 4.00, 4.00], target=self.gripper_start_pose.p, camera_prim_path="/OmniverseKit_Persp"
            )

            asset_path = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            # asset_path = get_assets_root_path() + "C:/Users/jeffk/Documents/Omniverse/GripperOnUR10.usd"
            

            add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur10e_1")

            # # batch process articulations via an ArticulationView

            self.ur10s_view = ArticulationView(prim_paths_expr="/World/ur10e_1", name="ur10s_view")
            

            # # set robot base position
            new_positions = np.array([[0, 0, 0.5]])

            # # set the joint positions for each articulation
            self.ur10s_view.set_world_poses(positions=new_positions)
            
            # asset_path2 = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
            # # asset_path = get_assets_root_path() + "C:/Users/jeffk/Documents/Omniverse/GripperOnUR10.usd"
            

            # add_reference_to_stage(usd_path=asset_path2, prim_path="/World/ur10e_2")

            # # # batch process articulations via an ArticulationView

            # self.ur10_2_view = ArticulationView(prim_paths_expr="/World/ur10e_2", name="ur10s_2_view")
            

            # # # set robot base position
            # new_positions = np.array([[2, 2, 0.5]])

            # # # set the joint positions for each articulation
            # self.ur10_2_view.set_world_poses(positions=new_positions)


            


            # assemble gripper on the robot
            # self.assembleRobot()
            self.assembleRobot2()

            # start the simulation
            self._physx_subs = _physx.get_physx_interface().subscribe_physics_step_events(self._on_simulation_step)
            self._timeline.play()

            # initialize the articulation (self.ur10s_view)
            await self._initialize_async()

            # set stiffness and damping for robot motion control
            dampings = np.array([10000, 10000, 10000, 10000, 10000, 10000])
            stiffnesses = np.array([0, 0, 0, 0, 0, 0])
            self.ur10s_view.set_gains(kps=stiffnesses, kds=dampings)
            
            # re-initialize the articulation
            await self._initialize_async()
            self.ur10s_view.set_joint_positions(np.array([3.3914,-2.06232,-2.160112,-0.4899456, 1.570796, 0.318968]))
            pos = self.ur10s_view.get_joint_positions()
            print("\nJoint positions:")
            print(pos)
            print("")

    async def _initialize_async(self):
        await omni.kit.app.get_app().next_update_async()
        print("initializing Articulation veiew")
        self.ur10s_view.initialize()

    # assemles basic gripper on UR10e robot
    def assembleRobot(self):
        robot_assembler = RobotAssembler()
        base_robot_path = "/World/ur10e_1"
        attach_robot_path = "/World/GripperCyl"
        robot_assembler.convert_prim_to_rigid_body(attach_robot_path)
        base_robot_mount_frame = "/tool0"
        attach_robot_mount_frame = ""
        fixed_joint_offset = np.array([0.0,0.0,0.25])
        fixed_joint_orient = np.array([1.0,0.0,0.0,0.0])
        assembled_bodies = robot_assembler.assemble_rigid_bodies(
            base_robot_path,
            attach_robot_path,
            base_robot_mount_frame,
            attach_robot_mount_frame,
            fixed_joint_offset,
            fixed_joint_orient,
            mask_all_collisions = True
            )

    # assembles MODE3 gripper on UR10e
    def assembleRobot2(self):
        robot_assembler = RobotAssembler()
        base_robot_path = "/World/ur10e_1"
        attach_robot_path = "/World/MODE3_Gripper"
        robot_assembler.convert_prim_to_rigid_body(attach_robot_path)
        base_robot_mount_frame = "/tool0"
        attach_robot_mount_frame = ""
        fixed_joint_offset = np.array([0.0,0.0,0.0])
        fixed_joint_orient = np.array([1.0,0.0,0.0,0.0])
        assembled_bodies = robot_assembler.assemble_rigid_bodies(
            base_robot_path,
            attach_robot_path,
            base_robot_mount_frame,
            attach_robot_mount_frame,
            fixed_joint_offset,
            fixed_joint_orient,
            mask_all_collisions = True
            )

    def _on_create_scenario_button_clicked(self):
        # wait for new stage before creating scenario
        task = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
        asyncio.ensure_future(self._create_scenario(task))

    def _on_toggle_gripper_button_clicked(self, val=False):
        if self._timeline.is_playing():
            if self.surface_gripper.is_closed():
                self.surface_gripper.open()
            else:
                self.surface_gripper.close()
            if self.surface_gripper.is_closed():
                self._models["toggle_button"].text = "OPEN"
            else:
                self._models["toggle_button"].text = "CLOSE"

    def _on_speed_button_clicked(self):
        if self._timeline.is_playing():
            self._dc.set_rigid_body_linear_velocity(
                self.cone, [0, 0, self._models["speed_slider"].get_value_as_float()]
            )

    def _on_force_button_clicked(self):
        if self._timeline.is_playing():
            self._dc.apply_body_force(
                self.cone, [0, 0, self._models["force_slider"].get_value_as_float()], [0, 0, 0], True
            )

    def createRigidBody(self, bodyType, boxActorPath, mass, scale, position, rotation, color):
        p = Gf.Vec3f(position[0], position[1], position[2])
        orientation = Gf.Quatf(rotation[0], rotation[1], rotation[2], rotation[3])
        scale = Gf.Vec3f(scale[0], scale[1], scale[2])

        bodyGeom = bodyType.Define(self._stage, boxActorPath)
        bodyPrim = self._stage.GetPrimAtPath(boxActorPath)
        bodyGeom.AddTranslateOp().Set(p)
        bodyGeom.AddOrientOp().Set(orientation)
        bodyGeom.AddScaleOp().Set(scale)
        bodyGeom.CreateDisplayColorAttr().Set([color])

        UsdPhysics.CollisionAPI.Apply(bodyPrim)
        if mass > 0:
            massAPI = UsdPhysics.MassAPI.Apply(bodyPrim)
            massAPI.CreateMassAttr(mass)
        UsdPhysics.RigidBodyAPI.Apply(bodyPrim)
        UsdPhysics.CollisionAPI(bodyPrim)
        print(bodyPrim.GetPath().pathString)
        return bodyGeom
    
    def calc_joint_angles(self, desired_TCP_pose, TCP_offset):
        # perfrom inverse kinematic calculations to find joint angles:

        flange_pose = self.PoseTrans(desired_TCP_pose, self.PoseInv(TCP_offset))
        # print("Flange pose: ", flange_pose)
        flange_affine = self.rotVec_to_rotMat_affine(flange_pose)
        jointAngles = self.invKine(flange_affine)
        calc_joint_angles = [jointAngles.item(0,5), jointAngles.item(1,5), jointAngles.item(2,5), jointAngles.item(3,5), jointAngles.item(4,5), jointAngles.item(5,5)]


        return calc_joint_angles

    ####
    # get DH parameters
    ##
    #### input arguments
    # gen: the generation of the robot
    #		3: CB3
    #		5: e-Series
    # model: the robot model
    #		3: UR3/UR3e
    #		5: UR5/UR5e
    #		10: UR10/UR10e
    ##
    #### return values
    # a: translational offset in x axis of n frame (returned in the pose variable format)
    # d: translational offset in z axis of n-1 frame (returned in the pose variable format)
    # alpha: rotatinal offset in x axis of n frame (returned in the pose variable format)
    ##
    ####
    def get_dh_parameter(self, gen,model):

        a_pose = [ 0, 0, 0, 0, 0, 0 ]
        d_pose = [ 0, 0, 0, 0, 0, 0 ]
        alpha_pose = [ self.d2r(90), 0, 0, self.d2r(90), -self.d2r(90), 0 ]

        if (gen == 3):
            if (model == 3): 
                a_pose = [ 0, -0.24365, -0.21325, 0, 0, 0 ]
                d_pose = [ 0.1519, 0, 0, 0.11235, 0.08535, 0.0819 ]
            elif (model == 5):
                a_pose = [ 0, -0.425, -0.39225, 0, 0, 0 ]
                d_pose = [ 0.089159, 0, 0, 0.10915, 0.09465, 0.0823 ]
            elif (model == 10):
                a_pose = [ 0, -0.612, -0.5723, 0, 0, 0 ]
                d_pose = [ 0.1273, 0, 0, 0.163941, 0.1157, 0.0922 ]
            # end
        elif (gen == 5 ):
            if (model == 3): 
                a_pose = [ 0, -0.24355, -0.2132, 0, 0, 0 ]
                d_pose = [ 0.15185, 0, 0, 0.13105, 0.08535, 0.0921 ]
            elif (model == 5):
                a_pose = [ 0, -0.425, -0.3922, 0, 0, 0 ]
                d_pose = [ 0.1625, 0, 0, 0.1333, 0.0997, 0.0996 ]
            elif (model == 10):
                a_pose = [ 0, -0.6127, -0.57155, 0, 0, 0 ]
                d_pose = [ 0.1807, 0, 0, 0.17415, 0.11985, 0.11655 ]
            # end
        # end
        
        dh_parameter = [ a_pose, d_pose, alpha_pose ]

        return dh_parameter

    # ************************************************** FORWARD KINEMATICS

    def AH(self, n,th,c  ):
        dh_parameter = self.get_dh_parameter(gen = 5, model = 10)
        a_pose = dh_parameter[0]
        d_pose = dh_parameter[1]
        alpha_pose = dh_parameter[2]
        T_a = np.matrix(np.identity(4), copy=False)
        T_a[0,3] = a_pose[n-1]
        T_d = np.matrix(np.identity(4), copy=False)
        T_d[2,3] = d_pose[n-1]

        Rzt = np.matrix([[math.cos(th[n-1,c]), -math.sin(th[n-1,c]), 0 ,0],
                    [math.sin(th[n-1,c]),  math.cos(th[n-1,c]), 0, 0],
                    [0,               0,              1, 0],
                    [0,               0,              0, 1]],copy=False)
            

        Rxa = np.matrix([[1, 0,                 0,                  0],
                    [0, math.cos(alpha_pose[n-1]), -math.sin(alpha_pose[n-1]),   0],
                    [0, math.sin(alpha_pose[n-1]),  math.cos(alpha_pose[n-1]),   0],
                    [0, 0,                 0,                  1]],copy=False)

        A_i = T_d * Rzt * T_a * Rxa
            

        return A_i

    # ************************************************** INVERSE KINEMATICS 

    def invKine(self, desired_pos):# T60
        dh_parameter = self.get_dh_parameter(gen = 5, model = 10)
        a_pose = dh_parameter[0]
        d_pose = dh_parameter[1]
        alpha_pose = dh_parameter[2]
        mat = np.matrix
        th = mat(np.zeros((6, 8)))

        # P_05 = (desired_pos * mat([0,0, -d6, 1]).T-mat([0,0,0,1 ]).T)
        P_05 = (desired_pos * mat([0,0, -d_pose[5], 1]).T-mat([0,0,0,1 ]).T)

        # **** theta1 ****

        psi = math.atan2(P_05[2-1,0], P_05[1-1,0])
        phi = math.acos(d_pose[3] /math.sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]))
        #The two solutions for theta1 correspond to the shoulder
        #being either left or right
        th[0, 0:4] = math.pi/2 + psi + phi
        th[0, 4:8] = math.pi/2 + psi - phi
        th = th.real

        # **** theta5 ****

        cl = [0, 4]# wrist up or down
        for i in range(0,len(cl)):
                c = cl[i]
                T_10 = np.linalg.inv(self.AH(1,th,c))
                T_16 = T_10 * desired_pos
                th[4, c:c+2] = + math.acos((T_16[2,3]-d_pose[3])/d_pose[5])
                th[4, c+2:c+4] = - math.acos((T_16[2,3]-d_pose[3])/d_pose[5])

        th = th.real

        # **** theta6 ****
        # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

        cl = [0, 2, 4, 6]
        for i in range(0,len(cl)):
                c = cl[i]
                T_10 = np.linalg.inv(self.AH(1,th,c))
                T_16 = np.linalg.inv( T_10 * desired_pos )
                th[5, c:c+2] = math.atan2((-T_16[1,2]/math.sin(th[4, c])),(T_16[0,2]/math.sin(th[4, c])))
                
        th = th.real

        # **** theta3 ****
        cl = [0, 2, 4, 6]
        for i in range(0,len(cl)):
                c = cl[i]
                T_10 = np.linalg.inv(self.AH(1,th,c))
                T_65 = self.AH( 6,th,c)
                T_54 = self.AH( 5,th,c)
                T_14 = ( T_10 * desired_pos) * np.linalg.inv(T_54 * T_65)
                P_13 = T_14 * mat([0, -d_pose[3], 0, 1]).T - mat([0,0,0,1]).T
                t3 = cmath.acos((np.linalg.norm(P_13)**2 - a_pose[1]**2 - a_pose[2]**2 )/(2 * a_pose[1] * a_pose[2])) # norm ?
                th[2, c] = t3.real
                th[2, c+1] = -t3.real

        # **** theta2 and theta 4 ****

        cl = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(0,len(cl)):
                c = cl[i]
                T_10 = np.linalg.inv(self.AH( 1,th,c ))
                T_65 = np.linalg.inv(self.AH( 6,th,c))
                T_54 = np.linalg.inv(self.AH( 5,th,c))
                T_14 = (T_10 * desired_pos) * T_65 * T_54
                P_13 = T_14 * mat([0, -d_pose[3], 0, 1]).T - mat([0,0,0,1]).T
                
                # theta 2
                th[1, c] = -math.atan2(P_13[1], -P_13[0]) + math.asin(a_pose[2]* math.sin(th[2,c])/np.linalg.norm(P_13))
                # theta 4
                T_32 = np.linalg.inv(self.AH( 3,th,c))
                T_21 = np.linalg.inv(self.AH( 2,th,c))
                T_34 = T_32 * T_21 * T_14
                th[3, c] = math.atan2(T_34[1,0], T_34[0,0])
        th = th.real

        return th

    # #================================================
    # #  Similar to UR Script pose_add(). Add Pose2 to Pose1 NOTE: Pose2 is added to Pose1 **relative to robot base**
    # #================================================

    def PoseTrans(self, Pose1, Pose2):

        P1 = self.GetPointMatrix(Pose1)
        P2 = self.GetPointMatrix(Pose2)

        R1 = self.rotVec_to_rotMat(Pose1)

        R2 = self.rotVec_to_rotMat(Pose2)

        P = np.add(P1, np.matmul(R1, P2))
        R = self.rotmat2rotvec(np.matmul(R1, R2))
    
        return [P[0],P[1],P[2],-R[0],-R[1],-R[2]]

    #================================================
    #  Similar to UR Script pose_add(). Add Pose2 to Pose1 NOTE: Pose2 is added to Pose1 **relative to robot base**
    #================================================
    def PoseInv(self, Pose1):

        P1 = self.GetPointMatrix(Pose1)

        R1 = self.rotVec_to_rotMat(Pose1)

        invR1 = np.linalg.inv(R1)

        P = np.matmul(invR1, P1)

        R = self.rotmat2rotvec(invR1)

        return [-P[0],-P[1],-P[2], -R[0],-R[1],-R[2]]

    #================================================
    #  Return X, Y & Z in a column vector
    #================================================
    def GetPointMatrix(self, Pose):
        X, Y, Z, Rx, Ry, Rz = Pose[0],Pose[1],Pose[2],Pose[3],Pose[4],Pose[5]
        return np.stack([X,Y,Z])

    #================================================
    # convert rotVect to affine rotation Matrix
    #================================================
    def rotVec_to_rotMat_affine(self, pose):
        xCoord, yCoord, zCoord, Rx, Ry, Rz = pose
        angle = math.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
        if angle == 0:
            angle = 0.0000000001
        axis_zero = Rx/angle
        axis_one = Ry/angle
        axis_two = Rz/angle
        matrix = self.axis_to_rotMat_affine(xCoord, yCoord, zCoord, angle, axis_zero, axis_one, axis_two)
        return matrix

    #================================================
    # convert axis-angles to affine rotation Matrix
    #================================================
    def axis_to_rotMat_affine(self, xCoord, yCoord, zCoord, angle, axis_zero, axis_one, axis_two):
        x = copy.copy(axis_zero)
        y = copy.copy(axis_one)
        z = copy.copy(axis_two)
        s = math.sin(angle)
        c = math.cos(angle)
        t = 1.0-c
        magnitude = math.sqrt(x*x + y*y + z*z)
        if magnitude == 0:
            # print("!Error! Magnitude = 0")
            magnitude = 0.0000000001
        else:
            x /= magnitude
            y /= magnitude
            z /= magnitude
        # calulate rotation matrix elements
        m00 = c + x*x*t
        m11 = c + y*y*t
        m22 = c + z*z*t
        tmp1 = x*y*t
        tmp2 = z*s
        m10 = tmp1 + tmp2
        m01 = tmp1 - tmp2
        tmp1 = x*z*t
        tmp2 = y*s
        m20 = tmp1 - tmp2
        m02 = tmp1 + tmp2    
        tmp1 = y*z*t
        tmp2 = x*s
        m21 = tmp1 + tmp2
        m12 = tmp1 - tmp2
        matrix = [ [m00, m01, m02, xCoord], [m10, m11, m12, yCoord], [m20, m21, m22, zCoord],[0.0, 0.0, 0.0, 1.0] ]
        # matrix = np.array(matrix)
        return matrix

    #================================================
    # convert rotVect to affine rotation Matrix
    #================================================
    def rotVec_to_rotMat(self, pose):
        xCoord, yCoord, zCoord, Rx, Ry, Rz = pose
        angle = math.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
        if angle == 0:
            axis_zero = 0
            axis_one = 0
            axis_two = 0
        else:
            axis_zero = Rx/angle
            axis_one = Ry/angle
            axis_two = Rz/angle
        matrix = self.axis_to_rotMat(angle, axis_zero, axis_one, axis_two)
        return matrix

    #================================================
    # convert axis-angles to rotation Matrix
    #================================================
    def axis_to_rotMat(self, angle, axis_zero, axis_one, axis_two):
        x = copy.copy(axis_zero)
        y = copy.copy(axis_one)
        z = copy.copy(axis_two)
        s = math.sin(angle)
        c = math.cos(angle)
        t = 1.0-c
        magnitude = math.sqrt(x*x + y*y + z*z)
        if magnitude == 0:
            # print("!Error! Magnitude = 0")
            magnitude = 0.0000000001
        else:
            x /= magnitude
            y /= magnitude
            z /= magnitude
        # calulate rotation matrix elements
        m00 = c + x*x*t
        m11 = c + y*y*t
        m22 = c + z*z*t
        tmp1 = x*y*t
        tmp2 = z*s
        m10 = tmp1 + tmp2
        m01 = tmp1 - tmp2
        tmp1 = x*z*t
        tmp2 = y*s
        m20 = tmp1 - tmp2
        m02 = tmp1 + tmp2    
        tmp1 = y*z*t
        tmp2 = x*s
        m21 = tmp1 + tmp2
        m12 = tmp1 - tmp2
        matrix = [ [m00, m01, m02], [m10, m11, m12], [m20, m21, m22] ]
        matrix = np.array(matrix)
        return matrix  

    #================================================
    # convert degrees to Radian
    #================================================
    def d2r(self, theta):
        return theta*math.pi/180

    #================================================
    # convert Radians to degrees
    #================================================
    def r2d(self, theta):
        return theta*180/math.pi

    ####
    ####
    # convert from rotation matrix to rotation vector
    ####
    def rotmat2rotvec(self, rotmat):

        # array to matrix
        r11 = rotmat[0][0]
        r21 = rotmat[0][1]
        r31 = rotmat[0][2]
        r12 = rotmat[1][0]
        r22 = rotmat[1][1]
        r32 = rotmat[1][2]
        r13 = rotmat[2][0]
        r23 = rotmat[2][1]
        r33 = rotmat[2][2]

        # print("\n rotmat:")
        # print(rotmat)

        # rotation matrix to rotation vector
        val = (r11+r22+r33-1)/2
        if val > 1:
            val = 1
        elif val <-1:
            val = -1
        theta = math.acos(val)
        sth = math.sin(theta)
        # print("Theta:", theta, "\ttheta = 179.99:", d2r(179.99), "\ttheta = -179.99:", d2r(-179.99),"\ttheta = 180:", d2r(180.0), "\tval:", val)
        if ( (theta > self.d2r(179.99)) or (theta < self.d2r(-179.99)) ):
            theta = self.d2r(180)
            # avoid math domain error when r11, r22 and r33 are less than 0
            if r11 < -1:
                r11 = -1
            if r22 < -1:
                r22 = -1
            if r33 < -1:
                r33 = -1
            if (r21 < 0):

                if (r31 < 0):
                    ux = math.sqrt((r11+1)/2)
                    uy = -math.sqrt((r22+1)/2)
                    uz = -math.sqrt((r33+1)/2)
                else:
                    ux = math.sqrt((r11+1)/2)
                    uy = -math.sqrt((r22+1)/2)
                    uz = math.sqrt((r33+1)/2)

            else:
                if (r31 < 0):
                    ux = math.sqrt((r11+1)/2)
                    uy = math.sqrt((r22+1)/2)
                    uz = -math.sqrt((r33+1)/2)
                else:
                    ux = math.sqrt((r11+1)/2)
                    uy = math.sqrt((r22+1)/2)
                    uz = math.sqrt((r33+1)/2)


        else:
            if theta == 0:
                ux = 0
                uy = 0
                uz = 0
            else:
                ux = (r32-r23)/(2*sth)
                uy = (r13-r31)/(2*sth)
                uz = (r21-r12)/(2*sth)


        rotvec = [(theta*ux),(theta*uy),(theta*uz)]

        return rotvec
