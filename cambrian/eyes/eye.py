"""Defines the `MjCambrianEye` class, which is used to define an eye for the cambrian
environment. The eye is essentially a camera that is attached to a body in the
environment. The eye can render images and provide observations to the agent."""

from typing import Callable, List, Optional, Self, Tuple
from xml.etree.ElementTree import Element

import mujoco as mj
import numpy as np
import torch
from gymnasium import spaces
from hydra_config import HydraContainerConfig, config_wrapper
from scipy.spatial.transform import Rotation as R

from cambrian.renderer import MjCambrianRenderer, MjCambrianRendererConfig
from cambrian.renderer.overlays import MjCambrianCursor, MjCambrianViewerOverlay
from cambrian.renderer.render_utils import convert_depth_distances, convert_depth_to_rgb
from cambrian.utils import MjCambrianGeometry, device, get_logger
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.spec import MjCambrianSpec
from cambrian.utils.types import ObsType


@config_wrapper
class MjCambrianEyeConfig(HydraContainerConfig):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        instance (Callable[[Self, str], MjCambrianEye]): The class instance to use
            when creating the eye. Takes the config and the name of the eye as
            arguments.

        fov (Tuple[float, float]): Independent of the `fovy` field in the MJCF
            xml. Used to calculate the sensorsize field. Specified in degrees. Mutually
            exclusive with `fovy`. If `focal` is unset, it is set to 1, 1. Will override
            `sensorsize`, if set. Fmt: fovy fovx.
        focal (Tuple[float, float]): The focal length of the camera.
            Fmt: focal_y focal_x.
        sensorsize (Tuple[float, float]): The size of the sensor. Fmt: height width.
        resolution (Tuple[int, int]): The width and height of the rendered image.
            Fmt: height width.
        coord (Tuple[float, float]): The x and y coordinates of the eye.
            This is used to determine the placement of the eye on the agent.
            Specified in degrees. This attr isn't actually used by eye, but by the
            agent. The eye has no knowledge of the geometry it's trying to be placed
            on. Fmt: lat lon
        orthographic (bool): Whether the camera is orthographic

        noise_std (float): Standard deviation of the Gaussian noise to be added to
            the rendered image. If 0, no noise is applied.
        integration_factor (float): Factor in [0, 1] controlling exponential
            smoothing of the eye observation. Higher values retain more of the
            previous observation, which suppresses noise, adds motion blur during
            movement, and encourages fixation.

        renderer (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer.
    """

    instance: Callable[[Self, str], "MjCambrianEye"]

    fov: Tuple[float, float]
    focal: Tuple[float, float]
    sensorsize: Tuple[float, float]
    resolution: Tuple[int, int]
    coord: Tuple[float, float]
    orthographic: bool

    noise_std: float
    integration_factor: float

    renderer: MjCambrianRendererConfig


class MjCambrianEye:
    """Defines an eye for the cambrian environment. It essentially wraps a mujoco Camera
    object and provides some helper methods for rendering and generating the XML. The
    eye is attached to the parent body such that movement of the parent body will move
    the eye.

    Args:
        config (MjCambrianEyeConfig): The configuration for the eye.
        name (str): The name of the eye.

    Keyword Args:
        disable_render (bool): Whether to disable rendering. Defaults to False.
            This is useful for derived classes which don't intend to use the default
            rendering mechanism.
    """

    def __init__(
        self, config: MjCambrianEyeConfig, name: str, *, disable_render: bool = False
    ):
        self._config = config
        self._name = name

        self._renders_rgb = "rgb_array" in self._config.renderer.render_modes
        self._renders_depth = "depth_array" in self._config.renderer.render_modes
        assert (
            self._renders_rgb or self._renders_depth
        ), "Need at least one render mode."

        self._prev_obs_shape = self.observation_space.shape
        self._prev_obs: torch.Tensor = None
        self._fixedcamid = -1
        self._spec: MjCambrianSpec = None

        self._has_prev_obs = False

        self._renderer: MjCambrianRenderer = None
        if not disable_render:
            self._renderer = MjCambrianRenderer(self._config.renderer)

    def generate_xml(
        self,
        parent_xml: MjCambrianXML,
        geom: MjCambrianGeometry,
        parent_body_name: Optional[str] = None,
        parent: Optional[List[Element] | Element] = None,
    ) -> MjCambrianXML:
        """Generate the xml for the eye.

        In order to combine the xml for an eye with the xml for the agent that it's
        attached to, we need to replicate the path with which we want to attach the eye.
        For instance, if the body with which we want to attach the eye to is at
        `mujoco/worldbody/torso`, then we need to replicate that path in the new xml.
        This is kind of difficult with the `xml` library, but we'll utilize the
        `CambrianXML` helpers for this.

        Args:
            parent_xml (MjCambrianXML): The xml of the parent body. Used as a reference
                to extract the path of the parent body.
            geom (MjCambrianGeometry): The geometry of the parent body. Used to
                calculate the pos and quat of the eye.
            parent_body_name (Optional[str]): The name of the parent body. Will
                search for the body tag with this name, i.e.
                <body name="<parent_body_name>" ...>. Either this or `parent` must be
                set.
            parent (Optional[List[Element] | Element]): The parent element to attach
                the eye to. If set, `parent_body_name` will be ignored. Either this or
                `parent_body_name` must be set.
        """

        xml = MjCambrianXML.make_empty()

        if parent is None:
            # Get the parent body reference
            parent_body = parent_xml.find(".//body", name=parent_body_name)
            assert parent_body is not None, f"Could not find body '{parent_body_name}'."

            # Iterate through the path and add the parent elements to the new xml
            parent = None
            elements, _ = parent_xml.get_path(parent_body)
            for element in elements:
                if (
                    temp_parent := xml.find(f".//{element.tag}", **element.attrib)
                ) is not None:
                    # If the element already exists, then we'll use that as the parent
                    parent = temp_parent
                    continue
                parent = xml.add(parent, element.tag, **element.attrib)
            assert parent is not None, f"Could not find parent for '{parent_body_name}'"

        # Finally add the camera element at the end
        pos, quat = self._calculate_pos_quat(geom, self._config.coord)
        resolution = [1, 1]
        if self._renderer is not None:
            resolution = [self._renderer.config.width, self._renderer.config.height]
        xml.add(
            parent,
            "camera",
            name=self._name,
            mode="fixed",
            pos=" ".join(map(str, pos)),
            quat=" ".join(map(str, quat)),
            focal=" ".join(map(str, self._config.focal)),
            sensorsize=" ".join(map(str, self._config.sensorsize)),
            resolution=" ".join(map(str, resolution)),
            orthographic=str(self._config.orthographic).lower(),
        )

        return xml

    def _calculate_pos_quat(
        self, geom: MjCambrianGeometry, coord: Tuple[float, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the position and quaternion of the eye based on the geometry of
        the parent body. The position is calculated by moving the eye to the edge of the
        geometry in the negative x direction. The quaternion is calculated by rotating
        the eye to face the center of the geometry.

        Todo:
            rotations are weird. fix this.
        """
        lat, lon = torch.deg2rad(torch.tensor(coord))
        lon += torch.pi / 2

        default_rot = R.from_euler("z", torch.pi / 2)
        pos_rot = default_rot * R.from_euler("yz", [lat, lon])
        rot_rot = R.from_euler("z", lat) * R.from_euler("y", -lon) * default_rot

        pos = pos_rot.apply([-geom.rbound, 0, 0]) + geom.pos
        quat = rot_rot.as_quat()
        return pos, quat

    def reset(self, spec: MjCambrianSpec) -> ObsType:
        """Sets up the camera for rendering. This should be called before rendering
        the first time."""

        self._spec = spec

        if self._renderer is None:
            return self.step()

        resolution = [self._renderer.config.width, self._renderer.config.height]
        self._renderer.reset(spec, *resolution)

        self._fixedcamid = spec.get_camera_id(self._name)
        assert self._fixedcamid != -1, f"Camera '{self._name}' not found."
        self._renderer.viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._renderer.viewer.camera.fixedcamid = self._fixedcamid

        self._prev_obs = torch.zeros(
            self._prev_obs_shape,
            dtype=torch.float32,
            device=device,
        )
        self._has_prev_obs = False

        obs = self.step()
        if obs.device != self._prev_obs.device:
            get_logger().warning(
                "Device mismatch. obs.device: "
                f"{obs.device}, self._prev_obs.device: {self._prev_obs.device}"
            )
        return obs

    def step(self, obs: ObsType = None) -> ObsType:
        """Simply calls `render` and sets the last observation. See `render()` for more
        information.

        Args:
            obs (Optional[ObsType]): The observation to set. Defaults to
                None. This can be used by derived classes to set the observation
                directly.
        """
        if obs is None:
            assert self._renderer is not None, "Cannot step without a renderer."
            obs = self._renderer.render()
            if self._renders_rgb and self._renders_depth:
                # If both are rendered, then we only return the rgb
                get_logger().warning(
                    "Both rgb and depth are rendered. Using only rgb.",
                    extra={"once": True},
                )
                obs = obs[0]

        obs = self._apply_sensor_noise(obs)
        obs = self._integrate_observation(obs)

        return self._update_obs(obs)

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Update the observation space."""
        self._prev_obs.copy_(obs, non_blocking=True)
        self._has_prev_obs = True
        return self._prev_obs

    def _apply_sensor_noise(self, obs: ObsType) -> ObsType:
        """Add Gaussian noise to the observation if configured."""

        std = self._config.noise_std
        if std == 0.0:
            return obs

        noise = torch.normal(mean=0.0, std=std, size=obs.shape, device=obs.device)
        return torch.clamp(obs + noise, 0, 1)

    def _integrate_observation(self, obs: ObsType) -> ObsType:
        """Exponential smoothing that creates motion blur and rewards fixation.

        The integration factor acts as a base level of denoising for static views.
        When the eye or scene moves, frame-to-frame differences increase the weight
        of the previous frame, introducing perceptible motion blur that encourages
        the agent to stabilize its gaze.
        """

        alpha = self._config.integration_factor
        if alpha == 0.0 or not self._has_prev_obs:
            return obs

        motion_level = torch.mean(torch.abs(obs - self._prev_obs)).clamp(0.0, 1.0)
        effective_alpha = torch.clamp(alpha * (0.5 + 0.5 * motion_level), 0.0, 1.0)

        return (effective_alpha * self._prev_obs) + ((1 - effective_alpha) * obs)

    def render(self) -> List[MjCambrianViewerOverlay]:
        """Render the image from the camera. Will always only return the rgb array.

        This differs from step in that this is a debug method. The rendered image here
        will be used to visualize the eye in the viewer.
        """
        if self._renders_depth and not self._renders_rgb:
            image = convert_depth_to_rgb(
                convert_depth_distances(self._spec.model, self._prev_obs),
                znear=0,
                zfar=self._spec.model.stat.extent,
            )
        image = self._prev_obs

        position = MjCambrianCursor.Position.BOTTOM_LEFT
        layer = MjCambrianCursor.Layer.BACK
        cursor = MjCambrianCursor(position=position, x=0, y=0, layer=layer)
        return [MjCambrianViewerOverlay.create_image_overlay(image, cursor=cursor)]

    @property
    def config(self) -> MjCambrianEyeConfig:
        """The config for the eye."""
        return self._config

    @property
    def name(self) -> str:
        """The name of the eye."""
        return self._name

    @property
    def observation_space(self) -> spaces.Box:
        """Constructs the observation space for the eye. The observation space is a
        `spaces.Box` with the shape of the resolution of the eye."""

        shape = (
            (*self._config.resolution, 3)
            if self._renders_rgb
            else self._config.resolution
        )
        return spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)

    @property
    def prev_obs(self) -> torch.Tensor:
        """The last observation returned by `self.render()`."""
        return self._prev_obs
