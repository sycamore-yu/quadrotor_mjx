import jax
import jax.numpy as jnp


class DoubleSphereCamera:
    """Double sphere camera model for landmark projection.

    Reference: third_party/rpg_flightning/flightning/sensors/double_sphere_camera.py
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        alpha: float,
        xi: float,
        width: int,
        height: int,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.alpha = alpha
        self.xi = xi
        self.width = width
        self.height = height
        # Camera points in x direction of quad frame with pitch=-90 (looking down)
        # rot_CprimeB: fixed rotation from body to camera prime frame
        # rot_cam: additional pitch rotation
        self._pitch = 0.0
        self._update_rot_cb()

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = value
        self._update_rot_cb()

    def _update_rot_cb(self):
        """Update rotation from body to camera frame."""
        # Fixed rotation: camera x aligns with body -y, camera y with body -z
        # This is a 90-degree rotation around X then 90 around Z
        rot_CprimeB = jnp.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ])
        # Pitch rotation around Y axis
        theta = jnp.deg2rad(self._pitch)
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        rot_cam = jnp.array([
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t],
        ])
        self.rot_CB = rot_CprimeB @ rot_cam

    def project_points_with_pose(
        self, points: jax.Array, p_WB: jax.Array, R_WB: jax.Array
    ) -> jax.Array:
        """Project world points to image plane given body pose.

        Args:
            points: Nx3 array of points in world frame
            p_WB: body position in world frame
            R_WB: body rotation matrix (world to body)

        Returns:
            Nx3 array of [u, v, valid] projected points
        """
        # Transform points to camera frame
        R_BW = R_WB.T
        R_CW = self.rot_CB @ R_BW
        p_CW = R_CW @ (-p_WB)

        points_C = (R_CW @ points.T).T + p_CW

        # Double sphere projection
        d1 = jnp.linalg.norm(points_C, axis=1)
        points_C_zxi = points_C.at[:, 2].add(d1 * self.xi)
        d2 = jnp.linalg.norm(points_C_zxi, axis=1)

        div = self.alpha * d2 + (1 - self.alpha) * points_C_zxi[:, 2]

        u = self.fx * (points_C[:, 0] / div) + self.cx
        v = self.fy * (points_C[:, 1] / div) + self.cy

        # Validity checks
        w1 = jax.lax.select(
            self.alpha <= 0.5,
            self.alpha / (1 - self.alpha),
            (1 - self.alpha) / self.alpha,
        )
        w2 = (w1 + self.xi) / jnp.sqrt(2 * w1 * self.xi + self.xi ** 2 + 1)

        predicates = jnp.array([
            points_C[:, 2] > 0,
            points_C[:, 2] > -w2 * d1,
            u >= 0,
            u < self.width,
            v >= 0,
            v < self.height,
        ])
        valid = jnp.all(predicates, axis=0)

        return jnp.column_stack((u, v, valid.astype(float)))


def create_example_camera() -> DoubleSphereCamera:
    """Create camera with example parameters from rpg_flightning."""
    return DoubleSphereCamera(
        fx=600.0,
        fy=450.0,
        cx=630.0,
        cy=390.0,
        alpha=0.6,
        xi=-0.0075,
        width=1280,
        height=1280,
    )
