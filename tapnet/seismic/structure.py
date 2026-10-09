"""Wu et al. (2020) folding and volumetric dip-slip fault transformations.

Coordinates are physical X, Y, Z with positive depth. See equations 1-14 of
doi:10.1190/geo2019-0375.1. Inverse maps resample continuous stratigraphy,
avoiding holes caused by forward raster splatting.
"""

from __future__ import annotations

import dataclasses

import numpy as np


SURFACE_POINT_COUNT = 9
FAULT_PARAMETER_NAMES = (
    'reference_x', 'reference_y', 'reference_z', 'strike_degrees', 'dip_degrees',
    'max_slip', 'strike_radius', 'dip_radius', 'drag_radius',
    'hanging_wall_fraction',
)


def _biharmonic_kernel(radius_squared):
  """Thin-plate Green function r**2 log(r), with its zero-radius limit."""
  return 0.5 * radius_squared * np.log(np.maximum(radius_squared, 1e-20))


@dataclasses.dataclass(frozen=True)
class WuFault:
  """A finite curved dip-slip fault; max_slip < 0 produces reverse slip.

  Surface controls have normalized strike/dip coordinates and physical normal
  offsets. The interpolating biharmonic spline includes an affine term.
  """

  reference: tuple[float, float, float]
  strike_degrees: float
  dip_degrees: float
  max_slip: float
  strike_radius: float
  dip_radius: float
  drag_radius: float
  hanging_wall_fraction: float = 0.5
  surface_controls: np.ndarray | None = None

  def __post_init__(self):
    if self.surface_controls is None:
      object.__setattr__(self, '_weights', None)
      object.__setattr__(self, '_affine', None)
      return
    points = self.surface_controls[:, :2]
    distance = np.sum((points[:, None] - points[None])**2, axis=-1)
    affine = np.column_stack((np.ones(len(points)), points))
    system = np.block([
        [_biharmonic_kernel(distance), affine],
        [affine.T, np.zeros((3, 3))],
    ])
    coefficients = np.linalg.solve(
        system, np.concatenate((self.surface_controls[:, 2], np.zeros(3)))
    )
    object.__setattr__(self, '_weights', coefficients[:-3])
    object.__setattr__(self, '_affine', coefficients[-3:])

  @property
  def rotation(self):
    """Orthonormal strike/dip/normal basis (corrects eq. 5's printed sign)."""
    strike, dip = np.deg2rad((self.strike_degrees, self.dip_degrees))
    s, c = np.sin(strike), np.cos(strike)
    sd, cd = np.sin(dip), np.cos(dip)
    return np.array([
        [s, c, 0], [c * cd, -s * cd, sd], [c * sd, -s * sd, -cd]
    ])

  def surface(self, x, y):
    """Evaluates the curved fault surface z=f(x,y) in local coordinates."""
    if self._weights is None:
      return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
    u, v = x / self.strike_radius, y / self.dip_radius
    result = self._affine[0] + self._affine[1] * u + self._affine[2] * v
    for point, weight in zip(self.surface_controls, self._weights):
      result = result + weight * _biharmonic_kernel(
          (u - point[0])**2 + (v - point[1])**2
      )
    return result

  def slip_profile(self, x, y):
    """Elliptic slip and its dip derivative (equations 6-7)."""
    radius = np.sqrt((x / self.strike_radius)**2 + (y / self.dip_radius)**2)
    r = np.minimum(radius, 1)
    slip = self.max_slip * np.sqrt((1 - r)**3 * (1 + 3 * r))
    derivative = (
        -6 * self.max_slip * np.sqrt((1 - r) / (1 + 3 * r))
        * y / self.dip_radius**2
    )
    return slip, derivative

  def _local(self, coordinates):
    offset = [a - origin for a, origin in zip(coordinates, self.reference)]
    return tuple(sum(row[i] * offset[i] for i in range(3))
                 for row in self.rotation)

  def _global(self, coordinates):
    return tuple(sum(row[i] * coordinates[i] for i in range(3)) + origin
                 for row, origin in zip(self.rotation.T, self.reference))

  def _block_weight(self, distance):
    drag = np.maximum(1 - np.abs(distance) / self.drag_radius, 0)**2
    return np.where(
        distance >= 0, self.hanging_wall_fraction, self.hanging_wall_fraction - 1
    ) * drag

  def forward(self, coordinates):
    """Moves both blocks along the curved surface (equations 8-14)."""
    x, y, z = self._local(coordinates)
    surface = self.surface(x, y)
    dy = self._block_weight(z - surface) * self.slip_profile(x, y)[0]
    dz = self.surface(x, y + dy) - surface
    return self._global((x, y + dy, z + dz))

  def restore(self, coordinates):
    """Inverts slip while preserving z-f(x,y), the fault-block membership.

    Generated slip/radius bounds keep y -> y+Dy strictly increasing. Newton's
    method solves that scalar map; curvature then gives the exact normal shift.
    """
    x, target_y, z = self._local(coordinates)
    distance = z - self.surface(x, target_y)
    weight = self._block_weight(distance)
    y = target_y.copy()
    for _ in range(10):
      slip, derivative = self.slip_profile(x, y)
      residual = y + weight * slip - target_y
      y = y - residual / (1 + weight * derivative)
      if np.max(np.abs(residual)) < 1e-5:
        break
    source_z = distance + self.surface(x, y)
    return self._global((x, y, source_z))

  def fault_coordinates(self, coordinates):
    """Returns distance to the surface and finite fault footprint."""
    x, y, z = self._local(coordinates)
    distance = z - self.surface(x, y)
    active = (x / self.strike_radius)**2 + (y / self.dip_radius)**2 < 1
    return distance, active

  def parameters(self):
    return np.asarray((
        *self.reference, self.strike_degrees, self.dip_degrees, self.max_slip,
        self.strike_radius, self.dip_radius, self.drag_radius,
        self.hanging_wall_fraction,
    ), dtype=np.float32)


def sample_faults(height, max_faults, max_throw, rng):
  """Samples Wu faults in depth-sample units, with a nominal throw budget."""
  if max_throw == 0:
    return []
  count = int(rng.integers(1, max_faults + 1))
  extent = height - 1
  faults = []
  for _ in range(count):
    dip = rng.uniform(50, 80)
    throw = rng.uniform(0.6, 1.0) * max_throw / count
    slip = throw / np.sin(np.deg2rad(dip)) * rng.choice((-1, 1))
    strike_radius = rng.uniform(0.35, 0.75) * extent
    dip_radius = rng.uniform(0.45, 0.85) * extent
    hanging_fraction = rng.uniform(0.3, 0.7)
    # Bound the slip derivative so each block remains invertible, including
    # configurations with a nominal throw approaching half the volume depth.
    slip_limit = 0.5 * dip_radius / max(hanging_fraction, 1 - hanging_fraction)
    slip = np.clip(slip, -slip_limit, slip_limit)
    u, v = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    points = np.column_stack((u.ravel(), v.ravel()))
    points += rng.uniform(-0.12, 0.12, size=points.shape)
    normal = extent * (
        rng.uniform(-0.05, 0.05) * points[:, 1]**2
        + rng.uniform(-0.012, 0.012, size=len(points))
    )
    faults.append(WuFault(
        reference=(rng.uniform(-0.15, 0.15) * extent,
                   rng.uniform(-0.15, 0.15) * extent,
                   rng.uniform(0.35, 0.65) * extent),
        strike_degrees=rng.uniform(0, 360), dip_degrees=dip, max_slip=slip,
        strike_radius=strike_radius, dip_radius=dip_radius,
        drag_radius=rng.uniform(0.2, 0.45) * extent,
        hanging_wall_fraction=hanging_fraction,
        surface_controls=np.column_stack((points, normal)),
    ))
  return faults


def sample_fold(height, scenario, rng):
  """Samples the linear shear and depth-scaled Gaussian folds of eqs. 1-3."""
  extent = height - 1
  dip = np.zeros(2)
  gaussians = np.empty((0, 4))
  if scenario != 'layered':
    limit = 0.25 if scenario == 'dipping' else 0.12
    dip = rng.uniform(-limit, limit, size=2)
  if scenario in ('folded', 'faulted', 'unconformity', 'mixed'):
    centers = rng.uniform(-0.4, 0.4, size=(4, 2)) * extent
    widths = rng.uniform(0.15, 0.4, size=4) * extent
    amplitudes = rng.uniform(-0.2, 0.2, size=4) * widths
    gaussians = np.column_stack((centers, widths, amplitudes))
  return dip, gaussians


def restore_fold(coordinates, height, dip, gaussians):
  """Analytic inverse of Z' = Z + aX+bY + (1.5/Zmax)*Z*G(X,Y)."""
  x, y, z = coordinates
  gaussian = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))
  for cx, cy, width, amplitude in gaussians:
    gaussian += amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * width**2))
  return (z - dip[0] * x - dip[1] * y) / (1 + 1.5 * gaussian / (height - 1))
