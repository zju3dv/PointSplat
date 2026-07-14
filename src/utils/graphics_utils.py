from math import prod

import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int64
from torch import Tensor
import itertools
from typing import Iterable, Literal, Optional, TypedDict

from torch.utils.data.dataloader import default_collate


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions[..., -1:]

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def sample_training_rays(
    image: Float[Tensor, "batch view channel ..."],
    intrinsics: Float[Tensor, "batch view dim dim"],
    extrinsics: Float[Tensor, "batch view dim+1 dim+1"],
    num_rays: int,
) -> tuple[
    Float[Tensor, "batch ray dim"],  # origins
    Float[Tensor, "batch ray dim"],  # directions
    Float[Tensor, "batch ray 3"],  # sampled color
]:
    device = extrinsics.device
    b, v, _, *grid_shape = image.shape

    # Generate all possible target rays.
    xy, _ = sample_image_grid(tuple(grid_shape), device)
    origins, directions = get_world_rays(
        rearrange(xy, "... d -> ... () () d"),
        extrinsics,
        intrinsics,
    )
    origins = rearrange(origins, "... b v xy -> b (v ...) xy", b=b, v=v)
    directions = rearrange(directions, "... b v xy -> b (v ...) xy", b=b, v=v)
    pixels = rearrange(image, "b v c ... -> b (v ...) c")

    # Sample random rays.
    num_possible_rays = v * prod(grid_shape)
    ray_indices = torch.randint(num_possible_rays, (b, num_rays), device=device)
    batch_indices = repeat(torch.arange(b, device=device), "b -> b n", n=num_rays)

    return (
        origins[batch_indices, ray_indices],
        directions[batch_indices, ray_indices],
        pixels[batch_indices, ray_indices],
    )


def intersect_rays(
    origins_x: Float[Tensor, "*#batch 3"],
    directions_x: Float[Tensor, "*#batch 3"],
    origins_y: Float[Tensor, "*#batch 3"],
    directions_y: Float[Tensor, "*#batch 3"],
    eps: float = 1e-5,
    inf: float = 1e10,
) -> Float[Tensor, "*batch 3"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast the rays so their shapes match.
    shape = torch.broadcast_shapes(
        origins_x.shape,
        directions_x.shape,
        origins_y.shape,
        directions_y.shape,
    )
    origins_x = origins_x.broadcast_to(shape)
    directions_x = directions_x.broadcast_to(shape)
    origins_y = origins_y.broadcast_to(shape)
    directions_y = directions_y.broadcast_to(shape)

    # Detect and remove batch elements where the directions are parallel.
    parallel = einsum(directions_x, directions_y, "... xyz, ... xyz -> ...") > 1 - eps
    origins_x = origins_x[~parallel]
    directions_x = directions_x[~parallel]
    origins_y = origins_y[~parallel]
    directions_y = directions_y[~parallel]

    # Stack the rays into (2, *shape).
    origins = torch.stack([origins_x, origins_y], dim=0)
    directions = torch.stack([directions_x, directions_y], dim=0)
    dtype = origins.dtype
    device = origins.device

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "r b i, r b j -> r b i j")
    n = n - torch.eye(3, dtype=dtype, device=device).broadcast_to((2, 1, 3, 3))

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "r b i j -> b i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "r b i j, r b j -> r b i")
    rhs = reduce(rhs, "r b i -> b i", "sum")

    # Left-matrix-multiply both sides by the pseudo-inverse of lhs to find p.
    result = torch.linalg.lstsq(lhs, rhs).solution

    # Handle the case of parallel lines by setting depth to infinity.
    result_all = torch.ones(shape, dtype=dtype, device=device) * inf
    result_all[~parallel] = result
    return result_all


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def _is_in_bounds(
    xy: Float[Tensor, "*batch 2"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified XY coordinates are within the normalized image plane,
    which has a range from 0 to 1 in each direction.
    """
    return (xy >= -epsilon).all(dim=-1) & (xy <= 1 + epsilon).all(dim=-1)


def _is_in_front_of_camera(
    xyz: Float[Tensor, "*batch 3"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified points in camera space are in front of the camera."""
    return xyz[..., -1] > -epsilon


def _is_positive_t(
    t: Float[Tensor, " *batch"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified t value is positive."""
    return t > -epsilon


class PointProjection(TypedDict):
    t: Float[Tensor, " *batch"]  # ray parameter, as in xyz = origin + t * direction
    xy: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # A "valid" projection satisfies two conditions:
    # 1. It is in front of the camera (i.e., its 3D Z coordinate is positive).
    # 2. It is within the image frame (i.e., its 2D coordinates are between 0 and 1).
    valid: Bool[Tensor, " *batch"]


def _intersect_image_coordinate(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    dimension: Literal["x", "y"],
    coordinate_value: float,
) -> PointProjection:
    """Compute the intersection of the projection of a camera-space ray with a line
    that's parallel to the image frame, either horizontally or vertically.
    """

    # Define shorthands.
    dim = "xy".index(dimension)
    other_dim = 1 - dim
    fs = intrinsics[..., dim, dim]  # focal length, same coordinate
    fo = intrinsics[..., other_dim, other_dim]  # focal length, other coordinate
    cs = intrinsics[..., dim, 2]  # principal point, same coordinate
    co = intrinsics[..., other_dim, 2]  # principal point, other coordinate
    os = origins[..., dim]  # ray origin, same coordinate
    oo = origins[..., other_dim]  # ray origin, other coordinate
    ds = directions[..., dim]  # ray direction, same coordinate
    do = directions[..., other_dim]  # ray direction, other coordinate
    oz = origins[..., 2]  # ray origin, z coordinate
    dz = directions[..., 2]  # ray direction, z coordinate
    c = (coordinate_value - cs) / fs  # coefficient (computed once and factored out)

    # Compute the value of t at the intersection.
    # Note: Infinite values of t are fine. No need to handle division by zero.
    t_numerator = c * oz - os
    t_denominator = ds - c * dz
    t = t_numerator / t_denominator

    # Compute the value of the other coordinate at the intersection.
    # Note: Infinite coordinate values are fine. No need to handle division by zero.
    coordinate_numerator = fo * (oo * (c * dz - ds) + do * (os - c * oz))
    coordinate_denominator = dz * os - ds * oz
    coordinate_other = co + coordinate_numerator / coordinate_denominator
    coordinate_same = torch.ones_like(coordinate_other) * coordinate_value
    xy = [coordinate_same]
    xy.insert(other_dim, coordinate_other)
    xy = torch.stack(xy, dim=-1)
    xyz = origins + t[..., None] * directions

    # These will all have exactly the same batch shape (no broadcasting necessary). In
    # terms of jaxtyping annotations, they all match *batch, not just *#batch.
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz) & _is_positive_t(t),
    }


def _compare_projections(
    intersections: Iterable[PointProjection],
    reduction: Literal["min", "max"],
) -> PointProjection:
    intersections = {k: v.clone() for k, v in default_collate(intersections).items()}
    t = intersections["t"]
    xy = intersections["xy"]
    valid = intersections["valid"]

    # Make sure out-of-bounds values are not chosen.
    lowest_priority = {
        "min": torch.inf,
        "max": -torch.inf,
    }[reduction]
    t[~valid] = lowest_priority

    # Run the reduction (either t.min() or t.max()).
    reduced, selector = getattr(t, reduction)(dim=0)

    # Index the results.
    return {
        "t": reduced,
        "xy": xy.gather(0, repeat(selector, "... -> () ... xy", xy=2))[0],
        "valid": valid.gather(0, selector[None])[0],
    }


def _compute_point_projection(
    xyz: Float[Tensor, "*#batch 3"],
    t: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> PointProjection:
    xy = project_camera_space(xyz, intrinsics)
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz) & _is_positive_t(t),
    }


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def project_rays(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    near: Optional[Float[Tensor, "*#batch"]] = None,
    far: Optional[Float[Tensor, "*#batch"]] = None,
    epsilon: float = 1e-6,
) -> RaySegmentProjection:
    # Transform the rays into camera space.
    world_to_cam = torch.linalg.inv(extrinsics)
    origins = homogenize_points(origins)
    origins = einsum(world_to_cam, origins, "... i j, ... j -> ... i")
    directions = homogenize_vectors(directions)
    directions = einsum(world_to_cam, directions, "... i j, ... j -> ... i")
    origins = origins[..., :3]
    directions = directions[..., :3]

    # Compute intersections with the image's frame.
    frame_intersections = (
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 0.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 1.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 0.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 1.0),
    )
    frame_intersection_min = _compare_projections(frame_intersections, "min")
    frame_intersection_max = _compare_projections(frame_intersections, "max")

    if near is None:
        # Compute the ray's projection at zero depth. If an origin's depth (z value) is
        # within epsilon of zero, this can mean one of two things:
        # 1. The origin is at the camera's position. In this case, use the direction
        #    instead (the ray is probably coming from the camera).
        # 2. The origin isn't at the camera's position, and randomly happens to be on
        #    the plane at zero depth. In this case, its projection is outside the image
        #    plane, and is thus marked as invalid.
        origins_for_projection = origins.clone()
        mask_depth_zero = origins_for_projection[..., -1] < epsilon
        mask_at_camera = origins_for_projection.norm(dim=-1) < epsilon
        origins_for_projection[mask_at_camera] = directions[mask_at_camera]
        projection_at_zero = _compute_point_projection(
            origins_for_projection,
            torch.zeros_like(frame_intersection_min["t"]),
            intrinsics,
        )
        projection_at_zero["valid"][mask_depth_zero & ~mask_at_camera] = False
    else:
        # If a near plane is specified, use it instead.
        t_near = near.broadcast_to(frame_intersection_min["t"].shape)
        projection_at_zero = _compute_point_projection(
            origins + near[..., None] * directions,
            t_near,
            intrinsics,
        )

    if far is None:
        # Compute the ray's projection at infinite depth. Using the projection function
        # with directions (vectors) instead of points may seem wonky, but is equivalent
        # to projecting the point at (origins + infinity * directions).
        projection_at_infinity = _compute_point_projection(
            directions,
            torch.ones_like(frame_intersection_min["t"]) * torch.inf,
            intrinsics,
        )
    else:
        # If a far plane is specified, use it instead.
        t_far = far.broadcast_to(frame_intersection_min["t"].shape)
        projection_at_infinity = _compute_point_projection(
            origins + far[..., None] * directions,
            t_far,
            intrinsics,
        )

    # Build the result by handling cases for ray intersection.
    result = {
        "t_min": torch.empty_like(projection_at_zero["t"]),
        "t_max": torch.empty_like(projection_at_infinity["t"]),
        "xy_min": torch.empty_like(projection_at_zero["xy"]),
        "xy_max": torch.empty_like(projection_at_infinity["xy"]),
        "overlaps_image": torch.empty_like(projection_at_zero["valid"]),
    }

    for min_valid, max_valid in itertools.product([True, False], [True, False]):
        min_mask = projection_at_zero["valid"] ^ (not min_valid)
        max_mask = projection_at_infinity["valid"] ^ (not max_valid)
        mask = min_mask & max_mask
        min_value = projection_at_zero if min_valid else frame_intersection_min
        max_value = projection_at_infinity if max_valid else frame_intersection_max
        result["t_min"][mask] = min_value["t"][mask]
        result["t_max"][mask] = max_value["t"][mask]
        result["xy_min"][mask] = min_value["xy"][mask]
        result["xy_max"][mask] = max_value["xy"][mask]
        result["overlaps_image"][mask] = (min_value["valid"] & max_value["valid"])[mask]

    return result


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def lift_to_3d(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 3"]:
    """Calculate the 3D positions that correspond to the specified 2D points on the
    epipolar lines defined by the origins and directions. The extrinsics and intrinsics
    are for the images the 2D points lie on.
    """

    xy_origins, xy_directions = get_world_rays(xy, extrinsics, intrinsics)
    return intersect_rays(origins, directions, xy_origins, xy_directions)


def get_depth(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, " *batch"]:
    """Calculate the depths that correspond to the specified 2D points on the epipolar
    lines defined by the origins and directions. The extrinsics and intrinsics are for
    the images the 2D points lie on.
    """
    xyz = lift_to_3d(origins, directions, xy, extrinsics, intrinsics)
    return (xyz - origins).norm(dim=-1)


from math import isqrt
from e3nn.o3 import matrix_to_angles, wigner_D
def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    alpha, beta, gamma = matrix_to_angles(rotations)
    result = []
    for degree in range(isqrt(n)):
        with torch.device(device):
            sh_rotations = wigner_D(degree, alpha, beta, gamma).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)