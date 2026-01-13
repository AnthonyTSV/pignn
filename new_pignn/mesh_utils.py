from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import torch
from torch_geometric.data import Data

import ngsolve as ng
from ngsolve import Mesh
from netgen.geom2d import SplineGeometry, CSG2d, Rectangle


# ---------------------------
# Geometry / mesh generation
# ---------------------------


def create_rectangular_mesh(
    width: float = 1.0,
    height: float = 1.0,
    maxh: float = 0.1,
    *,
    bc_bottom: str = "bottom",
    bc_right: str = "right",
    bc_top: str = "top",
    bc_left: str = "left",
) -> Mesh:
    """
    Create a 2D rectangular mesh with named boundary segments.
    """
    geo = SplineGeometry()
    p1 = geo.AppendPoint(0, 0)
    p2 = geo.AppendPoint(width, 0)
    p3 = geo.AppendPoint(width, height)
    p4 = geo.AppendPoint(0, height)

    geo.Append(["line", p1, p2], bc=bc_bottom)
    geo.Append(["line", p2, p3], bc=bc_right)
    geo.Append(["line", p3, p4], bc=bc_top)
    geo.Append(["line", p4, p1], bc=bc_left)

    ngmesh = geo.GenerateMesh(maxh=maxh)
    return ng.Mesh(ngmesh)


def create_lshape_mesh(
    length: float = 1.0,
    height: float = 1.0,
    a_l: float = 0.5,
    a_h: float = 0.5,
    corner: int = 1,
    maxh: float = 0.1,
) -> Mesh:
    """Create a 2D L-shaped mesh as difference of 2 rectangles.
    The small rectangle of size a_l * length x a_h * height is randomly placed at one of the corners
    of the large domain and subtracted from the large rectangle to yield the final L-shaped domain.
    """
    geo = CSG2d()
    # Large rectangle
    large_rect = Rectangle(pmin=(0, 0), pmax=(length, height))
    # Small rectangle position based on corner choice
    if corner == 1:
        small_rect = Rectangle(
            pmin=(length - a_l * length, height - a_h * height), pmax=(length, height)
        )
    elif corner == 2:
        small_rect = Rectangle(
            pmin=(0, height - a_h * height), pmax=(a_l * length, height)
        )
    elif corner == 3:
        small_rect = Rectangle(pmin=(0, 0), pmax=(a_l * length, a_h * height))
    elif corner == 4:
        small_rect = Rectangle(
            pmin=(length - a_l * length, 0), pmax=(length, a_h * height)
        )
    else:
        raise ValueError("corner must be an integer between 1 and 4")
    # L-shape as difference
    lshape = large_rect - small_rect
    lshape = lshape.BC("outer")
    geo.Add(lshape)
    ngmesh = geo.GenerateMesh(maxh=maxh)
    return ng.Mesh(ngmesh)


def create_ih_mesh():
    # Normalization constants
    r_star = 70 * 1e-3  # m
    A_star = 4.8 * 1e-4  # Wb/m
    mu_star = 4 * 3.1415926535e-7 # H/m
    J_star = A_star / (r_star**2 * mu_star)


    workpiece_diameter = 15 * 1e-3 / r_star  # m
    workpiece_height = 70 * 1e-3 / r_star  # m

    coil_diameter = 30 * 1e-3 / r_star  # m outer
    profile_width = 7 * 1e-3 / r_star  # m
    profile_height = 7 * 1e-3 / r_star  # m

    air_height = 3 * workpiece_height  # m
    air_width = 0.12 / r_star  # m

    geo = CSG2d()

    # rect_workpiece = (
    #     Rectangle(
    #         pmin=(0, workpiece_height),
    #         pmax=(workpiece_diameter, 2 * workpiece_height),
    #         mat="mat_workpiece",
    #         # bc="bc_workpiece",
    #         left="bc_workpiece_left",
    #     )
    #     .Mat("mat_workpiece")
    #     .Maxh(5e-3)
    # )

    rect_air = (
        Rectangle(
            pmin=(0, 0),
            pmax=(air_width, air_height),
            mat="mat_air",
            bc="bc_air",
            left="bc_axis",
        )
        .Mat("mat_air")
        .Maxh(60e-3 / r_star)
    )
    rect_coil = (
        Rectangle(
            pmin=(coil_diameter / 2 + profile_width, air_height / 2 - profile_height / 2),
            pmax=(
                coil_diameter / 2 + 2 * profile_width,
                air_height / 2 + profile_height / 2,
            ),
            mat="mat_coil",
            bc="bc_coil",
        )
        .Mat("mat_coil")
        .Maxh(1e-3 / r_star)
    )

    # workpiece = rect_workpiece * rect_air
    coil = rect_coil * rect_air
    air = rect_air - coil

    geo.Add(air)
    geo.Add(coil)

    # generate mesh
    m = geo.GenerateMesh()

    mesh = Mesh(m)
    # mesh.Curve(3)
    return mesh


def create_gaussian_initial_condition(
    pos,
    num_gaussians=4,
    amplitude_range=(0.4, 1.0),
    sigma_fraction_range=(0.05, 0.15),
    seed=None,
    centered=False,
    enforce_boundary_conditions=True,
):
    """
    Create Gaussian initial condition for heat equation that satisfies homogeneous Dirichlet BC.

    Args:
        pos: Node positions (N, 2)
        num_gaussians: Number of Gaussian peaks
        amplitude_range: Range of Gaussian amplitudes
        sigma_fraction_range: Range of sigma as fraction of domain size
        seed: Random seed
        centered: If True, place first Gaussian at domain center
        enforce_boundary_conditions: If True, enforce T=0 on boundaries

    Returns:
        T_initial: Initial temperature field (N,)
    """
    if seed is not None:
        np.random.seed(seed)

    pos = np.array(pos)
    n_nodes = pos.shape[0]

    # Get domain bounds
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    domain_size = max(x_max - x_min, y_max - y_min)

    T_initial = np.zeros(n_nodes)

    for i in range(num_gaussians):
        # Random amplitude
        amplitude = np.random.uniform(*amplitude_range)

        # Random center
        if i == 0 and centered:
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
        else:
            margin = 0
            center_x = np.random.uniform(x_min + margin, x_max - margin)
            center_y = np.random.uniform(y_min + margin, y_max - margin)

        # Random sigma
        sigma = np.random.uniform(*sigma_fraction_range) * domain_size

        # Add Gaussian
        distances_sq = (pos[:, 0] - center_x) ** 2 + (pos[:, 1] - center_y) ** 2
        gaussian = amplitude * np.exp(-distances_sq / (2 * sigma**2))
        T_initial += gaussian

    # Enforce boundary conditions if requested
    if enforce_boundary_conditions:
        # Find boundary nodes (nodes on domain boundary)
        tolerance = 1e-10
        boundary_mask = (
            (np.abs(pos[:, 0] - x_min) < tolerance)  # Left boundary
            | (np.abs(pos[:, 0] - x_max) < tolerance)  # Right boundary
            | (np.abs(pos[:, 1] - y_min) < tolerance)  # Bottom boundary
            | (np.abs(pos[:, 1] - y_max) < tolerance)  # Top boundary
        )
        T_initial[boundary_mask] = 0.0

    return T_initial
