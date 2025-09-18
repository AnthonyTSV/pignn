"""
Mesh utilities for creating meshes and building graphs.
"""

import numpy as np
import ngsolve as ng
from netgen.geom2d import Rectangle
import torch
from torch_geometric.data import Data
import random
from typing import Optional, List, Tuple
from ngsolve import Mesh
from netgen.geom2d import CSG2d, SplineGeometry, Circle
from scipy.spatial import ConvexHull


def create_rectangular_mesh(width=1.0, height=1.0, maxh=0.1, 
                          dirichlet_boundaries=None) -> Mesh:
    """
    Create a rectangular mesh using NGSolve/Netgen.
    
    Args:
        width: Width of rectangle
        height: Height of rectangle 
        maxh: Maximum element size
        dirichlet_boundaries: List of boundary names for Dirichlet conditions
    
    Returns:
        mesh: NGSolve mesh object
    """
    from netgen.geom2d import CSG2d
    
    geo = CSG2d()
    
    # Create rectangle with named boundaries
    rect = Rectangle(pmin=(-width, -height), pmax=(width, height))
    rect.BC("bottom").BC("right").BC("top").BC("left")
    
    geo.Add(rect)
    
    # Generate mesh
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = ng.Mesh(ngmesh)
    
    return mesh


def create_lshape_mesh(maxh: float, seed: Optional[int] = None) -> Mesh:
    """Generate a random 2D L-shaped domain as the difference of rectangles.
    Returns an NGSolve Mesh.
    """
    if seed is not None:
        random.seed(seed)
    geo = CSG2d()

    # Sample outer box size
    L = random.uniform(0.5, 1)
    H = random.uniform(0.5, 1)
    # Arm thicknesses as fractions of box sides
    aL = random.uniform(1/3, 2/3)
    aH = random.uniform(1/3, 2/3)
    tx, ty = aL * L, aH * H

    outer = Rectangle(pmin=(0, 0), pmax=(L, H))

    corner = random.choice(["top-right", "top-left", "bottom-right", "bottom-left"])
    if corner == "top-right":
        cutout = Rectangle(pmin=(tx, ty), pmax=(L, H))
    elif corner == "top-left":
        cutout = Rectangle(pmin=(0, ty), pmax=(L - tx, H))
    elif corner == "bottom-right":
        cutout = Rectangle(pmin=(tx, 0), pmax=(L, H - ty))
    else:  # bottom-left
        cutout = Rectangle(pmin=(0, 0), pmax=(L - tx, H - ty))

    geo.Add(outer - cutout)
    ngm = geo.GenerateMesh(maxh=maxh)
    return Mesh(ngm)


def create_polygon_mesh(maxh: float, num_points: int = 7, domain_size: float = 1.0, 
                       seed: Optional[int] = None) -> Mesh:
    """
    Create a random convex polygon mesh as described in the paper.
    
    Args:
        maxh: Maximum element size
        num_points: Number of points to sample for convex hull
        domain_size: Size of the domain to sample points from
        seed: Random seed
    
    Returns:
        mesh: NGSolve mesh object
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Sample random points in a square domain
    points = np.random.uniform(0, domain_size, (num_points, 2))
    
    # Compute convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Create polygon geometry using SplineGeometry
    geo = SplineGeometry()
    
    # Add points to geometry
    point_indices = []
    for point in hull_points:
        idx = geo.AppendPoint(float(point[0]), float(point[1]))
        point_indices.append(idx)
    
    # Create edges connecting the points in order
    for i in range(len(point_indices)):
        next_i = (i + 1) % len(point_indices)
        geo.Append(["line", point_indices[i], point_indices[next_i]], bc="boundary")
    
    # Generate mesh
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = ng.Mesh(ngmesh)
    
    return mesh


def create_circle_mesh(radius: float = 0.5, center: Tuple[float, float] = (0.5, 0.5),
                      maxh: float = 0.1, seed: Optional[int] = None) -> Mesh:
    """
    Create a circular mesh.
    
    Args:
        radius: Radius of the circle
        center: Center coordinates (x, y)
        maxh: Maximum element size
        seed: Random seed (for potential randomization)
    
    Returns:
        mesh: NGSolve mesh object
    """
    if seed is not None:
        random.seed(seed)
        # Optionally add some randomization to radius or center
        radius_variation = random.uniform(0.9, 1.1)
        radius *= radius_variation
    
    geo = CSG2d()
    
    # Create circle
    circle = Circle(center=center, radius=radius)
    circle.BC("boundary")
    
    geo.Add(circle)
    
    # Generate mesh
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = ng.Mesh(ngmesh)
    
    return mesh


def create_hollow_cylinder_mesh(length: float, outer_radius: float, inner_radius: float,
                               maxh: float = 0.1, seed: Optional[int] = None) -> Mesh:
    """
    Create a 3D hollow cylinder mesh for mixed boundary conditions.
    
    Args:
        length: Length of the cylinder
        outer_radius: Outer radius
        inner_radius: Inner radius
        maxh: Maximum element size
        seed: Random seed
    
    Returns:
        mesh: NGSolve mesh object
    """
    if seed is not None:
        random.seed(seed)
    
    try:
        from netgen.csg import CSG, Cylinder
        
        # Create 3D geometry
        geo = CSG()
        
        # Outer cylinder
        outer_cyl = Cylinder(center=(0, 0, 0), axis=(0, 0, length), radius=outer_radius)
        # Inner cylinder (to subtract)
        inner_cyl = Cylinder(center=(0, 0, 0), axis=(0, 0, length), radius=inner_radius)
        
        # Hollow cylinder = outer - inner
        hollow = outer_cyl - inner_cyl
        
        geo.Add(hollow)
        
        # Generate mesh
        ngmesh = geo.GenerateMesh(maxh=maxh)
        mesh = ng.Mesh(ngmesh)
        
        return mesh
        
    except ImportError:
        # Fallback to 2D if 3D CSG is not available
        print("Warning: 3D CSG not available, creating 2D hollow circle instead")
        return create_hollow_circle_mesh(outer_radius, inner_radius, maxh, seed)


def create_hollow_circle_mesh(outer_radius: float, inner_radius: float,
                             maxh: float = 0.1, seed: Optional[int] = None) -> Mesh:
    """
    Create a 2D hollow circle (annulus) mesh.
    
    Args:
        outer_radius: Outer radius
        inner_radius: Inner radius
        maxh: Maximum element size
        seed: Random seed
    
    Returns:
        mesh: NGSolve mesh object
    """
    if seed is not None:
        random.seed(seed)
    
    geo = CSG2d()
    
    # Create outer and inner circles
    outer_circle = Circle(center=(0, 0), radius=outer_radius)
    inner_circle = Circle(center=(0, 0), radius=inner_radius)
    
    # Set boundary conditions
    outer_circle.BC("outer")
    inner_circle.BC("inner")
    
    # Hollow circle = outer - inner
    hollow = outer_circle - inner_circle
    
    geo.Add(hollow)
    
    # Generate mesh
    ngmesh = geo.GenerateMesh(maxh=maxh)
    mesh = ng.Mesh(ngmesh)
    
    return mesh


# Factory function for creating meshes
def create_mesh(mesh_type: str, **kwargs) -> Mesh:
    """
    Factory function to create different types of meshes.
    
    Args:
        mesh_type: Type of mesh ('rectangular', 'lshape', 'polygon', 'circle', 'hollow_cylinder', 'hollow_circle')
        **kwargs: Mesh-specific parameters
    
    Returns:
        mesh: NGSolve mesh object
    """
    if mesh_type == 'rectangular':
        return create_rectangular_mesh(**kwargs)
    elif mesh_type == 'lshape':
        return create_lshape_mesh(**kwargs)
    elif mesh_type == 'polygon':
        return create_polygon_mesh(**kwargs)
    elif mesh_type == 'circle':
        return create_circle_mesh(**kwargs)
    elif mesh_type == 'hollow_cylinder':
        return create_hollow_cylinder_mesh(**kwargs)
    elif mesh_type == 'hollow_circle':
        return create_hollow_circle_mesh(**kwargs)
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")


def build_graph_from_mesh(mesh):
    """
    Build graph structure from NGSolve mesh.
    
    Args:
        mesh: NGSolve mesh object
    
    Returns:
        graph_data: Dictionary containing:
            - pos: Node positions (N, 2) or (N, 3) for 3D
            - edge_index: Edge connectivity (2, E)
            - boundary_mask: Boolean mask for boundary nodes (N,)
            - elements: Element connectivity for FEM computations
    """
    # Get node positions
    ngmesh = mesh.ngmesh
    points = ngmesh.Points()
    
    # Handle both 2D and 3D meshes
    if len(points) > 0:
        # Extract coordinates from points (use .p attribute and handle 3D coordinates)
        pos_list = []
        for point in points:
            coord = point.p  # Get coordinates
            if mesh.dim == 2:
                # For 2D meshes, use only x and y coordinates
                pos_list.append([coord[0], coord[1]])
            else:
                # For 3D meshes, use all coordinates
                pos_list.append([coord[0], coord[1], coord[2]])
        
        pos = np.array(pos_list, dtype=np.float64)
    else:
        raise ValueError("No points found in mesh")
    
    # Build edge connectivity from elements
    edges = set()
    elements = []
    
    for el in mesh.Elements():
        # Get vertex indices for this element
        verts = [v.nr for v in el.vertices]
        elements.append(verts)
        
        # Add edges between all pairs of vertices in this element
        n_verts = len(verts)
        for i in range(n_verts):
            for j in range(i + 1, n_verts):
                v1, v2 = verts[i], verts[j]
                edges.add((v1, v2))
                edges.add((v2, v1))  # Add both directions for undirected graph
    
    # Convert edges to numpy array
    if edges:
        edge_index = np.array(list(edges)).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    
    # Add self-loops as in MeshGraphNets
    n_nodes = len(pos)
    self_loops = np.stack([np.arange(n_nodes), np.arange(n_nodes)], axis=0)
    edge_index = np.concatenate([edge_index, self_loops], axis=1)
    
    # Identify boundary nodes
    boundary_mask = np.zeros(n_nodes, dtype=bool)
    
    # Mark boundary nodes using 1D boundary elements (2D) or 2D boundary elements (3D)
    if mesh.dim == 2:
        # 2D mesh - use 1D boundary elements
        for seg in mesh.ngmesh.Elements1D():
            if hasattr(seg, 'pnums'):
                pnums = seg.pnums
            elif hasattr(seg, 'points'):
                pnums = seg.points
            elif hasattr(seg, 'vertices'):
                pnums = [v.nr for v in seg.vertices]
            else:
                continue
                
            for pnum in pnums:
                if hasattr(pnum, 'nr'):
                    idx = pnum.nr
                else:
                    idx = int(pnum)
                if 0 <= idx < n_nodes:
                    boundary_mask[idx] = True
    else:
        # 3D mesh - use 2D boundary elements
        for face in mesh.ngmesh.Elements2D():
            if hasattr(face, 'pnums'):
                pnums = face.pnums
            elif hasattr(face, 'points'):
                pnums = face.points
            elif hasattr(face, 'vertices'):
                pnums = [v.nr for v in face.vertices]
            else:
                continue
                
            for pnum in pnums:
                if hasattr(pnum, 'nr'):
                    idx = pnum.nr
                else:
                    idx = int(pnum)
                if 0 <= idx < n_nodes:
                    boundary_mask[idx] = True
    
    graph_data = {
        'pos': pos,
        'edge_index': edge_index,
        'boundary_mask': boundary_mask,
        'elements': elements
    }
    
    return graph_data


def create_gaussian_initial_condition(pos, num_gaussians=4, amplitude_range=(0.4, 1.0),
                                    sigma_fraction_range=(0.05, 0.15), seed=None, centered=False):
    """
    Create Gaussian initial condition for heat equation.
    
    Args:
        pos: Node positions (N, 2)
        num_gaussians: Number of Gaussian peaks
        amplitude_range: Range of Gaussian amplitudes
        sigma_fraction_range: Range of sigma as fraction of domain size
        seed: Random seed
        centered: If True, place first Gaussian at domain center
    
    Returns:
        T0: Initial temperature field (N,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_nodes = pos.shape[0]
    T0 = np.zeros(n_nodes)
    
    # Domain bounds for sigma scaling
    x_min, y_min = pos.min(axis=0)
    x_max, y_max = pos.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    
    # Center of the domain
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    for _ in range(num_gaussians):
        # Use center coordinates for all Gaussians (or just the first one)
        if centered and _ == 0:
            # First Gaussian at center
            gauss_center_x = center_x
            gauss_center_y = center_y
        else:
            # Additional Gaussians can still be random if desired
            gauss_center_x = np.random.uniform(x_min, x_max)
            gauss_center_y = np.random.uniform(y_min, y_max)
        
        # Random amplitude
        amplitude = np.random.uniform(*amplitude_range)
        
        # Random sigma based on domain size
        sigma_x = np.random.uniform(*sigma_fraction_range) * width
        sigma_y = np.random.uniform(*sigma_fraction_range) * height
        
        # Add Gaussian to initial condition
        dx = pos[:, 0] - gauss_center_x
        dy = pos[:, 1] - gauss_center_y
        gaussian = amplitude * np.exp(-(dx**2 / (2 * sigma_x**2) + dy**2 / (2 * sigma_y**2)))
        T0 += gaussian
    
    return T0


def mesh_to_pyg_data(graph_data, T_current, alpha, dt, device='cpu'):
    """
    Convert mesh data to PyTorch Geometric Data object.
    
    Args:
        graph_data: Graph data from build_graph_from_mesh
        T_current: Current temperature values
        alpha: Thermal diffusivity
        dt: Time step
        device: PyTorch device
    
    Returns:
        data: PyTorch Geometric Data object
    """
    from ..model.pimgn import build_features
    return build_features(graph_data, T_current, alpha, dt, device)
