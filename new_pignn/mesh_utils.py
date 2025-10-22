from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import torch
from torch_geometric.data import Data

import ngsolve as ng
from ngsolve import Mesh
from netgen.geom2d import SplineGeometry


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
    p1 = geo.AppendPoint(-width, -height)
    p2 = geo.AppendPoint(width, -height)
    p3 = geo.AppendPoint(width, height)
    p4 = geo.AppendPoint(-width, height)

    geo.Append(["line", p1, p2], bc=bc_bottom)
    geo.Append(["line", p2, p3], bc=bc_right)
    geo.Append(["line", p3, p4], bc=bc_top)
    geo.Append(["line", p4, p1], bc=bc_left)

    ngmesh = geo.GenerateMesh(maxh=maxh)
    return ng.Mesh(ngmesh)


# ---------------------------
# Helpers for Netgen indexing
# ---------------------------

def _build_point_index_map(mesh: Mesh) -> tuple[np.ndarray, dict[int, int]]:
    """
    Build a zero-based contiguous index for netgen points.
    Returns:
      - pos: (N, dim)
      - pnum_to_idx: mapping from Netgen point number (pt.nr, often 1-based) -> [0..N-1]
    """
    ngmesh = mesh.ngmesh
    pts = list(ngmesh.Points())
    dim = mesh.dim

    pos = np.zeros((len(pts), dim), dtype=np.float64)
    pnum_to_idx: dict[int, int] = {}

    for new_idx, pt in enumerate(pts):
        old_nr = int(getattr(pt, "nr", new_idx + 1))  # pt.nr is typical
        pnum_to_idx[old_nr] = new_idx
        coord = pt.p
        if dim == 2:
            pos[new_idx] = [coord[0], coord[1]]
        else:
            pos[new_idx] = [coord[0], coord[1], coord[2]]

    return pos, pnum_to_idx


def _normalize_ids_to_idx(
    raw_ids: list,                  # numbers or objects with .nr
    pnum_to_idx: dict[int, int],    # mapping from Netgen point number -> contiguous idx
    N: int,                         # number of points
    *, context: str = "unknown"
) -> list[int]:
    """
    Robustly convert a list of 'point numbers' coming from various Netgen/NGSolve
    APIs to contiguous [0..N-1] indices.
    Tries exact, +1, -1 offsets; as a fallback, accepts 0-based [0..N-1] directly.
    Raises a clear error if nothing fits.
    """
    # Extract raw integers (handle objects with .nr)
    nums = []
    for v in raw_ids:
        if v is None:
            continue
        if hasattr(v, "nr"):
            nums.append(int(v.nr))
        else:
            nums.append(int(v))

    if not nums:
        return []

    def all_in(mapping, seq):
        return all(x in mapping for x in seq)

    # Try exact mapping
    if all_in(pnum_to_idx, nums):
        return [pnum_to_idx[x] for x in nums]

    # Try +1 (elements 0-based, pnum_to_idx 1-based)
    if all_in(pnum_to_idx, [x + 1 for x in nums]):
        return [pnum_to_idx[x + 1] for x in nums]

    # Try -1 (elements 2-based, unlikely but safe)
    if all_in(pnum_to_idx, [x - 1 for x in nums]):
        return [pnum_to_idx[x - 1] for x in nums]

    # Accept direct 0-based indices if they look like [0..N-1]
    if all((0 <= x < N) for x in nums):
        return nums

    # Nothing matched
    keys = sorted(pnum_to_idx.keys())
    raise KeyError(
        f"[{context}] Cannot map point numbers {nums} to indices.\n"
        f"Available pt.nr keys: min={keys[0] if keys else None}, max={keys[-1] if keys else None}, count={len(keys)}\n"
        f"Hint: this usually means a 0/1-based mismatch; ensure geometry was not reindexed after mesh export."
    )


def _segment_bcnr_to_name_map(ngmesh) -> dict[int, str]:
    """
    Build mapping from segment BC number (as stored on elements) -> name.
    Robust to 0/1-based differences in GetBCName(); falls back to synthetic names.
    """
    mapping: dict[int, str] = {}
    get_name = getattr(ngmesh, "GetBCName", None)
    # Collect all BC codes present on 1D boundary elements
    seen_codes = sorted({
        int(getattr(seg, "bc", getattr(seg, "si", getattr(seg, "index", 0))))
        for seg in ngmesh.Elements1D()
    })

    if callable(get_name) and seen_codes:
        # Try to determine whether GetBCName is 0-based or 1-based
        def try_name(idx: int) -> str | None:
            try:
                name = get_name(idx)
                return None if name is None else str(name)
            except Exception:
                return None

        # Probe with the smallest seen code
        # c0 = seen_codes[0]
        # name_at_c  = try_name(c0)
        # name_at_c1 = try_name(c0 - 1)

        # # Decide offset: if calling with (c0-1) works and (c0) fails, it's 0-based
        # # If calling with c0 works, it's 1-based (or both valid -> prefer exact match)
        # if name_at_c1 and not name_at_c:
        #     offset = -1  # 1-based codes on segments; GetBCName expects 0-based
        # else:
        #     offset = 0   # GetBCName aligns with segment codes

        # Build mapping, but guard against out-of-range
        for c in seen_codes:
            name = try_name(c-1)
            if not name:
                # As a last resort, give it a synthetic name
                name = f"bc_{c}"
            mapping[c] = name
        return mapping

    # Fallback: API not available -> synthesize names based on codes
    for c in seen_codes:
        mapping[c] = f"bc_{c}"
    return mapping


def _resolve_boundary_names_to_bcnr(
    mesh,
    names_or_ids,  # Iterable[str | int]
) -> list[int]:
    """
    Resolve boundary selectors to BC numbers used on segments.
    Accepts either strings (names) or ints (exact BC codes).
    """
    if names_or_ids is None:
        return []

    # Split inputs into strings and ints
    name_like = []
    id_like = []
    for v in names_or_ids:
        if isinstance(v, (int, np.integer)):
            id_like.append(int(v))
        else:
            name_like.append(str(v))

    ngmesh = mesh.ngmesh
    bcnr_to_name = _segment_bcnr_to_name_map(ngmesh)

    # If names are provided, invert the mapping
    resolved_from_names: list[int] = []
    if name_like:
        # Build name -> [codes] map (names can be reused across segments)
        name_to_codes: dict[str, list[int]] = {}
        for code, nm in bcnr_to_name.items():
            name_to_codes.setdefault(nm, []).append(code)

        missing = [nm for nm in name_like if nm not in name_to_codes]
        if missing:
            available = sorted(set(bcnr_to_name.values()))
            raise ValueError(
                "Some boundary names were not found.\n"
                f" Missing: {missing}\n"
                f" Available: {available}\n"
                " Tip: you can also pass integer BC IDs instead of names."
            )
        for nm in name_like:
            resolved_from_names.extend(name_to_codes[nm])

    # Integer IDs are taken as-is (validate they exist on the mesh)
    seen_codes = set(bcnr_to_name.keys())
    invalid_ids = [c for c in id_like if c not in seen_codes]
    if invalid_ids:
        raise ValueError(
            f"Boundary IDs not present on this mesh: {sorted(set(invalid_ids))}. "
            f"Seen BC IDs: {sorted(seen_codes)}"
        )

    resolved = sorted(set(resolved_from_names + id_like))
    return resolved


# ---------------------------
# Graph construction
# ---------------------------

def build_graph_from_mesh(
    mesh: Mesh,
    dirichlet_names: List[str] = None,
    neumann_names: List[str] = None,
    T_current: Optional[np.ndarray] = None,
    t_scalar: float = 0.0,
    material_node_field: Optional[np.ndarray] = None,
    neumann_values: Optional[np.ndarray] = None,
    dirichlet_values: Optional[np.ndarray] = None,
    connectivity_method: str = "fem",
    n_neighbors: int = 8,
    add_self_loops: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[Data, Dict]:
    """
    Build a PyTorch Geometric graph from an NGSolve mesh according to PI-MGN methodology.
    
    Args:
        mesh: NGSolve mesh
        dirichlet_names: List of Dirichlet boundary names
        neumann_names: List of Neumann boundary names  
        T_current: Current temperature field (N,) - optional
        t_scalar: Current time scalar
        material_node_field: Per-node material properties (N,) - optional
        neumann_values: Per-node Neumann boundary values (N,) - optional, h_N values for flux BC
        dirichlet_values: Per-node Dirichlet boundary values (N,) - optional, prescribed values for Dirichlet BC
        connectivity_method: "fem" (mesh connectivity) or "knn" (k-nearest neighbors)
        n_neighbors: Number of neighbors for k-NN connectivity
        add_self_loops: Whether to add self-loops to all nodes
        device: Target device for tensors (CPU if None)
        
    Returns:
        data: PyTorch Geometric Data object
        aux: Auxiliary data dictionary containing mappings and metadata
    """
    if device is None:
        device = torch.device('cpu')
    
    # 1. Build node positions and index mapping
    pos, pnum_to_idx = _build_point_index_map(mesh)
    n_nodes = len(pos)
    
    # Convert to torch tensor on target device
    pos_tensor = torch.tensor(pos, dtype=torch.float32, device=device)
    
    # 2. Determine node types (Dirichlet/Neumann/Interior)
    node_types = _classify_node_types(mesh, dirichlet_names or [], neumann_names or [], pnum_to_idx)
    node_types = node_types.to(device)
    
    # 3. Build connectivity based on method
    if connectivity_method.lower() == "fem":
        edge_index = _build_fem_connectivity(mesh, pnum_to_idx, add_self_loops, device)
    elif connectivity_method.lower() == "knn":
        edge_index = _build_knn_connectivity(pos_tensor, n_neighbors, add_self_loops, device)
    else:
        raise ValueError(f"Unknown connectivity method: {connectivity_method}")
        
    # 4. Build node features
    node_features = _build_node_features(
        node_types, T_current, t_scalar, material_node_field, neumann_values, dirichlet_values, n_nodes, device
    )
    
    # 5. Build edge features
    edge_features = _build_edge_features(edge_index, pos_tensor, T_current, device)
    
    # 6. Build global features
    global_features = _build_global_features(t_scalar, n_nodes, device)
    
    # 7. Create auxiliary data
    aux = {
        'pnum_to_idx': pnum_to_idx,
        'node_types': node_types,
        'dirichlet_mask': node_types == 0,
        'neumann_mask': node_types == 1, 
        'interior_mask': node_types == 2,
        'free_mask': node_types != 0,  # Non-Dirichlet nodes
        'neumann_values': neumann_values,
        'dirichlet_values': dirichlet_values,
        'mesh': mesh,
        'connectivity_method': connectivity_method
    }
    
    # 8. Create PyTorch Geometric Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        pos=pos_tensor,
        global_attr=global_features,
        num_nodes=n_nodes
    )
    
    return data, aux


def create_free_node_subgraph(data: Data, aux: Dict) -> Tuple[Data, Dict, Dict]:
    """
    Create subgraph containing only free (non-Dirichlet) nodes for strict BC enforcement.
    
    Args:
        data: Original graph data
        aux: Auxiliary data from build_graph_from_mesh
        
    Returns:
        free_data: Subgraph with only free nodes
        node_mapping: Mapping from subgraph node indices to original node indices
        free_aux: Auxiliary data for the free node subgraph
    """
    free_mask = aux['free_mask']
    
    # Ensure all tensors are on the same device as the data
    device = data.x.device if hasattr(data.x, 'device') else torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    free_mask = free_mask.to(device)
    
    # Move all aux tensors to the same device
    aux_tensors_to_move = ['node_types', 'dirichlet_mask', 'neumann_mask', 'interior_mask']
    for key in aux_tensors_to_move:
        if key in aux and hasattr(aux[key], 'to'):
            aux[key] = aux[key].to(device)
    
    free_indices = torch.where(free_mask)[0]
    n_free = len(free_indices)
    
    if n_free == 0:
        raise ValueError("No free nodes found - all nodes are on Dirichlet boundary")
    
    # Create mapping from original indices to subgraph indices
    old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
    old_to_new[free_indices] = torch.arange(n_free, device=device)
    
    # Extract free node features and positions
    free_x = data.x[free_indices]
    free_pos = data.pos[free_indices] 
    
    # Ensure edge_index is on the same device
    edge_index = data.edge_index.to(device)
    
    # Filter edges to only include connections between free nodes
    edge_mask = free_mask[edge_index[0]] & free_mask[edge_index[1]]
    free_edges = edge_index[:, edge_mask]
    free_edge_attr = data.edge_attr[edge_mask]
    
    # Remap edge indices to subgraph
    free_edge_index = old_to_new[free_edges]
    
    # Create subgraph data
    free_data = Data(
        x=free_x,
        edge_index=free_edge_index,
        edge_attr=free_edge_attr,
        pos=free_pos,
        global_attr=data.global_attr,
        num_nodes=n_free
    )

    free_aux = {
        'pnum_to_idx': {k: v for k, v in aux['pnum_to_idx'].items() if v in free_indices.tolist()},
        'node_types': aux['node_types'][free_indices],
        'dirichlet_mask': aux['dirichlet_mask'][free_indices],
        'neumann_mask': aux['neumann_mask'][free_indices],
        'interior_mask': aux['interior_mask'][free_indices],
        'free_mask': torch.ones(n_free, dtype=torch.bool, device=device),  # All nodes in subgraph are free
        'neumann_values': aux['neumann_values'][free_indices.cpu().numpy()] if aux['neumann_values'] is not None else None,
        'dirichlet_values': aux['dirichlet_values'][free_indices.cpu().numpy()] if aux['dirichlet_values'] is not None else None,
        'mesh': aux['mesh'],
        'connectivity_method': aux['connectivity_method']
    }
    
    # Node mapping for reconstruction
    node_mapping = {
        'free_to_original': free_indices,
        'original_to_free': old_to_new,
        'n_original': data.num_nodes,
        'n_free': n_free
    }
    
    return free_data, node_mapping, free_aux


def _classify_node_types(
    mesh: Mesh, 
    dirichlet_names: List[str], 
    neumann_names: List[str],
    pnum_to_idx: Dict[int, int]
) -> torch.Tensor:
    """
    Classify nodes as Dirichlet (0), Neumann (1), or Interior (2).
    Dirichlet conditions take priority over Neumann at corner nodes.
    """
    n_nodes = len(pnum_to_idx)
    node_types = torch.full((n_nodes,), 2, dtype=torch.long)  # Default: interior
    
    # Get boundary segments
    segments_1d = list(mesh.ngmesh.Elements1D())
    if not segments_1d:
        return node_types  # No boundaries - all interior
    
    # Resolve boundary names to BC numbers
    dirichlet_bcnr = _resolve_boundary_names_to_bcnr(mesh, dirichlet_names)
    neumann_bcnr = _resolve_boundary_names_to_bcnr(mesh, neumann_names)
    
    # First pass: Mark Neumann nodes
    for seg in segments_1d:
        bc_code = int(getattr(seg, "bc", getattr(seg, "si", getattr(seg, "index", 0))))
        if bc_code in neumann_bcnr:
            vertices = seg.vertices
            vertex_indices = _normalize_ids_to_idx(vertices, pnum_to_idx, n_nodes, context="boundary_segments")
            node_types[vertex_indices] = 1  # Neumann
    
    # Second pass: Mark Dirichlet nodes (overrides Neumann at corners)
    for seg in segments_1d:
        bc_code = int(getattr(seg, "bc", getattr(seg, "si", getattr(seg, "index", 0))))
        if bc_code in dirichlet_bcnr:
            vertices = seg.vertices
            vertex_indices = _normalize_ids_to_idx(vertices, pnum_to_idx, n_nodes, context="boundary_segments")
            node_types[vertex_indices] = 0  # Dirichlet (takes priority)
    
    return node_types


def _build_fem_connectivity(
    mesh: Mesh, 
    pnum_to_idx: Dict[int, int], 
    add_self_loops: bool = True,
    device: torch.device = None
) -> torch.Tensor:
    """
    Build graph connectivity from FEM mesh topology (directed edges + self-loops).
    """
    edges_set = set()
    n_nodes = len(pnum_to_idx)
    
    # Add edges from 2D elements
    for elem in mesh.ngmesh.Elements2D():
        vertices = elem.vertices
        vertex_indices = _normalize_ids_to_idx(vertices, pnum_to_idx, n_nodes, context="2D_elements")
        
        # Add edges between all pairs of vertices in element (fully connected)
        for i, vi in enumerate(vertex_indices):
            for j, vj in enumerate(vertex_indices):
                if i != j:  # No self-loops from element connectivity
                    edges_set.add((vi, vj))
    
    # Add edges from 1D boundary elements
    for elem in mesh.ngmesh.Elements1D():
        vertices = elem.vertices
        vertex_indices = _normalize_ids_to_idx(vertices, pnum_to_idx, n_nodes, context="1D_elements")
        
        # Add bidirectional edges between adjacent vertices
        for i in range(len(vertex_indices)):
            for j in range(len(vertex_indices)):
                if i != j:
                    edges_set.add((vertex_indices[i], vertex_indices[j]))
    
    # Add self-loops if requested
    if add_self_loops:
        for i in range(n_nodes):
            edges_set.add((i, i))
    
    # Convert to tensor
    if device is None:
        device = torch.device('cpu')
    
    if edges_set:
        edges = list(edges_set)
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    return edge_index


def _build_knn_connectivity(
    pos: torch.Tensor, 
    k: int, 
    add_self_loops: bool = True,
    device: torch.device = None
) -> torch.Tensor:
    """
    Build k-nearest neighbor connectivity.
    """
    n_nodes = pos.shape[0]
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(pos, pos, p=2)
    
    # Find k+1 nearest neighbors (including self if self-loops enabled)
    k_actual = k + 1 if add_self_loops else k
    _, knn_indices = torch.topk(dist_matrix, k_actual, dim=1, largest=False)
    
    # Build edge list
    edges = []
    for i in range(n_nodes):
        neighbors = knn_indices[i]
        if not add_self_loops:
            neighbors = neighbors[neighbors != i]  # Remove self
        for j in neighbors:
            edges.append([i, j.item()])
    
    if device is None:
        device = pos.device
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    return edge_index


def _build_node_features(
    node_types: torch.Tensor,
    T_current: Optional[np.ndarray],
    t_scalar: float,
    material_field: Optional[np.ndarray],
    neumann_values: Optional[np.ndarray],
    dirichlet_values: Optional[np.ndarray],
    n_nodes: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Build node feature matrix according to PI-MGN methodology.
    Features: [one_hot_node_type(3), T_current(1), t_scalar(1), material(1), neumann_value(1), dirichlet_value(1)]
    """
    if device is None:
        device = node_types.device
    
    features = []
    
    # One-hot encoding of node types (Dirichlet=0, Neumann=1, Interior=2)
    node_type_onehot = torch.zeros(n_nodes, 3, device=device)
    node_type_onehot[torch.arange(n_nodes, device=device), node_types] = 1.0
    features.append(node_type_onehot)
    
    # Current temperature field
    if T_current is not None:
        T_tensor = torch.tensor(T_current, dtype=torch.float32, device=device).unsqueeze(1)
    else:
        T_tensor = torch.zeros(n_nodes, 1, device=device)
    features.append(T_tensor)
    
    # Time scalar (broadcasted to all nodes)
    t_tensor = torch.full((n_nodes, 1), t_scalar, dtype=torch.float32, device=device)
    features.append(t_tensor)
    
    # Material field
    if material_field is not None:
        mat_tensor = torch.tensor(material_field, dtype=torch.float32, device=device).unsqueeze(1)
    else:
        mat_tensor = torch.ones(n_nodes, 1, device=device)  # Default material
    features.append(mat_tensor)
    
    # Neumann boundary values (flux values h_N from paper)
    if neumann_values is not None:
        neumann_tensor = torch.tensor(neumann_values, dtype=torch.float32, device=device).unsqueeze(1)
    else:
        neumann_tensor = torch.zeros(n_nodes, 1, device=device)  # Default: no flux
    features.append(neumann_tensor)
    
    # Dirichlet boundary values (prescribed values for Dirichlet BC)
    if dirichlet_values is not None:
        dirichlet_tensor = torch.tensor(dirichlet_values, dtype=torch.float32, device=device).unsqueeze(1)
    else:
        dirichlet_tensor = torch.zeros(n_nodes, 1, device=device)  # Default: homogeneous Dirichlet
    features.append(dirichlet_tensor)
    
    return torch.cat(features, dim=1)


def _build_edge_features(
    edge_index: torch.Tensor, 
    pos: torch.Tensor,
    T_current: Optional[np.ndarray],
    device: torch.device = None
) -> torch.Tensor:
    """
    Build edge feature matrix according to PI-MGN methodology.
    Features: [relative_vector(2/3), euclidean_distance(1), temp_difference(1)]
    """
    if device is None:
        device = pos.device
    
    if edge_index.numel() == 0:
        return torch.empty((0, pos.shape[1] + 2), dtype=torch.float32, device=device)
    
    # Get source and target positions
    src_pos = pos[edge_index[0]]  # Source node positions
    dst_pos = pos[edge_index[1]]  # Target node positions
    
    # Relative position vector (target - source)
    relative_pos = dst_pos - src_pos
    
    # Euclidean distance
    euclidean_dist = torch.norm(relative_pos, dim=1, keepdim=True)
    
    # Temperature difference (if available)
    if T_current is not None:
        T_tensor = torch.tensor(T_current, dtype=torch.float32, device=device)
        src_temp = T_tensor[edge_index[0]]
        dst_temp = T_tensor[edge_index[1]]
        temp_diff = (dst_temp - src_temp).unsqueeze(1)
    else:
        temp_diff = torch.zeros(edge_index.shape[1], 1, device=device)
    
    # Concatenate all edge features
    edge_features = torch.cat([relative_pos, euclidean_dist, temp_diff], dim=1)
    
    return edge_features


def _build_global_features(t_scalar: float, n_nodes: int, device: torch.device = None) -> torch.Tensor:
    """
    Build global feature vector.
    """
    if device is None:
        device = torch.device('cpu')
    global_features = torch.tensor([t_scalar, float(n_nodes)], dtype=torch.float32, device=device)
    return global_features

def create_gaussian_initial_condition(pos, num_gaussians=4, amplitude_range=(0.4, 1.0),
                                    sigma_fraction_range=(0.05, 0.15), seed=None, centered=False,
                                    enforce_boundary_conditions=True):
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
            # Place Gaussians away from boundaries
            margin = 0.1 * domain_size
            center_x = np.random.uniform(x_min + margin, x_max - margin)
            center_y = np.random.uniform(y_min + margin, y_max - margin)
        
        # Random sigma
        sigma = np.random.uniform(*sigma_fraction_range) * domain_size
        
        # Add Gaussian
        distances_sq = (pos[:, 0] - center_x)**2 + (pos[:, 1] - center_y)**2
        gaussian = amplitude * np.exp(-distances_sq / (2 * sigma**2))
        T_initial += gaussian
    
    # Enforce boundary conditions if requested
    if enforce_boundary_conditions:
        # Find boundary nodes (nodes on domain boundary)
        tolerance = 1e-10
        boundary_mask = (
            (np.abs(pos[:, 0] - x_min) < tolerance) |  # Left boundary
            (np.abs(pos[:, 0] - x_max) < tolerance) |  # Right boundary
            (np.abs(pos[:, 1] - y_min) < tolerance) |  # Bottom boundary
            (np.abs(pos[:, 1] - y_max) < tolerance)    # Top boundary
        )
        T_initial[boundary_mask] = 0.0
    
    return T_initial


def _get_nodes_on_boundary(mesh: Mesh, boundary_name: str, pnum_to_idx: Dict[int, int]) -> List[int]:
    """
    Get list of node indices that lie on a specific boundary.
    
    Args:
        mesh: NGSolve mesh
        boundary_name: Name of the boundary
        pnum_to_idx: Mapping from Netgen point numbers to indices
        
    Returns:
        List of node indices on the specified boundary
    """
    n_nodes = len(pnum_to_idx)
    boundary_nodes = set()
    
    # Get boundary segments
    segments_1d = list(mesh.ngmesh.Elements1D())
    if not segments_1d:
        return []
    
    # Resolve boundary name to BC number
    boundary_bcnr = _resolve_boundary_names_to_bcnr(mesh, [boundary_name])
    
    # Find all nodes on this boundary
    for seg in segments_1d:
        bc_code = int(getattr(seg, "bc", getattr(seg, "si", getattr(seg, "index", 0))))
        if bc_code in boundary_bcnr:
            vertices = seg.vertices
            vertex_indices = _normalize_ids_to_idx(vertices, pnum_to_idx, n_nodes, context="boundary_segments")
            boundary_nodes.update(vertex_indices)
    
    return list(boundary_nodes)


def create_neumann_values(pos, aux_data, neumann_names, flux_values=None, flux_magnitude=1.0, seed=None):
    """
    Create Neumann boundary values (flux values h_N) according to PI-MGN methodology.
    
    Args:
        pos: Node positions (N, 2) - can be torch tensor or numpy array
        aux_data: Auxiliary data containing node type information
        neumann_names: List of Neumann boundary names or single boundary name
        flux_values: Dict mapping boundary names to flux values, or single value for constant flux
        flux_magnitude: Default flux magnitude if flux_values not provided (backward compatibility)
        seed: Random seed for reproducible values
        
    Returns:
        neumann_values: Per-node Neumann flux values (N,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert pos to numpy if it's a torch tensor
    if hasattr(pos, 'numpy'):
        pos_np = pos.numpy()
    else:
        pos_np = np.array(pos)
    
    n_nodes = pos_np.shape[0]
    neumann_values = np.zeros(n_nodes)
    
    # Get mesh from aux data
    mesh = aux_data['mesh']
    pnum_to_idx = aux_data['pnum_to_idx']
    
    # Handle different input formats for neumann_names
    if isinstance(neumann_names, str):
        neumann_names = [neumann_names]
    
    # Handle different input formats for flux_values
    if flux_values is None:
        # Use default flux_magnitude for all boundaries
        flux_values = {name: flux_magnitude for name in neumann_names}
    elif isinstance(flux_values, (int, float)):
        # Single constant value for all boundaries
        flux_values = {name: flux_values for name in neumann_names}
    elif not isinstance(flux_values, dict):
        raise ValueError("flux_values must be None, a number, or a dict mapping boundary names to values")
    
    # Assign flux values based on boundary names
    for boundary_name in neumann_names:
        if boundary_name not in flux_values:
            raise ValueError(f"No flux value specified for boundary '{boundary_name}'")
        
        # Get nodes on this specific boundary
        boundary_nodes = _get_nodes_on_boundary(mesh, boundary_name, pnum_to_idx)
        flux_value = flux_values[boundary_name]
        
        for node_idx in boundary_nodes:
            neumann_values[node_idx] = flux_value
    
    return neumann_values


def create_dirichlet_values(pos, aux_data, dirichlet_names, boundary_values=None, 
                           temperature_function=None, homogeneous_value=0.0, 
                           inhomogeneous_type="constant", seed=None):
    """
    Create Dirichlet boundary values according to PI-MGN methodology.
    
    For nodes that belong to multiple Dirichlet boundaries, higher values take priority
    over lower values. This ensures consistent boundary condition enforcement when
    boundaries intersect at corners or edges.
    
    Args:
        pos: Node positions (N, 2) - can be torch tensor or numpy array
        aux_data: Auxiliary data containing node type information
        dirichlet_names: List of Dirichlet boundary names or single boundary name
        boundary_values: Dict mapping boundary names to constant values, or single value for all boundaries
        temperature_function: Custom function f(x, y) -> temperature (optional, overrides boundary_values)
        homogeneous_value: Default value if boundary_values not provided (backward compatibility)
        inhomogeneous_type: Type of BC - "constant" (use boundary_values), or legacy types for backward compatibility
        seed: Random seed for reproducible values
        
    Returns:
        dirichlet_values: Per-node Dirichlet boundary values (N,) with higher values prioritized at corner nodes
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert pos to numpy if it's a torch tensor
    if hasattr(pos, 'numpy'):
        pos_np = pos.numpy()
    else:
        pos_np = np.array(pos)
    
    n_nodes = pos_np.shape[0]
    dirichlet_values = np.zeros(n_nodes)
    
    # Get mesh from aux data
    mesh = aux_data['mesh']
    pnum_to_idx = aux_data['pnum_to_idx']
    
    # Handle different input formats for dirichlet_names
    if isinstance(dirichlet_names, str):
        dirichlet_names = [dirichlet_names]
    
    # If temperature_function is provided, use it (legacy behavior)
    if temperature_function is not None:
        dirichlet_mask = aux_data['dirichlet_mask']
        if hasattr(dirichlet_mask, 'numpy'):
            dirichlet_mask_np = dirichlet_mask.numpy()
        else:
            dirichlet_mask_np = np.array(dirichlet_mask)
        
        if dirichlet_mask_np.any():
            dirichlet_indices = np.where(dirichlet_mask_np)[0]
            for idx in dirichlet_indices:
                x, y = pos_np[idx]
                dirichlet_values[idx] = temperature_function(x, y)
        return dirichlet_values
    
    # Handle boundary-specific constant values
    if inhomogeneous_type == "constant":
        # Handle different input formats for boundary_values
        if boundary_values is None:
            # Use default homogeneous_value for all boundaries
            boundary_values = {name: homogeneous_value for name in dirichlet_names}
        elif isinstance(boundary_values, (int, float)):
            # Single constant value for all boundaries
            boundary_values = {name: boundary_values for name in dirichlet_names}
        elif not isinstance(boundary_values, dict):
            raise ValueError("boundary_values must be None, a number, or a dict mapping boundary names to values")
        
        # Assign values based on boundary names with priority (higher values override lower ones)
        for boundary_name in dirichlet_names:
            if boundary_name not in boundary_values:
                raise ValueError(f"No value specified for boundary '{boundary_name}'")
            
            # Get nodes on this specific boundary
            boundary_nodes = _get_nodes_on_boundary(mesh, boundary_name, pnum_to_idx)
            boundary_value = boundary_values[boundary_name]
            
            for node_idx in boundary_nodes:
                # Only assign if this value is higher than the current value
                # This prioritizes higher Dirichlet values over lower ones
                if boundary_value > dirichlet_values[node_idx]:
                    dirichlet_values[node_idx] = boundary_value
    
    else:
        # Legacy behavior - use old inhomogeneous types on all Dirichlet boundaries
        dirichlet_mask = aux_data['dirichlet_mask']
        if hasattr(dirichlet_mask, 'numpy'):
            dirichlet_mask_np = dirichlet_mask.numpy()
        else:
            dirichlet_mask_np = np.array(dirichlet_mask)
        
        if dirichlet_mask_np.any():
            dirichlet_indices = np.where(dirichlet_mask_np)[0]
            
            if inhomogeneous_type == "homogeneous":
                # Homogeneous Dirichlet BC (constant value)
                for idx in dirichlet_indices:
                    dirichlet_values[idx] = homogeneous_value
            
            elif inhomogeneous_type == "sinusoidal":
                # Sinusoidal variation along boundaries
                for idx in dirichlet_indices:
                    x, y = pos_np[idx]
                    dirichlet_values[idx] = np.sin(np.pi * x) * np.cos(np.pi * y)
            
            elif inhomogeneous_type == "linear":
                # Linear variation based on position
                x_coords = pos_np[dirichlet_indices, 0]
                y_coords = pos_np[dirichlet_indices, 1]
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                for idx in dirichlet_indices:
                    x, y = pos_np[idx]
                    # Linear interpolation from 0 to 1 based on position
                    if x_max > x_min:
                        dirichlet_values[idx] = (x - x_min) / (x_max - x_min)
                    else:
                        dirichlet_values[idx] = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.0
            
            elif inhomogeneous_type == "quadratic":
                # Quadratic variation
                for idx in dirichlet_indices:
                    x, y = pos_np[idx]
                    dirichlet_values[idx] = x**2 + y**2
            
            elif inhomogeneous_type == "gaussian":
                # Gaussian profile centered at domain center
                x_center = np.mean(pos_np[:, 0])
                y_center = np.mean(pos_np[:, 1])
                domain_size = max(pos_np[:, 0].max() - pos_np[:, 0].min(),
                                 pos_np[:, 1].max() - pos_np[:, 1].min())
                sigma = domain_size * 0.2  # 20% of domain size
                
                for idx in dirichlet_indices:
                    x, y = pos_np[idx]
                    r2 = (x - x_center)**2 + (y - y_center)**2
                    dirichlet_values[idx] = np.exp(-r2 / (2 * sigma**2))
            
            else:
                raise ValueError(f"Unknown inhomogeneous_type: {inhomogeneous_type}. "
                               f"Available options: 'constant', 'homogeneous', 'sinusoidal', 'linear', 'quadratic', 'gaussian'")
    
    return dirichlet_values

