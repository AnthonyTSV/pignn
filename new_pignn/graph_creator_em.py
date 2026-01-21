import torch
import numpy as np
import ngsolve as ng
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Dict, Tuple
from torch_geometric.data import Data
from ngsolve import Mesh


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
    raw_ids: list,  # numbers or objects with .nr
    pnum_to_idx: dict[int, int],  # mapping from Netgen point number -> contiguous idx
    N: int,  # number of points
    *,
    context: str = "unknown",
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
    seen_codes = sorted(
        {
            int(getattr(seg, "bc", getattr(seg, "si", getattr(seg, "index", 0))))
            for seg in ngmesh.Elements1D()
        }
    )

    if callable(get_name) and seen_codes:
        # Try to determine whether GetBCName is 0-based or 1-based
        def try_name(idx: int) -> str | None:
            try:
                name = get_name(idx)
                return None if name is None else str(name)
            except Exception:
                return None

        # Build mapping, but guard against out-of-range
        for c in seen_codes:
            name = try_name(c - 1)
            if not name:
                # As a last resort, give it a synthetic name
                name = f"bc_{c}"
            mapping[c] = name
        return mapping

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


def _classify_node_types(
    mesh: Mesh,
    dirichlet_names: List[str],
    neumann_names: List[str],
    robin_names: List[str],
    pnum_to_idx: Dict[int, int],
) -> torch.Tensor:
    """
    Classify nodes as Dirichlet (0), Neumann (1), Interior (2), or Robin (3).
    Dirichlet conditions take priority over Neumann/Robin at corner nodes.
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
    robin_bcnr = _resolve_boundary_names_to_bcnr(mesh, robin_names)

    # First pass: Mark Neumann nodes
    for seg in segments_1d:
        bc_code = int(getattr(seg, "bc", getattr(seg, "si", getattr(seg, "index", 0))))
        if bc_code in neumann_bcnr:
            vertices = seg.vertices
            vertex_indices = _normalize_ids_to_idx(
                vertices, pnum_to_idx, n_nodes, context="boundary_segments"
            )
            node_types[vertex_indices] = 1  # Neumann

    # Second pass: Mark Robin nodes (overrides Neumann if overlap, though usually distinct)
    for seg in segments_1d:
        bc_code = int(getattr(seg, "bc", getattr(seg, "si", getattr(seg, "index", 0))))
        if bc_code in robin_bcnr:
            vertices = seg.vertices
            vertex_indices = _normalize_ids_to_idx(
                vertices, pnum_to_idx, n_nodes, context="boundary_segments"
            )
            node_types[vertex_indices] = 3  # Robin

    # Third pass: Mark Dirichlet nodes (takes priority over everything)
    for seg in segments_1d:
        bc_code = int(getattr(seg, "bc", getattr(seg, "si", getattr(seg, "index", 0))))
        if bc_code in dirichlet_bcnr:
            vertices = seg.vertices
            vertex_indices = _normalize_ids_to_idx(
                vertices, pnum_to_idx, n_nodes, context="boundary_segments"
            )
            node_types[vertex_indices] = 0  # Dirichlet (takes priority)

    return node_types


def _build_fem_connectivity(
    mesh: Mesh,
    pnum_to_idx: Dict[int, int],
    add_self_loops: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build graph connectivity from FEM mesh topology (directed edges + self-loops).
    """
    edges_set = set()
    n_nodes = len(pnum_to_idx)

    # Add edges from 2D elements
    for elem in mesh.ngmesh.Elements2D():
        vertices = elem.vertices
        vertex_indices = _normalize_ids_to_idx(
            vertices, pnum_to_idx, n_nodes, context="2D_elements"
        )

        # Add edges between all pairs of vertices in element (fully connected)
        for i, vi in enumerate(vertex_indices):
            for j, vj in enumerate(vertex_indices):
                if i != j:  # No self-loops from element connectivity
                    edges_set.add((vi, vj))

    # Add edges from 1D boundary elements
    for elem in mesh.ngmesh.Elements1D():
        vertices = elem.vertices
        vertex_indices = _normalize_ids_to_idx(
            vertices, pnum_to_idx, n_nodes, context="1D_elements"
        )

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
        device = torch.device("cpu")

    if edges_set:
        edges = list(edges_set)
        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return edge_index


def _build_knn_connectivity(
    pos: torch.Tensor, k: int, add_self_loops: bool = True, device: torch.device = None
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
        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return edge_index


def _build_node_features(
    node_types: torch.Tensor,
    A_current: Optional[np.ndarray],
    current_density: Optional[np.ndarray],
    material_field: Optional[np.ndarray],
    dirichlet_values: Optional[np.ndarray],
    omega: float,
    n_nodes: int,
    device: torch.device = None,
) -> torch.Tensor:
    if device is None:
        device = node_types.device

    features = []

    # One-hot encoding of node types (Dirichlet=0, Neumann=1, Interior=2, Robin=3)
    # We use 4 classes now
    node_type_onehot = torch.zeros(n_nodes, 4, device=device)
    node_type_onehot[torch.arange(n_nodes, device=device), node_types] = 1.0
    features.append(node_type_onehot)

    # Current temperature field
    if A_current is not None:
        T_tensor = torch.tensor(
            A_current, dtype=torch.float32, device=device
        ).unsqueeze(1)
    else:
        T_tensor = torch.zeros(n_nodes, 1, device=device)
    features.append(T_tensor)

    # Omega scalar (broadcasted to all nodes)
    omega_tensor = torch.full((n_nodes, 1), omega, dtype=torch.float32, device=device)
    features.append(omega_tensor)

    # Material field (mu_r)
    # Use log(mu_r) as feature for better scaling when mu_r varies widely (e.g., 1 to 100)
    # This helps the network learn because:
    # 1. log(1) = 0, log(100) ≈ 4.6 is a much smaller range than 1 to 100
    # 2. The PDE uses nu = 1/(mu0*mu_r), and log transforms multiplicative relationships to additive
    if material_field is not None:
        # Use log(mu_r) for better scaling
        log_mu_r = np.log(material_field + 1e-10)  # Add small epsilon to avoid log(0)
        mat_tensor = torch.tensor(
            log_mu_r, dtype=torch.float32, device=device
        ).unsqueeze(1)
    else:
        mat_tensor = torch.zeros(n_nodes, 1, device=device)  # log(1) = 0 for default material
    features.append(mat_tensor)

    # Dirichlet boundary values (prescribed values for Dirichlet BC)
    if dirichlet_values is not None:
        dirichlet_tensor = torch.tensor(
            dirichlet_values, dtype=torch.float32, device=device
        ).unsqueeze(1)
    else:
        dirichlet_tensor = torch.zeros(
            n_nodes, 1, device=device
        )  # Default: homogeneous Dirichlet
    features.append(dirichlet_tensor)

    # Source term values (current density)
    # Normalize current density to be in similar range as other features
    # Use log(1 + |J|) to handle the large values and map to ~[0, 5] range
    if current_density is not None:
        if isinstance(current_density, torch.Tensor):
            j_vals = (
                current_density.detach()
                .clone()
                .to(dtype=torch.float32, device=device)
            )
        else:
            j_vals = torch.tensor(
                current_density, dtype=torch.float32, device=device
            )
        # Normalize: log(1 + |J|) to compress the range
        # Also add a binary indicator for coil region (where J != 0)
        j_indicator = (torch.abs(j_vals) > 1e-10).float().unsqueeze(1)  # Binary: is coil?
        j_normalized = torch.log1p(torch.abs(j_vals)).unsqueeze(1)  # log(1 + |J|)
        source_tensor = torch.cat([j_indicator, j_normalized], dim=1)
    else:
        source_tensor = torch.zeros(n_nodes, 2, device=device)  # Default: no source (indicator=0, log_j=0)
    features.append(source_tensor)

    return torch.cat(features, dim=1)


def _build_edge_features(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    A_current: Optional[np.ndarray],
    device: torch.device = None,
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
    if A_current is not None:
        T_tensor = torch.tensor(A_current, dtype=torch.float32, device=device)
        src_temp = T_tensor[edge_index[0]]
        dst_temp = T_tensor[edge_index[1]]
        temp_diff = (dst_temp - src_temp).unsqueeze(1)
    else:
        temp_diff = torch.zeros(edge_index.shape[1], 1, device=device)

    # Concatenate all edge features
    edge_features = torch.cat([relative_pos, euclidean_dist, temp_diff], dim=1)

    return edge_features


def _build_global_features(
    omega: float, n_nodes: int, device: torch.device = None
) -> torch.Tensor:
    """
    Build global feature vector.
    """
    if device is None:
        device = torch.device("cpu")
    global_features = torch.tensor(
        [omega, float(n_nodes)], dtype=torch.float32, device=device
    )
    return global_features


def _get_nodes_on_boundary(
    mesh: Mesh, boundary_name: str, pnum_to_idx: Dict[int, int]
) -> List[int]:
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
            vertex_indices = _normalize_ids_to_idx(
                vertices, pnum_to_idx, n_nodes, context="boundary_segments"
            )
            boundary_nodes.update(vertex_indices)

    return list(boundary_nodes)


class GraphCreatorEM:
    def __init__(
        self,
        mesh: ng.Mesh,
        n_neighbors: int = 8,
        dirichlet_names: List[str] = None
    ):
        self.mesh = mesh
        self.n_neighbors = n_neighbors
        self.dirichlet_names = dirichlet_names or []
        self.connectivity_method = "fem"

    def create_graph(
        self,
        A_current: Optional[np.ndarray] = None,
        current_density: Optional[np.ndarray] = None,
        material_node_field: Optional[np.ndarray] = None,
        dirichlet_values: Optional[np.ndarray] = None,
        omega: float = 0.0,
        add_self_loops: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[Data, Dict]:
        """
        Create PI-MGN graph from mesh
        """
        if device is None:
            device = torch.device("cpu")

        # 1. Build node positions and index mapping
        pos, pnum_to_idx = _build_point_index_map(self.mesh)
        n_nodes = len(pos)

        # Convert to torch tensor on target device
        pos_tensor = torch.tensor(pos, dtype=torch.float32, device=device)

        # 2. Determine node types (Dirichlet/Neumann/Robin/Interior)
        node_types = _classify_node_types(
            self.mesh,
            self.dirichlet_names or [],
            [],
            [],
            pnum_to_idx,
        )
        node_types = node_types.to(device)

        # 3. Build connectivity based on method
        if self.connectivity_method.lower() == "fem":
            edge_index = _build_fem_connectivity(
                self.mesh, pnum_to_idx, add_self_loops, device
            )
        elif self.connectivity_method.lower() == "knn":
            edge_index = _build_knn_connectivity(
                pos_tensor, self.n_neighbors, add_self_loops, device
            )
        else:
            raise ValueError(f"Unknown connectivity method: {self.connectivity_method}")

        # 4. Build node features
        node_features = _build_node_features(
            node_types,
            A_current,
            current_density,
            material_node_field,
            dirichlet_values,
            omega,
            n_nodes,
            device,
        )

        # 5. Build edge features
        edge_features = _build_edge_features(edge_index, pos_tensor, A_current, device)

        # 6. Build global features
        global_features = _build_global_features(omega, n_nodes, device)

        # 7. Create auxiliary data
        aux = {
            "pnum_to_idx": pnum_to_idx,
            "node_types": node_types,
            "dirichlet_mask": node_types == 0,
            "interior_mask": node_types == 2,
            "robin_mask": node_types == 3,
            "free_mask": node_types != 0,  # Non-Dirichlet nodes
            "dirichlet_values": dirichlet_values,
            "current_density": current_density,
            "mesh": self.mesh,
            "connectivity_method": self.connectivity_method,
        }

        # 8. Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            pos=pos_tensor,
            global_attr=global_features,
            num_nodes=n_nodes,
        )

        return data, aux

    def create_free_node_subgraph(
        self, data: Data, aux: Dict
    ) -> Tuple[Data, torch.Tensor, Dict]:
        """
        Create a subgraph containing only free (non-Dirichlet) nodes.

        Args:
            data: Full PyTorch Geometric Data object
            aux: Auxiliary data dictionary from create_graph

        Returns:
            free_graph: Subgraph with only free nodes
            node_mapping: Tensor mapping free node indices to original graph indices
            new_aux: Updated auxiliary data for the subgraph
        """
        free_mask = aux["free_mask"]

        # Ensure all tensors are on the same device as the data
        device = (
            data.x.device
            if hasattr(data.x, "device")
            else torch.device("gpu" if torch.cuda.is_available() else "cpu")
        )
        free_mask = free_mask.to(device)

        # Move all aux tensors to the same device
        aux_tensors_to_move = [
            "node_types",
            "dirichlet_mask",
            "neumann_mask",
            "interior_mask",
            "robin_mask",
        ]
        for key in aux_tensors_to_move:
            if key in aux and hasattr(aux[key], "to"):
                aux[key] = aux[key].to(device)

        free_indices = torch.where(free_mask)[0]
        n_free = len(free_indices)

        if n_free == 0:
            raise ValueError(
                "No free nodes found - all nodes are on Dirichlet boundary"
            )

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

        # Handle Robin values for subgraph
        robin_values_sub = None
        if aux.get("robin_values") is not None:
            h_vals, amb_vals = aux["robin_values"]
            robin_values_sub = (
                h_vals[free_indices.cpu().numpy()],
                amb_vals[free_indices.cpu().numpy()],
            )

        # Create subgraph data
        free_data = Data(
            x=free_x,
            edge_index=free_edge_index,
            edge_attr=free_edge_attr,
            pos=free_pos,
            global_attr=data.global_attr,
            num_nodes=n_free,
        )

        free_aux = {
            "pnum_to_idx": {
                k: v
                for k, v in aux["pnum_to_idx"].items()
                if v in free_indices.tolist()
            },
            "node_types": aux["node_types"][free_indices],
            "dirichlet_mask": aux["dirichlet_mask"][free_indices],
            "interior_mask": aux["interior_mask"][free_indices],
            "free_mask": torch.ones(
                n_free, dtype=torch.bool, device=device
            ),  # All nodes in subgraph are free
            "dirichlet_values": (
                aux["dirichlet_values"][free_indices.cpu().numpy()]
                if aux["dirichlet_values"] is not None
                else None
            ),
            "source_values": (
                aux["source_values"][free_indices.cpu().numpy()]
                if aux.get("source_values") is not None
                else None
            ),
            "mesh": aux["mesh"],
            "connectivity_method": aux["connectivity_method"],
        }

        # Node mapping for reconstruction
        node_mapping = {
            "free_to_original": free_indices,
            "original_to_free": old_to_new,
            "n_original": data.num_nodes,
            "n_free": n_free,
        }

        return free_data, node_mapping, free_aux

    def create_neumann_values(
        self,
        pos,
        aux_data,
        neumann_names,
        flux_values=None,
        flux_magnitude=1.0,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        # Convert pos to numpy if it's a torch tensor
        if hasattr(pos, "numpy"):
            pos_np = pos.numpy()
        else:
            pos_np = np.array(pos)

        n_nodes = pos_np.shape[0]
        neumann_values = np.zeros(n_nodes)

        # Get mesh from aux data
        mesh = aux_data["mesh"]
        pnum_to_idx = aux_data["pnum_to_idx"]

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
            raise ValueError(
                "flux_values must be None, a number, or a dict mapping boundary names to values"
            )

        # Assign flux values based on boundary names
        for boundary_name in neumann_names:
            if boundary_name not in flux_values:
                raise ValueError(
                    f"No flux value specified for boundary '{boundary_name}'"
                )

            # Get nodes on this specific boundary
            boundary_nodes = _get_nodes_on_boundary(mesh, boundary_name, pnum_to_idx)
            flux_value = flux_values[boundary_name]

            for node_idx in boundary_nodes:
                neumann_values[node_idx] = flux_value

        return neumann_values

    def create_dirichlet_values(
        self,
        pos,
        aux_data,
        dirichlet_names,
        boundary_values=None,
        homogeneous_value=0.0,
        seed=None,
    ):
        """
        Create Dirichlet boundary values.

        For nodes that belong to multiple Dirichlet boundaries, higher values take priority
        over lower values. This ensures consistent boundary condition enforcement when
        boundaries intersect at corners or edges.
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert pos to numpy if it's a torch tensor
        if hasattr(pos, "numpy"):
            pos_np = pos.numpy()
        else:
            pos_np = np.array(pos)

        n_nodes = pos_np.shape[0]
        dirichlet_values = np.zeros(n_nodes)

        # Get mesh from aux data
        mesh = aux_data["mesh"]
        pnum_to_idx = aux_data["pnum_to_idx"]

        # Handle different input formats for dirichlet_names
        if isinstance(dirichlet_names, str):
            dirichlet_names = [dirichlet_names]

        # Handle different input formats for boundary_values
        if boundary_values is None:
            # Use default homogeneous_value for all boundaries
            boundary_values = {name: homogeneous_value for name in dirichlet_names}
        elif isinstance(boundary_values, (int, float)):
            # Single constant value for all boundaries
            boundary_values = {name: boundary_values for name in dirichlet_names}
        elif not isinstance(boundary_values, dict):
            raise ValueError(
                "boundary_values must be None, a number, or a dict mapping boundary names to values"
            )

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

        return dirichlet_values

    def create_robin_values(
        self,
        pos,
        aux_data,
        robin_names,
        robin_values=None,
        h_default=1.0,
        amb_default=0.0,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        # Convert pos to numpy if it's a torch tensor
        if hasattr(pos, "numpy"):
            pos_np = pos.numpy()
        else:
            pos_np = np.array(pos)

        n_nodes = pos_np.shape[0]
        h_values = np.zeros(n_nodes)
        amb_values = np.zeros(n_nodes)

        # Get mesh from aux data
        mesh = aux_data["mesh"]
        pnum_to_idx = aux_data["pnum_to_idx"]

        # Handle different input formats for robin_names
        if isinstance(robin_names, str):
            robin_names = [robin_names]

        # Handle different input formats for robin_values
        if robin_values is None:
            # Use default values for all boundaries
            robin_values = {name: (h_default, amb_default) for name in robin_names}
        elif isinstance(robin_values, (tuple, list)) and len(robin_values) == 2:
            # Single tuple for all boundaries
            robin_values = {name: robin_values for name in robin_names}
        elif not isinstance(robin_values, dict):
            raise ValueError(
                "robin_values must be None, a tuple (h, T_amb), or a dict mapping boundary names to tuples"
            )

        # Assign values based on boundary names
        for boundary_name in robin_names:
            if boundary_name not in robin_values:
                raise ValueError(
                    f"No Robin values specified for boundary '{boundary_name}'"
                )

            # Get nodes on this specific boundary
            boundary_nodes = _get_nodes_on_boundary(mesh, boundary_name, pnum_to_idx)
            h_val, amb_val = robin_values[boundary_name]

            for node_idx in boundary_nodes:
                h_values[node_idx] = h_val
                amb_values[node_idx] = amb_val

        return h_values, amb_values

    def visualize_graph(
        self,
        data: Data,
        aux: Dict,
        figsize: Tuple[int, int] = (24, 6),
        node_size: int = 50,
        edge_alpha: float = 0.6,
        save_path: Optional[str] = None,
        only_free_nodes: bool = False,
    ):
        """
        Visualize the graph using NetworkX and matplotlib.
        """
        # Convert to NetworkX graph
        G = self._pyg_to_networkx(data)

        # Create figure with subplots - add fourth subplot for Dirichlet values
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)

        # Get positions from data
        pos_dict = {i: data.pos[i].numpy() for i in range(data.num_nodes)}

        # Color nodes by type - handle both full graph and subgraph cases
        if only_free_nodes:
            # For free node subgraphs, all nodes are interior (green)
            node_colors = ["green"] * data.num_nodes
            node_type_counts = {
                0: 0,
                1: 0,
                2: data.num_nodes,
                3: 0,
            }  # No Dirichlet/Neumann/Robin in free nodes
        else:
            # For full graphs, use the node types from aux
            node_types = aux["node_types"].numpy()
            # Ensure we only process as many node types as we have nodes in the graph
            node_types = node_types[: data.num_nodes]
            node_colors = []
            node_type_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            for node_type in node_types:
                if node_type == 0:  # Dirichlet
                    node_colors.append("red")
                    node_type_counts[0] += 1
                elif node_type == 1:  # Neumann
                    node_colors.append("blue")
                    node_type_counts[1] += 1
                elif node_type == 3:  # Robin
                    node_colors.append("orange")
                    node_type_counts[3] += 1
                else:  # Interior
                    node_colors.append("green")
                    node_type_counts[2] += 1

        # Plot 1: Graph structure with node types
        nx.draw_networkx_nodes(
            G, pos_dict, node_color=node_colors, node_size=node_size, ax=ax1, alpha=0.8
        )
        nx.draw_networkx_edges(
            G, pos_dict, alpha=edge_alpha, ax=ax1, edge_color="gray", width=0.5
        )

        graph_type = (
            "Free Nodes"
            if only_free_nodes
            else f"{self.connectivity_method.upper()} Graph"
        )
        ax1.set_title(
            f"{graph_type} Structure\\n"
            f"Nodes: {data.num_nodes}, Edges: {data.num_edges}"
        )
        ax1.set_aspect("equal")
        ax1.axis("off")

        # Add legend for node types
        legend_elements = [
            mpatches.Patch(color="red", label=f"Dirichlet ({node_type_counts[0]})"),
            mpatches.Patch(color="blue", label=f"Neumann ({node_type_counts[1]})"),
            mpatches.Patch(color="orange", label=f"Robin ({node_type_counts[3]})"),
            mpatches.Patch(color="green", label=f"Interior ({node_type_counts[2]})"),
        ]
        ax1.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

        # Plot 2: Temperature field visualization (if available)
        if data.x.shape[1] > 3:  # Has temperature data
            temp_values = data.x[:, 3].numpy()  # Temperature is 4th feature

            # Create scatter plot colored by temperature
            scatter = ax2.scatter(
                data.pos[:, 0].numpy(),
                data.pos[:, 1].numpy(),
                c=temp_values,
                cmap="coolwarm",
                s=node_size,
                alpha=0.8,
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label("Temperature")

            ax2.set_title(
                f"Temperature Field\\nRange: [{temp_values.min():.3f}, {temp_values.max():.3f}]"
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "No temperature data\\navailable",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_title("Temperature Field (N/A)")

        ax2.set_aspect("equal")
        ax2.axis("off")

        # Plot 3: Neumann values visualization (if available)
        if data.x.shape[1] > 6:  # Has Neumann data (7th feature)
            neumann_values = data.x[
                :, 6
            ].numpy()  # Neumann values are 7th feature (index 6)
            neumann_mask = aux["neumann_mask"]

            # Only show non-zero Neumann values or Neumann boundary nodes
            if neumann_mask.any() or (neumann_values != 0).any():
                # Create scatter plot colored by Neumann values
                scatter = ax3.scatter(
                    data.pos[:, 0].numpy(),
                    data.pos[:, 1].numpy(),
                    c=neumann_values,
                    cmap="viridis",
                    s=node_size,
                    alpha=0.8,
                )

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label("Neumann Values, ($h_N$)")

                # Highlight Neumann boundary nodes with red circles
                if neumann_mask.any():
                    neumann_indices = torch.where(neumann_mask)[0]
                    neumann_pos = data.pos[neumann_indices].numpy()
                    ax3.scatter(
                        neumann_pos[:, 0],
                        neumann_pos[:, 1],
                        s=node_size * 2,
                        facecolors="none",
                        edgecolors="red",
                        linewidth=2,
                    )

                ax3.set_title(
                    f"Neumann Values\\nRange: [{neumann_values.min():.3f}, {neumann_values.max():.3f}]"
                )
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No Neumann values\\navailable",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=12,
                )
                ax3.set_title("Neumann Values (N/A)")
        else:
            ax3.text(
                0.5,
                0.5,
                "No Neumann data\\navailable",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title("Neumann Values (N/A)")

        ax3.set_aspect("equal")
        ax3.axis("off")

        # Plot 4: Dirichlet values visualization (if available)
        if data.x.shape[1] > 7:  # Has Dirichlet data (8th feature)
            dirichlet_values = data.x[
                :, 7
            ].numpy()  # Dirichlet values are 8th feature (index 7)
            dirichlet_mask = aux["dirichlet_mask"]

            # Only show non-zero Dirichlet values or Dirichlet boundary nodes
            if dirichlet_mask.any() or (dirichlet_values != 0).any():
                # Create scatter plot colored by Dirichlet values
                scatter = ax4.scatter(
                    data.pos[:, 0].numpy(),
                    data.pos[:, 1].numpy(),
                    c=dirichlet_values,
                    cmap="plasma",
                    s=node_size,
                    alpha=0.8,
                )

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label("Dirichlet Values")

                # Highlight Dirichlet boundary nodes with blue circles
                if dirichlet_mask.any():
                    dirichlet_indices = torch.where(dirichlet_mask)[0]
                    dirichlet_pos = data.pos[dirichlet_indices].numpy()
                    ax4.scatter(
                        dirichlet_pos[:, 0],
                        dirichlet_pos[:, 1],
                        s=node_size * 2,
                        facecolors="none",
                        edgecolors="blue",
                        linewidth=2,
                    )

                ax4.set_title(
                    f"Dirichlet Values\\nRange: [{dirichlet_values.min():.3f}, {dirichlet_values.max():.3f}]"
                )
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No Dirichlet values\\navailable",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                    fontsize=12,
                )
                ax4.set_title("Dirichlet Values (N/A)")
        else:
            ax4.text(
                0.5,
                0.5,
                "No Dirichlet data\\navailable",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
            ax4.set_title("Dirichlet Values (N/A)")

        ax4.set_aspect("equal")
        ax4.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graph visualization saved to: {save_path}")

        # plt.show()

    def _pyg_to_networkx(self, data: Data) -> nx.Graph:
        """
        Convert PyTorch Geometric Data to NetworkX graph.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            G: NetworkX graph
        """
        G = nx.Graph()

        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i)

        # Add edges (convert to undirected by removing duplicates)
        edge_list = data.edge_index.t().numpy()
        for src, dst in edge_list:
            if src != dst:  # Skip self-loops for cleaner visualization
                G.add_edge(src, dst)

        return G
