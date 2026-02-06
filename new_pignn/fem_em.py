import numpy as np
from containers import MeshConfig, MeshProblem, MeshProblemEM
import ngsolve as ng
from typing import Optional, List, Tuple, Union, Literal, Callable
import torch
import scipy.sparse as sp
import os
from pathlib import Path

TensorLike = Union[np.ndarray, torch.Tensor]
NormalizeMode = Literal["none", "ndof", "rhs"]
EnergyInvMode = Literal["jacobi", "cg"]

r_star = 70 * 1e-3  # m
A_star = 4.8 * 1e-4  # Wb/m
mu_star = 4 * 3.1415926535e-7  # H/m
J_star = A_star / (r_star**2 * mu_star)
frequency = 1000  # Hz
omega = 2 * ng.pi * frequency  # rad/s
sigma_star = J_star / (omega * A_star)


def _sparse_diag(A: torch.Tensor) -> torch.Tensor:
    """
    Extract diagonal of a sparse COO/CSR tensor as a dense vector.
    Works reliably for COO. For CSR, coalesce may convert internally.
    """
    if not A.is_sparse:
        return torch.diagonal(A)

    A = A.coalesce()
    idx = A.indices()  # [2, nnz]
    val = A.values()
    mask = idx[0] == idx[1]
    diag_idx = idx[0, mask]
    diag_val = val[mask]

    n = A.shape[0]
    diag = torch.zeros(n, device=val.device, dtype=val.dtype)
    diag.scatter_add_(0, diag_idx, diag_val)
    return diag


def _cg_solve(
    A_mv: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    M_inv: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> torch.Tensor:
    """
    Conjugate Gradient solve for SPD operator: A x = b.

    Note: This is differentiable (uses torch ops), but the loop builds a graph.
    Keep max_iter modest if you backprop through it.
    """
    x = torch.zeros_like(b) if x0 is None else x0
    r = b - A_mv(x)

    if M_inv is None:
        z = r
    else:
        z = M_inv(r)

    p = z
    rz_old = torch.dot(r, z)

    b_norm = torch.linalg.norm(b).clamp_min(1e-30)

    for _ in range(max_iter):
        Ap = A_mv(p)
        alpha = rz_old / torch.dot(p, Ap).clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap

        if torch.linalg.norm(r) / b_norm < tol:
            break

        z = r if M_inv is None else M_inv(r)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz_old.clamp_min(1e-30)
        p = z + beta * p
        rz_old = rz_new

    return x


class FEMSolverEM:
    def __init__(
        self,
        mesh: ng.Mesh,
        order=1,
        problem: Optional[MeshProblemEM] = None,
        device: Optional[torch.device] = None,
    ):
        self.mesh = mesh
        self.order = order
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.bilinear_form = None
        self.linear_form = None
        self.fes = None

        self.problem = problem
        # if problem is not None:
        #     if problem.complex:
        #         self.init_complex_matrices()
        #     else:
        #         self.init_matrices()
        self.init_mixed_matrices()

    def init_matrices(self):
        if self.problem is None:
            raise ValueError("Problem must be set before initializing matrices")
        # Define material properties as coefficient functions
        mu_r = self.mesh.MaterialCF(
            {
                "mat_workpiece": self.problem.mu_r_workpiece,
                "mat_air": self.problem.mu_r_air,
                "mat_coil": self.problem.mu_r_coil,
            },
            default=1.0,
        )

        sigma = self.mesh.MaterialCF(
            {
                "mat_workpiece": self.problem.sigma_workpiece,
                "mat_air": self.problem.sigma_air,
                "mat_coil": self.problem.sigma_coil,
            },
            default=0.0,
        )
        fes = ng.H1(
            self.mesh,
            order=self.order,
            complex=False,
            dirichlet=self.problem.mesh_config.dirichlet_pipe,
        )
        A, v = fes.TnT()
        gfA = ng.GridFunction(fes)

        r = ng.x
        dr_rA = r * ng.grad(A)[0] + A
        dr_rv = r * ng.grad(v)[0] + v
        dzA, dzv = ng.grad(A)[1], ng.grad(v)[1]

        # Avoid 1/r singularity on the symmetry axis (r=0)
        inv_r = ng.IfPos(r, 1.0 / r, 0.0)

        nu = 1.0 / (self.problem.mu0 * mu_r)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + inv_r * dr_rA * dr_rv) * ng.dx
        # a += 1j * self.problem.omega * sigma * r * A * v * ng.dx
        Acoil = self.problem.profile_width_phys * self.problem.profile_height_phys
        Js_phi = self.problem.N_turns * self.problem.I_coil / Acoil
        Js_phi = Js_phi / J_star  # Normalize current density
        print("Normalized current density Js_phi:", Js_phi)
        f = ng.LinearForm(fes)
        f += r * Js_phi * v * ng.dx("mat_coil")

        a.Assemble()
        f.Assemble()

        self.bilinear_form = self._ngsolve_to_torch(a.mat)
        self.linear_form = torch.tensor(
            f.vec.FV().NumPy().copy(), dtype=torch.float64
        ).to(self.device)
        self.fes = fes

    def init_complex_matrices(self):
        if self.problem is None:
            raise ValueError("Problem must be set before initializing matrices")
        # Define material properties as coefficient functions
        mu_r = self.mesh.MaterialCF(
            {
                "mat_workpiece": self.problem.mu_r_workpiece,
                "mat_air": self.problem.mu_r_air,
                "mat_coil": self.problem.mu_r_coil,
            },
            default=1.0,
        )

        sigma = self.mesh.MaterialCF(
            {
                "mat_workpiece": self.problem.sigma_workpiece,
                "mat_air": self.problem.sigma_air,
                "mat_coil": self.problem.sigma_coil,
            },
            default=0.0,
        )
        fes = ng.H1(
            self.mesh,
            order=self.order,
            complex=True,
            dirichlet=self.problem.mesh_config.dirichlet_pipe,
        )
        A, v = fes.TnT()
        gfA = ng.GridFunction(fes)

        r = ng.x
        dr_rA = r * ng.grad(A)[0] + A
        dr_rv = r * ng.grad(v)[0] + v
        dzA, dzv = ng.grad(A)[1], ng.grad(v)[1]

        # Avoid 1/r singularity on the symmetry axis (r=0)
        inv_r = ng.IfPos(r, 1.0 / r, 0.0)

        nu = 1.0 / (self.problem.mu0 * mu_r)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + inv_r * dr_rA * dr_rv) * ng.dx
        # Eddy current term: applied to all conductive regions (including workpiece)
        # In induction heating, eddy currents are induced in conductive materials
        a += 1j * sigma * r * A * v * ng.dx
        Acoil = self.problem.profile_width_phys * self.problem.profile_height_phys
        Js_phi = self.problem.N_turns * self.problem.I_coil / Acoil
        Js_phi = Js_phi / J_star  # Normalize current density
        print("Normalized current density Js_phi:", Js_phi)
        f = ng.LinearForm(fes)
        f += r * Js_phi * v * ng.dx("mat_coil")

        a.Assemble()
        f.Assemble()

        k_real, k_imag = self._ngsolve_to_torch(a.mat)

        self.bilinear_form = (k_real, k_imag)
        self.linear_form = torch.tensor(
            f.vec.FV().NumPy().copy(), dtype=torch.float64
        ).to(self.device)
        self.fes = fes

    def init_mixed_matrices(self):
        """Initialize matrices for A-φ mixed formulation."""
        # FE spaces
        fes_a = ng.H1(
            self.mesh,
            order=self.order,
            complex=True,
            dirichlet=self.problem.mesh_config.dirichlet_pipe,
        )
        fes_phi = ng.H1(
            self.mesh,
            order=self.order,
            complex=True,
            definedon=self.mesh.Materials("mat_coil"),
        )

        fes = ng.FESpace([fes_a, fes_phi])

        trials = fes.TrialFunction()
        tests = fes.TestFunction()
        A, phi_coil = trials[0], trials[1]
        v, psi = tests[0], tests[1]

        mu_r = self.mesh.MaterialCF(
            {
                "mat_workpiece": self.problem.mu_r_workpiece,
                "mat_air": self.problem.mu_r_air,
                "mat_coil": self.problem.mu_r_coil,
            },
            default=1.0,
        )

        sigma = self.mesh.MaterialCF(
            {
                "mat_workpiece": self.problem.sigma_workpiece,
                "mat_air": self.problem.sigma_air,
                "mat_coil": self.problem.sigma_coil,
            },
            default=0.0,
        )

        r = ng.x
        r1 = ng.IfPos(r, 1.0 / r, 0.0)

        dr_rA = r * ng.grad(A)[0] + A
        dr_rv = r * ng.grad(v)[0] + v
        dzA, dzv = ng.grad(A)[1], ng.grad(v)[1]

        nu = 1.0 / (self.problem.mu0 * mu_r)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + r1 * dr_rA * dr_rv) * ng.dx

        A_eff = A + phi_coil * r1
        v_eff = v + psi * r1
        a += 1j * sigma * r * A_eff * v_eff * ng.dx

        I_spec = self.problem.N_turns * self.problem.I_coil
        area_coil = self.problem.profile_width_phys * self.problem.profile_height_phys
        Js_phi = I_spec / area_coil
        Js_phi = Js_phi / J_star  # Normalize current density
        print("Normalized current density Js_phi:", Js_phi)

        f = ng.LinearForm(fes)
        f += (-Js_phi) * psi * ng.dx("mat_coil")

        a.Assemble()
        f.Assemble()

        # Store DOF info
        self.n_dofs_A = fes_a.ndof
        self.n_dofs_phi = fes_phi.ndof
        self.fes = fes
        self.fes_a = fes_a
        self.fes_phi = fes_phi

        # Build mapping from graph nodes to phi DOFs
        # phi is only defined on coil nodes, so we need to know which graph nodes
        # correspond to which phi DOFs
        self._build_node_to_phi_dof_mapping()

        k_real, k_imag = self._ngsolve_to_torch(a.mat)
        self.bilinear_form = (k_real, k_imag)
        self.linear_form = torch.tensor(
            f.vec.FV().NumPy().copy(), dtype=torch.float64
        ).to(self.device)

    def _build_node_to_phi_dof_mapping(self):
        """
        Build mapping from graph node indices to phi DOF indices.

        In NGSolve, for order=1 H1 space, DOFs correspond to mesh vertices.
        For the phi space (defined only on mat_coil), we need to identify
        which vertices are in the coil region and their corresponding DOF indices.
        """
        import ngsolve as ng
        
        # Get the number of mesh vertices (graph nodes)
        n_vertices = len(list(self.mesh.ngmesh.Points()))
        
        # Build mapping using NGSolve's DOF structure
        # For each vertex, check if it has a DOF in fes_phi
        node_to_phi_dof = np.full(n_vertices, -1, dtype=np.int64)
        coil_node_indices = []
        
        for v_idx in range(n_vertices):
            # Get DOF numbers for this vertex in the phi space
            # NodeId(VERTEX, v_idx) gives the vertex node
            try:
                dof_nrs = self.fes_phi.GetDofNrs(ng.NodeId(ng.VERTEX, v_idx))
                if len(dof_nrs) > 0 and dof_nrs[0] >= 0:
                    phi_dof = dof_nrs[0]
                    node_to_phi_dof[v_idx] = phi_dof
                    coil_node_indices.append((phi_dof, v_idx))  # (dof_idx, node_idx)
            except:
                # Vertex not in phi domain
                pass
        
        # Sort by DOF index to get correct ordering
        coil_node_indices.sort(key=lambda x: x[0])
        
        # coil_node_indices[i] = node_idx for phi DOF i
        self.coil_node_indices = np.array([node_idx for _, node_idx in coil_node_indices], dtype=np.int64)
        self.node_to_phi_dof = node_to_phi_dof
        self.coil_vertex_mask = node_to_phi_dof >= 0
        self.n_coil_nodes = len(self.coil_node_indices)
        
        # Verify the mapping matches fes_phi.ndof
        if self.n_coil_nodes != self.n_dofs_phi:
            print(f"Warning: n_coil_nodes ({self.n_coil_nodes}) != n_dofs_phi ({self.n_dofs_phi})")
            print(f"This may cause dimension mismatches in loss computation.")

    def _ngsolve_to_torch(self, ngsolve_matrix):
        """Convert NGSolve matrix to torch tensor."""

        def _coo_to_tensor(rows, cols, vals, shape, dtype):
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=dtype)
            tensor = torch.sparse_coo_tensor(indices, values, size=shape, dtype=dtype)
            return tensor.to(self.device)

        if ngsolve_matrix.is_complex:
            rows, cols, vals = ngsolve_matrix.COO()
            rows_r, cols_r, vals_r = rows, cols, np.real(vals)
            rows_i, cols_i, vals_i = rows, cols, np.imag(vals)

            shape = (ngsolve_matrix.height, ngsolve_matrix.width)
            tensor_r = _coo_to_tensor(rows_r, cols_r, vals_r, shape, torch.float64)
            tensor_i = _coo_to_tensor(rows_i, cols_i, vals_i, shape, torch.float64)

            return tensor_r, tensor_i
        else:
            rows, cols, vals = ngsolve_matrix.COO()
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float64)
            shape = (ngsolve_matrix.height, ngsolve_matrix.width)
            tensor = torch.sparse_coo_tensor(
                indices, values, size=shape, dtype=torch.float64
            )
            return tensor.to(self.device)

    def solve(self, problem):
        # Define material properties as coefficient functions
        mu_r = self.mesh.MaterialCF(
            {
                "mat_workpiece": self.problem.mu_r_workpiece,
                "mat_air": self.problem.mu_r_air,
                "mat_coil": self.problem.mu_r_coil,
            },
            default=1.0,
        )

        sigma = self.mesh.MaterialCF(
            {
                "mat_workpiece": self.problem.sigma_workpiece,
                "mat_air": self.problem.sigma_air,
                "mat_coil": self.problem.sigma_coil,
            },
            default=0.0,
        )
        fes = self.fes
        A, v = fes.TnT()
        gfA = ng.GridFunction(fes)

        r = ng.x
        dr_rA = r * ng.grad(A)[0] + A
        dr_rv = r * ng.grad(v)[0] + v
        dzA, dzv = ng.grad(A)[1], ng.grad(v)[1]

        # Avoid 1/r singularity on the symmetry axis (r=0)
        inv_r = ng.IfPos(r, 1.0 / r, 0.0)

        # In normalized form: nu = 1/(mu0_normalized * mu_r) = 1/(1 * mu_r) = 1/mu_r
        nu = 1.0 / (self.problem.mu0 * mu_r)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + inv_r * dr_rA * dr_rv) * ng.dx
        # Eddy current term: applied to all conductive regions (including workpiece)
        a += 1j * sigma * r * A * v * ng.dx
        Acoil = self.problem.profile_width_phys * self.problem.profile_height_phys
        Js_phi = self.problem.N_turns * self.problem.I_coil / Acoil
        Js_phi = Js_phi / J_star  # Normalize current density
        print("Normalized current density Js_phi:", Js_phi)
        f = ng.LinearForm(fes)
        f += r * Js_phi * v * ng.dx("mat_coil")

        a.Assemble()
        f.Assemble()
        gfA.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

        # def curl(u):
        #     gradu = ng.grad(u)
        #     return ng.CF((-gradu[1], gradu[0] + ng.IfPos(ng.x, u / ng.x, gradu[0])))

        # gfa_curl = curl(gfA)
        # gfa_norm = ng.Norm(gfa_curl)
        return gfA.vec.FV().NumPy().copy()

    def solve_mixed_em(self, problem: MeshProblemEM):
        mesh = problem.mesh
        # A_phi space
        mu_r = mesh.MaterialCF(
            {
                "mat_workpiece": problem.mu_r_workpiece,
                "mat_air": problem.mu_r_air,
                "mat_coil": problem.mu_r_coil,
            },
            default=1.0,
        )

        sigma = mesh.MaterialCF(
            {
                "mat_workpiece": problem.sigma_workpiece,
                "mat_air": problem.sigma_air,
                "mat_coil": problem.sigma_coil,
            },
            default=0.0,
        )

        fes_a = ng.H1(
            mesh,
            order=1,
            complex=True,
            dirichlet="bc_air|bc_axis|bc_workpiece_left",
        )

        fes_phi = ng.H1(
            mesh,
            order=1,
            complex=True,
            definedon=mesh.Materials("mat_coil"),
        )

        fes = ng.FESpace([fes_a, fes_phi])

        trials = fes.TrialFunction()
        tests = fes.TestFunction()

        A = trials[0]
        phi_coil = trials[1]
        v = tests[0]
        psi = tests[1]

        gfu = ng.GridFunction(fes)
        gfA = gfu.components[0]
        gfPhi = gfu.components[1]

        r = ng.x
        r1 = ng.IfPos(r, 1.0 / r, 0.0)

        dr_rA = r * ng.grad(A)[0] + A
        dr_rv = r * ng.grad(v)[0] + v
        dzA = ng.grad(A)[1]
        dzv = ng.grad(v)[1]

        nu = 1.0 / (problem.mu0 * mu_r)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + r1 * dr_rA * dr_rv) * ng.dx

        A_eff = A + phi_coil * r1
        v_eff = v + psi * r1
        a += 1j * sigma * r * A_eff * v_eff * ng.dx

        I_spec = problem.N_turns * problem.I_coil
        area_coil = problem.profile_width_phys * problem.profile_height_phys
        Js_phi = I_spec / area_coil
        Js_phi = Js_phi / J_star  # Normalize current density

        f = ng.LinearForm(fes)
        f += (-Js_phi) * psi * ng.dx("mat_coil")

        a.Assemble()
        f.Assemble()

        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="pardiso") * f.vec

        return gfA, gfPhi, r1

    def compute_residual(
        self,
        pred_sol,
    ):
        """
        Compute the residual of the FEM solution at the current time step.

        Args:
            pred_sol: Can be either a numpy array or a torch tensor
        """
        if self.bilinear_form is None or self.linear_form is None:
            raise ValueError("Bilinear and linear forms must be initialized")

        # Handle both numpy arrays and torch tensors
        if isinstance(pred_sol, np.ndarray):
            pred_sol_tensor = torch.tensor(pred_sol, dtype=torch.float64)
        else:
            # It's already a tensor - convert dtype while preserving gradients
            pred_sol_tensor = pred_sol.to(dtype=torch.float64)

        pred_sol_tensor = pred_sol_tensor.to(self.device)

        free_dofs_bitarray = self.fes.FreeDofs()

        # Convert BitArray to torch boolean mask
        free_dofs_mask = torch.tensor(
            [free_dofs_bitarray[i] for i in range(len(free_dofs_bitarray))],
            dtype=torch.bool,
            device=self.device,
        )

        # enforce Dirichlet boundary conditions
        boundary_dofs_vector = self._create_boundary_dofs_vector(
            problem=self.problem,
            device=self.device,
            dtype=torch.float64,
        )
        pred_full = pred_sol_tensor.clone()
        pred_full[~free_dofs_mask] = boundary_dofs_vector[~free_dofs_mask]

        Ax = torch.sparse.mm(self.bilinear_form, pred_full.unsqueeze(1)).squeeze(1)
        res = Ax - self.linear_form

        res = res[free_dofs_mask]

        return res

    def _to_tensor64(self, x: TensorLike) -> torch.Tensor:
        """Convert numpy/torch input to float64 torch tensor on the correct device."""
        if isinstance(x, np.ndarray):
            t = torch.tensor(x, dtype=torch.float64, device=self.device)
        else:
            t = x.to(device=self.device, dtype=torch.float64)
        return t

    def _free_dofs_mask(self) -> torch.Tensor:
        """Torch boolean mask for free DOFs."""
        free_dofs_bitarray = self.fes.FreeDofs()
        return torch.tensor(
            [free_dofs_bitarray[i] for i in range(len(free_dofs_bitarray))],
            dtype=torch.bool,
            device=self.device,
        )

    def _apply_dirichlet(
        self,
        pred_sol_tensor: torch.Tensor,
        free_dofs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build full DOF vector with Dirichlet values enforced."""
        boundary_dofs_vector = self._create_boundary_dofs_vector(
            problem=self.problem,
            device=self.device,
            dtype=torch.float64,
        )

        pred_full = pred_sol_tensor.clone()
        pred_full[~free_dofs_mask] = boundary_dofs_vector[~free_dofs_mask]
        return pred_full

    @torch.no_grad()
    def _rhs_norm(self, free_dofs_mask: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute stable norm of RHS on free DOFs (no gradients needed)."""
        rhs_free = self.linear_form[free_dofs_mask]
        return torch.linalg.norm(rhs_free) + eps

    def compute_energy_loss(
        self,
        pred_sol: TensorLike,
        *,
        normalize: NormalizeMode = "ndof",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Energy functional loss:
            Pi(a) = 0.5 * a^T K a - f^T a

        Minimizing Pi(a) yields the same stationary point as Ka = f
        (assuming K is SPD after Dirichlet constraints).

        Args:
            pred_sol: numpy array or torch tensor of DOFs (size ndof)
            normalize: "none" | "ndof" | "rhs"
            eps: small value for numerical stability
        Returns:
            Scalar torch tensor loss
        """
        if self.bilinear_form is None or self.linear_form is None:
            raise ValueError("Bilinear and linear forms must be initialized")

        pred_sol_tensor = self._to_tensor64(pred_sol)
        free_dofs_mask = self._free_dofs_mask()

        # Enforce Dirichlet BCs (hard)
        pred_full = self._apply_dirichlet(pred_sol_tensor, free_dofs_mask)

        # Ax = K a
        ax = torch.sparse.mm(self.bilinear_form, pred_full.unsqueeze(1)).squeeze(1)

        # Energy functional: 0.5 a^T (K a) - f^T a
        energy = 0.5 * torch.dot(pred_full, ax) - torch.dot(self.linear_form, pred_full)

        if normalize == "none":
            return energy
        if normalize == "ndof":
            ndof_free = free_dofs_mask.sum().clamp_min(1).to(dtype=torch.float64)
            return energy / ndof_free
        if normalize == "rhs":
            rhsn = self._rhs_norm(free_dofs_mask=free_dofs_mask, eps=eps)
            return energy / rhsn

        raise ValueError(f"Unknown normalize mode: {normalize}")

    def compute_complex_residual(
        self,
        pred_sol_real,
        pred_sol_imag,
        *,
        normalize: NormalizeMode = "ndof",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Compute the residual for complex-valued solutions.
        """
        free_mask = self._free_dofs_mask()

        pred_sol_real_tensor = self._to_tensor64(pred_sol_real)
        pred_sol_imag_tensor = self._to_tensor64(pred_sol_imag)

        Kr, Ki = self.bilinear_form  # Unpack real and imaginary parts

        Kr_ar = torch.sparse.mm(Kr, pred_sol_real_tensor.unsqueeze(1)).squeeze(1)
        Ki_ai = torch.sparse.mm(Ki, pred_sol_imag_tensor.unsqueeze(1)).squeeze(1)
        r_real = Kr_ar - Ki_ai - self.linear_form

        Kr_ai = torch.sparse.mm(Kr, pred_sol_imag_tensor.unsqueeze(1)).squeeze(1)
        Ki_ar = torch.sparse.mm(Ki, pred_sol_real_tensor.unsqueeze(1)).squeeze(1)
        r_imag = Kr_ai + Ki_ar

        r_real_free = r_real[free_mask]
        r_imag_free = r_imag[free_mask]

        loss = r_real_free.square().mean() + r_imag_free.square().mean()

        if normalize == "none":
            return loss
        if normalize == "rhs":
            rhsn = torch.linalg.norm(self.linear_form[free_mask]) + eps
            return loss / (rhsn * rhsn)
        if normalize == "ndof":
            return loss  # already mean()

        raise ValueError(f"Unknown normalize mode: {normalize}")

    def compute_complex_energy_norm_loss(
        self,
        pred_sol_real,
        pred_sol_imag,
        *,
        energy_inv: EnergyInvMode = "jacobi",
        cg_max_iter: int = 30,
        cg_tol: float = 1e-10,
        normalize: NormalizeMode = "rhs",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Energy-norm minimum residual loss:
            J = 0.5 * (r_r^T |K|^{-1} r_r + r_i^T |K|^{-1} r_i) / n_free

        Uses |diag(K)| = sqrt(diag(Kr)^2 + diag(Ki)^2) for the Jacobi
        preconditioner to be robust when Kr has zero-diagonal entries.

        energy_inv:
          - "jacobi": |K|^{-1} approx diag(|K|)^{-1}
          - "cg":     Kr^{-1} applied by CG solve on free DOFs
        """
        free_mask = self._free_dofs_mask()

        ar = self._to_tensor64(pred_sol_real)
        ai = self._to_tensor64(pred_sol_imag)

        Kr, Ki = (
            self.bilinear_form
        )  # (real stiffness, imag/mass-like) as sparse tensors

        # Residual blocks (full)
        Kr_ar = torch.sparse.mm(Kr, ar.unsqueeze(1)).squeeze(1)
        Ki_ai = torch.sparse.mm(Ki, ai.unsqueeze(1)).squeeze(1)
        r_real = Kr_ar - Ki_ai - self.linear_form

        Kr_ai = torch.sparse.mm(Kr, ai.unsqueeze(1)).squeeze(1)
        Ki_ar = torch.sparse.mm(Ki, ar.unsqueeze(1)).squeeze(1)
        r_imag = Kr_ai + Ki_ar  # assuming fi = 0

        # Restrict to free DOFs
        rr = r_real[free_mask]
        ri = r_imag[free_mask]

        n_free = rr.numel()
        if n_free == 0:
            return torch.zeros((), dtype=torch.float64, device=rr.device)

        # Build an operator for Kr_ff * x
        # We avoid explicitly forming Kr_ff by multiplying full then masking.
        def Kr_ff_mv(x_free: torch.Tensor) -> torch.Tensor:
            x_full = torch.zeros_like(ar)
            x_full[free_mask] = x_free
            y_full = torch.sparse.mm(Kr, x_full.unsqueeze(1)).squeeze(1)
            return y_full[free_mask]

        # Build diagonal preconditioner using |diag(K)| = sqrt(diag(Kr)^2 + diag(Ki)^2)
        # for robustness when Kr may have zero diagonal entries.
        diag_kr = _sparse_diag(Kr).to(dtype=torch.float64, device=rr.device)
        diag_ki = _sparse_diag(Ki).to(dtype=torch.float64, device=rr.device)
        diag_abs = torch.sqrt(diag_kr ** 2 + diag_ki ** 2)
        diag_free = diag_abs[free_mask].clamp_min(eps)

        def Minv(v_free: torch.Tensor) -> torch.Tensor:
            return v_free / diag_free

        # Define approximate inverse application
        if energy_inv == "jacobi":
            wr = Minv(rr)
            wi = Minv(ri)

        elif energy_inv == "cg":
            wr = _cg_solve(
                A_mv=Kr_ff_mv,
                b=rr,
                M_inv=Minv,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
            wi = _cg_solve(
                A_mv=Kr_ff_mv,
                b=ri,
                M_inv=Minv,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
        else:
            raise ValueError(f"Unknown energy_inv mode: {energy_inv}")

        # Energy-norm residual (scalar, >= 0)
        # J = 0.5 * (r^T w) / n_free
        loss = 0.5 * (torch.dot(rr, wr) + torch.dot(ri, wi)) / float(n_free)

        if normalize == "none":
            return loss

        if normalize == "rhs":
            # Scale by energy-norm of RHS on free DOFs (same inverse approx)
            fr = self.linear_form[free_mask]
            rhs_energy = torch.dot(fr, Minv(fr)).clamp_min(eps)
            return loss / rhs_energy

        if normalize == "ndof":
            return loss  # already / n_free

        raise ValueError(f"Unknown normalize mode: {normalize}")

    def compute_pi_abs_loss(
        self,
        pred_sol_real,
        pred_sol_imag,
        *,
        squared: bool = True,
        normalize: NormalizeMode = "ndof",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        WORKS PURELY!
        """
        free_mask = self._free_dofs_mask()

        ur = self._to_tensor64(pred_sol_real)
        ui = self._to_tensor64(pred_sol_imag)

        Kr, Ki = self.bilinear_form  # sparse real matrices

        # Ku real/imag parts
        Kr_ur = torch.sparse.mm(Kr, ur.unsqueeze(1)).squeeze(1)
        Ki_ui = torch.sparse.mm(Ki, ui.unsqueeze(1)).squeeze(1)
        Ku_r = Kr_ur - Ki_ui

        Ki_ur = torch.sparse.mm(Ki, ur.unsqueeze(1)).squeeze(1)
        Kr_ui = torch.sparse.mm(Kr, ui.unsqueeze(1)).squeeze(1)
        Ku_i = Ki_ur + Kr_ui

        # u^H Ku parts
        uHKu_real = torch.dot(ur, Ku_r) + torch.dot(ui, Ku_i)
        uHKu_imag = torch.dot(ur, Ku_i) - torch.dot(ui, Ku_r)

        # f^H u parts (assuming f is real)
        f = self.linear_form.to(dtype=torch.float64, device=ur.device)
        fHu_real = torch.dot(f, ur)
        fHu_imag = torch.dot(f, ui)

        Pi_real = 0.5 * uHKu_real - fHu_real
        Pi_imag = 0.5 * uHKu_imag - fHu_imag

        # |Pi| or |Pi|^2
        Pi_abs2 = Pi_real * Pi_real + Pi_imag * Pi_imag
        if squared:
            loss = Pi_abs2
        else:
            loss = torch.sqrt(Pi_abs2 + eps)

        # Optional normalization (purely for scale/stability)
        if normalize == "none":
            return loss

        if normalize == "ndof":
            n_free = free_mask.sum().clamp_min(1).to(dtype=torch.float64)
            return loss / n_free

        if normalize == "rhs":
            rhsn = torch.linalg.norm(f[free_mask]) + eps
            # scale like (|Pi| / ||f||)^2 if squared, else |Pi|/||f||
            return loss / (rhsn * rhsn) if squared else loss / rhsn

        raise ValueError(f"Unknown normalize mode: {normalize}")

    def compute_mixed_energy_norm_loss(
        self,
        pred_sol_a_real,
        pred_sol_a_imag,
        pred_sol_phi_real,
        pred_sol_phi_imag,
        *,
        energy_inv: EnergyInvMode = "jacobi",
        cg_max_iter: int = 30,
        cg_tol: float = 1e-10,
        normalize: NormalizeMode = "ndof",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Compute residual for mixed A-phi formulation.

        Uses |diag(K)| = sqrt(diag(Kr)^2 + diag(Ki)^2) for the Jacobi
        preconditioner so that phi DOFs (which have zero diagonal in Kr)
        are handled correctly.
        """
        n_A = self.n_dofs_A
        n_phi = self.n_dofs_phi
        n_total = n_A + n_phi

        # Convert inputs to tensors
        ar_A = self._to_tensor64(pred_sol_a_real)
        ai_A = self._to_tensor64(pred_sol_a_imag)
        ar_phi = self._to_tensor64(pred_sol_phi_real)
        ai_phi = self._to_tensor64(pred_sol_phi_imag)

        # Concatenate into full DOF vectors: [A_dofs | phi_dofs]
        ar = torch.zeros(n_total, device=self.device, dtype=torch.float64)
        ai = torch.zeros(n_total, device=self.device, dtype=torch.float64)

        ar[:n_A] = ar_A
        ar[n_A:] = ar_phi
        ai[:n_A] = ai_A
        ai[n_A:] = ai_phi

        # Get free DOFs mask for the product space
        free_mask = self._free_dofs_mask()

        Kr, Ki = self.bilinear_form  # (real stiffness, imag/mass-like) as sparse tensors

        # Residual blocks (full): r = K*u - f
        # For complex: (Kr + i*Ki)(ar + i*ai) = Kr*ar - Ki*ai + i*(Kr*ai + Ki*ar)
        Kr_ar = torch.sparse.mm(Kr, ar.unsqueeze(1)).squeeze(1)
        Ki_ai = torch.sparse.mm(Ki, ai.unsqueeze(1)).squeeze(1)
        r_real = Kr_ar - Ki_ai - self.linear_form

        Kr_ai = torch.sparse.mm(Kr, ai.unsqueeze(1)).squeeze(1)
        Ki_ar = torch.sparse.mm(Ki, ar.unsqueeze(1)).squeeze(1)
        r_imag = Kr_ai + Ki_ar  # assuming f_imag = 0

        # Restrict to free DOFs
        rr = r_real[free_mask]
        ri = r_imag[free_mask]

        n_free = rr.numel()
        if n_free == 0:
            return torch.zeros((), dtype=torch.float64, device=rr.device)

        # Build an operator for Kr_ff * x (free-free block)
        def Kr_ff_mv(x_free: torch.Tensor) -> torch.Tensor:
            x_full = torch.zeros_like(ar)
            x_full[free_mask] = x_free
            y_full = torch.sparse.mm(Kr, x_full.unsqueeze(1)).squeeze(1)
            return y_full[free_mask]

        # Build diagonal preconditioner using |diag(K)| = sqrt(diag(Kr)^2 + diag(Ki)^2).
        # This is essential for the mixed A-phi formulation because the curl-curl
        # term only contributes to Kr for A-DOFs, leaving diag(Kr) = 0 for all
        # phi-DOFs. Using only diag(Kr) would make the Jacobi inverse blow up.
        diag_kr = _sparse_diag(Kr).to(dtype=torch.float64, device=rr.device)
        diag_ki = _sparse_diag(Ki).to(dtype=torch.float64, device=rr.device)
        diag_abs = torch.sqrt(diag_kr ** 2 + diag_ki ** 2)
        diag_free = diag_abs[free_mask].clamp_min(eps)

        def Minv(v_free: torch.Tensor) -> torch.Tensor:
            return v_free / diag_free

        # Define approximate inverse application w = |K_ff|^{-1} r
        if energy_inv == "jacobi":
            wr = Minv(rr)
            wi = Minv(ri)

        elif energy_inv == "cg":
            wr = _cg_solve(
                A_mv=Kr_ff_mv,
                b=rr,
                M_inv=Minv,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
            wi = _cg_solve(
                A_mv=Kr_ff_mv,
                b=ri,
                M_inv=Minv,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
        else:
            raise ValueError(f"Unknown energy_inv mode: {energy_inv}")

        # Energy-norm residual (scalar, >= 0)
        # J = 0.5 * (r^T w) / n_free
        loss = 0.5 * (torch.dot(rr, wr) + torch.dot(ri, wi)) / float(n_free)

        if normalize == "none":
            return loss

        if normalize == "rhs":
            # Scale by energy-norm of RHS on free DOFs (same inverse approx)
            fr = self.linear_form[free_mask]
            rhs_energy = torch.dot(fr, Minv(fr)).clamp_min(eps)
            return loss / rhs_energy

        if normalize == "ndof":
            return loss  # already / n_free

        raise ValueError(f"Unknown normalize mode: {normalize}")

    def compute_mixed_energy_norm_loss_balanced(
        self,
        pred_sol_a_real,
        pred_sol_a_imag,
        pred_sol_phi_real,
        pred_sol_phi_imag,
        *,
        phi_weight: float = 1.0,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SEPARATE residual losses for A and phi components.
        
        This helps with the scale imbalance between A and phi by:
        1. Computing residuals for A and phi DOFs separately
        2. Normalizing each by their own DOF count
        3. Returning both so they can be weighted in the trainer
        
        Args:
            pred_sol_a_real: Real part of A prediction [n_dofs_A]
            pred_sol_a_imag: Imaginary part of A prediction [n_dofs_A]
            pred_sol_phi_real: Real part of phi prediction [n_dofs_phi]
            pred_sol_phi_imag: Imaginary part of phi prediction [n_dofs_phi]
            phi_weight: Weight for phi loss relative to A loss
            eps: Small value to avoid division by zero
            
        Returns:
            (loss_A, loss_phi, total_loss): Losses for A, phi, and weighted total
        """
        n_A = self.n_dofs_A
        n_phi = self.n_dofs_phi
        n_total = n_A + n_phi

        # Convert inputs to tensors
        ar_A = self._to_tensor64(pred_sol_a_real)
        ai_A = self._to_tensor64(pred_sol_a_imag)
        ar_phi = self._to_tensor64(pred_sol_phi_real)
        ai_phi = self._to_tensor64(pred_sol_phi_imag)

        # Concatenate into full DOF vectors: [A_dofs | phi_dofs]
        ar = torch.zeros(n_total, device=self.device, dtype=torch.float64)
        ai = torch.zeros(n_total, device=self.device, dtype=torch.float64)

        ar[:n_A] = ar_A
        ar[n_A:] = ar_phi
        ai[:n_A] = ai_A
        ai[n_A:] = ai_phi

        # Get free DOFs mask for the product space
        free_mask = self._free_dofs_mask()

        Kr, Ki = self.bilinear_form

        # Residual blocks (full): r = K*u - f
        Kr_ar = torch.sparse.mm(Kr, ar.unsqueeze(1)).squeeze(1)
        Ki_ai = torch.sparse.mm(Ki, ai.unsqueeze(1)).squeeze(1)
        r_real = Kr_ar - Ki_ai - self.linear_form

        Kr_ai = torch.sparse.mm(Kr, ai.unsqueeze(1)).squeeze(1)
        Ki_ar = torch.sparse.mm(Ki, ar.unsqueeze(1)).squeeze(1)
        r_imag = Kr_ai + Ki_ar

        # Build diagonal preconditioner
        diag_kr = _sparse_diag(Kr).to(dtype=torch.float64, device=self.device)
        diag_ki = _sparse_diag(Ki).to(dtype=torch.float64, device=self.device)
        diag_abs = torch.sqrt(diag_kr ** 2 + diag_ki ** 2).clamp_min(eps)

        # Separate masks for A and phi free DOFs
        free_mask_A = free_mask[:n_A]
        free_mask_phi = free_mask[n_A:]

        # --- A component loss ---
        rr_A = r_real[:n_A][free_mask_A]
        ri_A = r_imag[:n_A][free_mask_A]
        diag_A = diag_abs[:n_A][free_mask_A]
        
        n_free_A = rr_A.numel()
        if n_free_A > 0:
            wr_A = rr_A / diag_A
            wi_A = ri_A / diag_A
            loss_A = 0.5 * (torch.dot(rr_A, wr_A) + torch.dot(ri_A, wi_A)) / float(n_free_A)
        else:
            loss_A = torch.zeros((), dtype=torch.float64, device=self.device)

        # --- Phi component loss ---
        rr_phi = r_real[n_A:][free_mask_phi]
        ri_phi = r_imag[n_A:][free_mask_phi]
        diag_phi = diag_abs[n_A:][free_mask_phi]
        
        n_free_phi = rr_phi.numel()
        if n_free_phi > 0:
            wr_phi = rr_phi / diag_phi
            wi_phi = ri_phi / diag_phi
            loss_phi = 0.5 * (torch.dot(rr_phi, wr_phi) + torch.dot(ri_phi, wi_phi)) / float(n_free_phi)
        else:
            loss_phi = torch.zeros((), dtype=torch.float64, device=self.device)

        # Weighted total: equal weight means A and phi contribute equally
        total_loss = loss_A + phi_weight * loss_phi

        return loss_A, loss_phi, total_loss

    def _create_boundary_dofs_vector(
        self, problem: MeshProblemEM, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create a vector with Dirichlet boundary values at boundary DOFs and zeros at free DOFs.

        Args:
            problem: MeshProblem containing boundary conditions
            device: Target device for the tensor
            dtype: Target dtype for the tensor

        Returns:
            boundary_vector: Vector [N_dofs] with boundary values at boundary nodes, zeros elsewhere
        """
        n_total_dofs = len(self.fes.FreeDofs())
        boundary_vector = torch.zeros(n_total_dofs, device=device, dtype=dtype)

        # Get the mesh boundary segments and their corresponding boundary names
        # We need to map the boundary names to actual DOF indices
        if hasattr(problem, "dirichlet_values") and problem.dirichlet_values:
            # Create a grid function to set boundary values
            gfu_boundary = ng.GridFunction(self.fes)

            # Set boundary values using NGSolve's BoundaryCF
            try:
                boundary_cf = self.mesh.BoundaryCF(problem.dirichlet_values, default=0)
                gfu_boundary.Set(boundary_cf, ng.BND)

                # Extract the boundary values as a numpy array and convert to torch
                boundary_values_np = gfu_boundary.vec.FV().NumPy().copy()
                boundary_vector = torch.tensor(
                    boundary_values_np, device=device, dtype=dtype
                )

                # Zero out the free DOFs (keep only boundary values)
                free_dofs_bitarray = self.fes.FreeDofs()
                free_dofs_mask = torch.tensor(
                    [free_dofs_bitarray[i] for i in range(len(free_dofs_bitarray))],
                    dtype=torch.bool,
                    device=device,
                )
                boundary_vector[free_dofs_mask] = 0.0

            except Exception as e:
                print(f"Warning: Could not set boundary values automatically: {e}")
                print("Using zero boundary values")

        return boundary_vector

    def export_to_vtk(
        self,
        array_true,
        array_pred,
        filename="results/vtk/results.vtk",
    ):
        """
        Export solutions to VTK file for visualization in Paraview.
        """

        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        array_true = np.asarray(array_true, dtype=np.float64)
        array_pred = np.asarray(array_pred, dtype=np.float64)

        fes_real = ng.H1(self.mesh, order=self.order)

        gfu_true_real = ng.GridFunction(fes_real)
        gfu_pred_real = ng.GridFunction(fes_real)
        gfu_err_abs = ng.GridFunction(fes_real)
        gfu_err_rel = ng.GridFunction(fes_real)

        gfu_true_real.vec.FV().NumPy()[:] = np.real(array_true)

        gfu_pred_real.vec.FV().NumPy()[:] = np.real(array_pred)
        gfu_err_abs.vec.FV().NumPy()[:] = np.abs(array_true - array_pred)
        gfu_err_rel.vec.FV().NumPy()[:] = np.abs(array_true - array_pred) / (
            np.abs(array_true) + 1e-10
        )  # avoid division by zero

        coefs = [
            gfu_true_real,
            gfu_pred_real,
            gfu_err_abs,
            gfu_err_rel,
        ]
        names = [
            "ExactSolution_real",
            "PredictedSolution_real",
            "AbsError",
            "RelError",
        ]

        vtk_out = ng.VTKOutput(
            self.mesh,
            coefs=coefs,
            names=names,
            filename=str(filename),
            order=self.order,
        )
        vtk_out.Do()
        print(f"VTK file saved as {filename}")

        # save mesh
        file_path = Path(filename).parent.parent / "results_data"
        os.makedirs(file_path, exist_ok=True)
        mesh_filename = file_path / "mesh.vol"
        self.mesh.ngmesh.Save(str(mesh_filename))

        # save exact, predicted, difference as npz
        npz_filename = file_path / "results.npz"
        np.savez_compressed(
            npz_filename,
            exact=array_true,
            predicted=array_pred,
        )
        print(f"Results saved as {npz_filename}")

    def export_to_vtk_complex(
        self, array_true, array_pred, filename="results/vtk/results.vtk"
    ):
        """
        Export solutions to VTK file for visualization in Paraview.
        """
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        array_true = np.asarray(array_true, dtype=np.complex128)
        array_pred = np.asarray(array_pred, dtype=np.complex128)

        fes_complex = ng.H1(self.mesh, order=self.order, complex=True)
        fes_real = ng.H1(self.mesh, order=self.order, complex=False)

        gfu_true = ng.GridFunction(fes_complex)
        gfu_pred = ng.GridFunction(fes_complex)
        gfu_err_abs = ng.GridFunction(fes_real)
        gfu_err_rel = ng.GridFunction(fes_real)

        gfu_true.vec.FV().NumPy()[:] = array_true
        gfu_pred.vec.FV().NumPy()[:] = array_pred
        gfu_err_abs.vec.FV().NumPy()[:] = np.abs(array_true - array_pred)
        gfu_err_rel.vec.FV().NumPy()[:] = np.abs(array_true - array_pred) / (
            np.abs(array_true) + 1e-10
        )  # avoid division by zero

        coefs = [
            gfu_true.real,
            gfu_true.imag,
            ng.Norm(gfu_true),
            gfu_pred.real,
            gfu_pred.imag,
            ng.Norm(gfu_pred),
            gfu_err_abs,
            gfu_err_rel,
        ]
        names = [
            "ExactSolution_real",
            "ExactSolution_imag",
            "ExactSolution_abs",
            "PredictedSolution_real",
            "PredictedSolution_imag",
            "PredictedSolution_abs",
            "AbsError",
            "RelError",
        ]

        vtk_out = ng.VTKOutput(
            self.mesh,
            coefs=coefs,
            names=names,
            filename=str(filename),
            order=self.order,
        )
        vtk_out.Do()
        print(f"VTK file saved as {filename}")

        # save mesh
        file_path = Path(filename).parent.parent / "results_data"
        os.makedirs(file_path, exist_ok=True)
        mesh_filename = file_path / "mesh.vol"
        self.mesh.ngmesh.Save(str(mesh_filename))

        # save exact, predicted, difference as npz
        npz_filename = file_path / "results.npz"
        np.savez_compressed(
            npz_filename,
            exact=array_true,
            predicted=array_pred,
        )
        print(f"Results saved as {npz_filename}")

    def export_to_vtk_mixed(
        self, array_true, array_pred, filename="results/vtk/results_mixed.vtk"
    ):
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        array_true = np.asarray(array_true, dtype=np.complex128)
        array_pred = np.asarray(array_pred, dtype=np.complex128)

        n_A = self.n_dofs_A
        n_phi = self.n_dofs_phi

        # Split into A and phi components
        A_true = array_true[:n_A]
        A_pred = array_pred[:n_A]
        phi_true = array_true[n_A:n_A + n_phi]
        phi_pred = array_pred[n_A:n_A + n_phi]

        # Create GridFunctions for A (full mesh)
        fes_A = ng.H1(self.mesh, order=self.order, complex=True)
        fes_A_real = ng.H1(self.mesh, order=self.order, complex=False)
        
        gfA_true = ng.GridFunction(fes_A)
        gfA_pred = ng.GridFunction(fes_A)
        gfA_err_abs = ng.GridFunction(fes_A_real)
        
        gfA_true.vec.FV().NumPy()[:] = A_true
        gfA_pred.vec.FV().NumPy()[:] = A_pred
        gfA_err_abs.vec.FV().NumPy()[:] = np.abs(A_true - A_pred)

        # Create GridFunctions for phi (coil region only)
        fes_phi = ng.H1(
            self.mesh, order=self.order, complex=True,
            definedon=self.mesh.Materials("mat_coil")
        )
        fes_phi_real = ng.H1(
            self.mesh, order=self.order, complex=False,
            definedon=self.mesh.Materials("mat_coil")
        )
        
        gfPhi_true = ng.GridFunction(fes_phi)
        gfPhi_pred = ng.GridFunction(fes_phi)
        gfPhi_err_abs = ng.GridFunction(fes_phi_real)
        
        gfPhi_true.vec.FV().NumPy()[:] = phi_true
        gfPhi_pred.vec.FV().NumPy()[:] = phi_pred
        gfPhi_err_abs.vec.FV().NumPy()[:] = np.abs(phi_true - phi_pred)

        coefs = [
            gfA_true.real,
            gfA_true.imag,
            ng.Norm(gfA_true),
            gfA_pred.real,
            gfA_pred.imag,
            ng.Norm(gfA_pred),
            gfA_err_abs,
            gfPhi_true.real,
            gfPhi_true.imag,
            ng.Norm(gfPhi_true),
            gfPhi_pred.real,
            gfPhi_pred.imag,
            ng.Norm(gfPhi_pred),
            gfPhi_err_abs,
        ]
        names = [
            "A_true_real",
            "A_true_imag",
            "A_true_abs",
            "A_pred_real",
            "A_pred_imag",
            "A_pred_abs",
            "A_error_abs",
            "Phi_true_real",
            "Phi_true_imag",
            "Phi_true_abs",
            "Phi_pred_real",
            "Phi_pred_imag",
            "Phi_pred_abs",
            "Phi_error_abs",
        ]

        vtk_out = ng.VTKOutput(
            self.mesh,
            coefs=coefs,
            names=names,
            filename=str(filename),
            order=self.order,
        )
        vtk_out.Do()
        print(f"VTK file saved as {filename}")

        # Save mesh and data
        file_path = Path(filename).parent.parent / "results_data"
        os.makedirs(file_path, exist_ok=True)
        mesh_filename = file_path / "mesh.vol"
        self.mesh.ngmesh.Save(str(mesh_filename))

        npz_filename = file_path / "results_mixed.npz"
        np.savez_compressed(
            npz_filename,
            A_true=A_true,
            A_pred=A_pred,
            phi_true=phi_true,
            phi_pred=phi_pred,
        )
        print(f"Mixed results saved as {npz_filename}")

def previous_em():
    import ngsolve as ng
    from containers import MeshProblemEM
    from graph_creator import GraphCreator
    from train_problems import create_em_problem, create_em_problem_complex

    problem = create_em_problem_complex()

    # Initialize FEM solver
    fem_solver = FEMSolverEM(problem.mesh, order=1, problem=problem)

    gfA = fem_solver.solve(problem)

    if np.iscomplexobj(gfA):
        gfA_real = np.real(gfA)
        gfA_imag = np.imag(gfA)
        residual_int = fem_solver.compute_complex_residual(
            torch.tensor(gfA_real, dtype=torch.float64),
            torch.tensor(gfA_imag, dtype=torch.float64),
        )
        print(f"Complex energy residual: {residual_int.item()}")
    else:
        residual_int = fem_solver.compute_energy_loss(gfA)
        print(f"Energy residual: {residual_int.item()}")

        residual = fem_solver.compute_residual(gfA)
        residuals_abs = np.absolute(residual.cpu().numpy())
        print(f"Mean residual: {np.mean(residuals_abs)}")

    # fem_solver.export_to_vtk(
    #     curl_gfa,
    #     curl_gfa,
    #     filename="results/fem_tests_em/vtk/result",
    # )

def mixed_em():
    import ngsolve as ng
    from containers import MeshProblemEM
    from graph_creator import GraphCreator
    from train_problems import create_em_mixed

    problem = create_em_mixed()

    # Initialize FEM solver
    fem_solver = FEMSolverEM(problem.mesh, order=1, problem=problem)

    gfA, gfPhi, r1 = fem_solver.solve_mixed_em(problem)
    # E_phi = -1j * problem.omega * (gfA + gfPhi * r1)

    fem_solver.export_to_vtk_complex(
        gfA.vec.FV().NumPy(),
        gfA.vec.FV().NumPy(),
        filename="results/fem_tests_em/vtk/mixed_result",
    )

    residual = fem_solver.compute_mixed_energy_norm_loss(
        pred_sol_a_real=gfA.vec.FV().NumPy().real,
        pred_sol_a_imag=gfA.vec.FV().NumPy().imag,
        pred_sol_phi_real=gfPhi.vec.FV().NumPy().real,
        pred_sol_phi_imag=gfPhi.vec.FV().NumPy().imag,
        energy_inv="jacobi",
        normalize="ndof",
    )
    print(f"Mixed formulation energy norm residual: {residual.item()}")


if __name__ == "__main__":

    # previous_em()
    mixed_em()