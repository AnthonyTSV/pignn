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
mu_star = 4 * 3.1415926535e-7 # H/m
J_star = A_star / (r_star**2 * mu_star)

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
        if problem is not None:
            if problem.complex:
                self.init_complex_matrices()
            else:
                self.init_matrices()

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

        kappa = self.problem.omega * mu_star * (r_star**2)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + inv_r * dr_rA * dr_rv) * ng.dx
        a += 1j * kappa * sigma * r * A * v * ng.dx("mat_coil")
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

    def _ngsolve_to_torch(self, ngsolve_matrix):
        """Convert NGSolve matrix to torch tensor."""

        def _coo_to_tensor(rows, cols, vals, shape, dtype):
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=dtype)
            tensor = torch.sparse_coo_tensor(
                indices, values, size=shape, dtype=dtype
            )
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

        kappa = self.problem.omega * mu_star * (r_star**2)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + inv_r * dr_rA * dr_rv) * ng.dx
        a += 1j * kappa * sigma * r * A * v * ng.dx("mat_coil")
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

        Ax = torch.sparse.mm(self.bilinear_form, pred_full.unsqueeze(1)).squeeze(
            1
        )
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

    def compute_complex_residual(self, pred_sol_real, pred_sol_imag, *, normalize: NormalizeMode = "ndof", eps: float = 1e-12) -> torch.Tensor:
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

        loss = (r_real_free.square().mean() + r_imag_free.square().mean())

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
        normalize: NormalizeMode = "ndof",
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Energy-norm minimum residual loss:
            J = 0.5 * (r_r^T Kr^{-1} r_r + r_i^T Kr^{-1} r_i) / n_free

        energy_inv:
          - "jacobi": Kr^{-1} approx diag(Kr)^{-1}
          - "cg":     Kr^{-1} applied by CG solve on free DOFs
        """
        free_mask = self._free_dofs_mask()

        ar = self._to_tensor64(pred_sol_real)
        ai = self._to_tensor64(pred_sol_imag)

        Kr, Ki = self.bilinear_form  # (real stiffness, imag/mass-like) as sparse tensors

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

        # Define approximate inverse application w = Kr_ff^{-1} r
        if energy_inv == "jacobi":
            diag_full = _sparse_diag(Kr).to(dtype=torch.float64, device=rr.device)
            diag_free = diag_full[free_mask].clamp_min(eps)

            def Minv(v_free: torch.Tensor) -> torch.Tensor:
                return v_free / diag_free

            wr = Minv(rr)
            wi = Minv(ri)

        elif energy_inv == "cg":
            # Preconditioner: Jacobi on Kr_ff (helps CG)
            diag_full = _sparse_diag(Kr).to(dtype=torch.float64, device=rr.device)
            diag_free = diag_full[free_mask].clamp_min(eps)

            def Minv(v_free: torch.Tensor) -> torch.Tensor:
                return v_free / diag_free

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
            if energy_inv == "jacobi":
                rhs_energy = torch.dot(fr, fr / diag_free).clamp_min(eps)
            else:
                # CG for rhs norm too (can be expensive); use Jacobi norm here as a stable baseline.
                rhs_energy = torch.dot(fr, fr / diag_free).clamp_min(eps)

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
        Loss = |Pi(u)|  or |Pi(u)|^2, where
            Pi(u) = 0.5 * u^H (Kr + i Ki) u - f^H u.

        Notes:
        - pred_sol_real/pred_sol_imag should already have Dirichlet enforced
          (or you enforce them before calling).
        - We compute Pi using the full DOF vectors (recommended), then optionally scale.
        - If squared=True, returns |Pi|^2 (smoother, avoids sqrt).
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
        gfu_err_rel.vec.FV().NumPy()[:] = np.abs(array_true - array_pred) / (np.abs(
            array_true
        ) + 1e-10) # avoid division by zero

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

    def export_to_vtk_complex(self, array_true, array_pred, filename="results/vtk/results.vtk"):
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
        gfu_err_rel.vec.FV().NumPy()[:] = np.abs(array_true - array_pred) / (np.abs(
            array_true
        ) + 1e-10) # avoid division by zero

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


if __name__ == "__main__":

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
        residual_int = fem_solver.compute_pi_abs_loss(
            torch.tensor(gfA_real, dtype=torch.float64),
            torch.tensor(gfA_imag, dtype=torch.float64),
            squared=True, normalize="ndof",
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
