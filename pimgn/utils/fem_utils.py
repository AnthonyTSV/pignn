"""
FEM utilities for computing residuals and assembling operators.
Based on the methodology section in the paper.
"""

import numpy as np
import scipy.sparse as sp
import torch
import ngsolve as ng
from ngsolve import Mesh, H1, BilinearForm, LinearForm, grad, dx


class FEMSolver:
    """
    FEM solver for the heat equation using NGSolve.
    Implements the weak form from the paper methodology with strict Dirichlet BC enforcement.
    """
    
    def __init__(self, mesh, alpha=1.0, order=1, dirichlet_boundaries="left|right|top|bottom"):
        """
        Initialize FEM solver.
        
        Args:
            mesh: NGSolve mesh
            alpha: Thermal diffusivity coefficient
            order: Finite element order (default 1 for linear elements)
            dirichlet_boundaries: String specifying Dirichlet boundary regions
        """
        self.mesh = mesh
        self.alpha = alpha
        self.order = order
        self.dirichlet_boundaries = dirichlet_boundaries
        
        # Create finite element space with Dirichlet boundaries
        self.fes = H1(mesh, order=order, dirichlet=dirichlet_boundaries)
        self.ndof = self.fes.ndof
        
        # Get free degrees of freedom (excluding Dirichlet nodes)
        self.free_dofs = self.fes.FreeDofs()
        self.n_free_dofs = sum(self.free_dofs)
        
        # Create trial and test functions
        self.u = self.fes.TrialFunction()
        self.v = self.fes.TestFunction()
        
        # Assemble mass and stiffness matrices
        self._assemble_matrices()
    
    def _assemble_matrices(self):
        """
        Assemble mass and stiffness matrices.
        
        Following the paper methodology: matrices are assembled for free DOFs only
        to strictly enforce Dirichlet boundary conditions.
        """
        # Mass matrix
        mass_form = BilinearForm(self.fes, symmetric=True, check_unused=False)
        mass_form += self.u * self.v * dx
        mass_form.Assemble()
        self.mass_matrix_ngsolve = mass_form.mat
        
        # Stiffness matrix
        stiff_form = BilinearForm(self.fes, symmetric=True, check_unused=False)
        stiff_form += self.alpha * grad(self.u) * grad(self.v) * dx
        stiff_form.Assemble()
        self.stiff_matrix_ngsolve = stiff_form.mat
        
        # Convert to scipy matrices - full system first
        self.mass_matrix_full = self._ngsolve_to_scipy(mass_form.mat)
        self.stiff_matrix_full = self._ngsolve_to_scipy(stiff_form.mat)
        
        # Extract reduced matrices for free DOFs only
        # This implements the strict BC enforcement from the paper
        free_indices = np.where(self.free_dofs)[0]
        bc_indices = np.where(~self.free_dofs)[0]
        
        # Reduced mass matrix (M_FF in FEM notation)
        self.mass_matrix_reduced = self.mass_matrix_full[np.ix_(free_indices, free_indices)]
        
        # Reduced stiffness matrix (K_FF in FEM notation)  
        self.stiff_matrix_reduced = self.stiff_matrix_full[np.ix_(free_indices, free_indices)]
        
        # Boundary coupling matrix (K_FB in FEM notation)
        # This couples free DOFs to boundary DOFs
        self.stiff_coupling_matrix = self.stiff_matrix_full[np.ix_(free_indices, bc_indices)]
        
        # Store indices for later use
        self.free_indices = free_indices
        self.bc_indices = bc_indices
        
        # For compatibility, keep full matrices as default
        self.mass_matrix = self.mass_matrix_full
        self.stiff_matrix = self.stiff_matrix_full
    
    def _ngsolve_to_scipy(self, ngsolve_matrix):
        """Convert NGSolve matrix to scipy sparse matrix."""
        if hasattr(ngsolve_matrix, "COO"):
            rows, cols, vals = ngsolve_matrix.COO()
            return sp.csr_matrix((vals, (rows, cols)), 
                               shape=(ngsolve_matrix.height, ngsolve_matrix.width))
        else:
            # Fallback to dense conversion
            dense = np.array(ngsolve_matrix.ToDenseMatrix(), dtype=np.float64)
            return sp.csr_matrix(dense)
    
    def solve_time_problem(self, T0, dt, nsamples, t0=0.0, t_final=1.0):
        states = []
        mstar = self.mass_matrix_ngsolve.CreateMatrix()
        mstar.AsVector().data = self.mass_matrix_ngsolve.AsVector() + dt * self.stiff_matrix_ngsolve.AsVector()
        invmstar = mstar.Inverse(freedofs=self.fes.FreeDofs())
        gfu = ng.GridFunction(self.fes)
        gfu.vec.FV().NumPy()[:] = T0
        states.append(T0)
        cnt = 0; time = t0
        import math
        sample_int = int(math.floor(t_final / dt / nsamples)+1)
        gfut = ng.GridFunction(gfu.space,multidim=0)
        gfut.AddMultiDimComponent(gfu.vec)
        while time < t_final - 0.5 * dt:
            res = - dt * self.stiff_matrix_ngsolve * gfu.vec
            gfu.vec.data += invmstar * res
            print("\r",time,end="")
            if cnt % sample_int == 0:
                gfut.AddMultiDimComponent(gfu.vec)
            cnt += 1; time = cnt * dt
            states.append(gfu.vec.FV().NumPy().copy())
        return states


def compute_fem_residual_strict_bc(predictions_free, T_prev, problem, t_current, t_prev, fem_solver, dt):
    """
    Compute FEM residual with strict Dirichlet boundary condition enforcement.
    
    Based on the paper methodology section - this eliminates competing training losses
    by only computing residuals for free degrees of freedom (test functions that are
    not 1 on the Dirichlet boundary).
    
    This implements the weak form discretized as:
    M_FF(T_F^{n+1} - T_F^n)/dt + K_FF*T_F^{n+1} + K_FB*T_Γ^{n+1} + M_FB*(T_Γ^{n+1} - T_Γ^n)/dt = 0
    
    Where:
    - M_FF: Mass matrix for free DOFs
    - K_FF: Stiffness matrix for free DOFs
    - K_FB: Coupling matrix (free to boundary)
    - M_FB: Mass coupling matrix (for time-varying BC)
    - T_F: Free DOF temperatures
    - T_Γ: Boundary temperatures
    
    Args:
        predictions_free: Predicted temperature values for FREE DOFs only (N_free,)
        T_prev: Previous temperature values for ALL DOFs (N_total,)
        problem: MeshProblem instance for boundary condition specification
        t_current: Current time
        t_prev: Previous time
        fem_solver: FEMSolver instance with assembled matrices
        dt: Time step size
    
    Returns:
        residual: FEM residual for free test functions only (N_free,)
    """
    device = predictions_free.device if hasattr(predictions_free, 'device') else 'cpu'
    
    # Ensure predictions is 1D and only for free DOFs
    if predictions_free.dim() > 1:
        predictions_free = predictions_free.squeeze()
    
    # Convert to torch tensors if needed
    if not isinstance(T_prev, torch.Tensor):
        T_prev = torch.tensor(T_prev, dtype=torch.float32, device=device)
    
    # Get free and boundary DOF indices
    free_indices = fem_solver.free_indices
    bc_indices = fem_solver.bc_indices
    
    # Extract free DOF values from previous state
    T_prev_free = T_prev[free_indices]
    
    # Convert reduced matrices to torch tensors
    mass_matrix_reduced = torch.tensor(
        fem_solver.mass_matrix_reduced.toarray(), 
        dtype=torch.float32, device=device
    )
    stiff_matrix_reduced = torch.tensor(
        fem_solver.stiff_matrix_reduced.toarray(),
        dtype=torch.float32, device=device
    )
    stiff_coupling_matrix = torch.tensor(
        fem_solver.stiff_coupling_matrix.toarray(),
        dtype=torch.float32, device=device
    )
    
    # Get boundary values for current and previous times
    T_boundary_current = get_boundary_values_tensor(problem, t_current, device)[bc_indices]
    T_boundary_prev = get_boundary_values_tensor(problem, t_prev, device)[bc_indices]
    
    # Compute base residual: M_FF(T_F^{n+1} - T_F^n)/dt + K_FF*T_F^{n+1}
    time_derivative_free = torch.matmul(mass_matrix_reduced, (predictions_free - T_prev_free)) / dt
    diffusion_term_free = torch.matmul(stiff_matrix_reduced, predictions_free)
    
    # Add boundary coupling term: K_FB @ T_Γ^{n+1}
    boundary_coupling_term = torch.matmul(stiff_coupling_matrix, T_boundary_current)
    
    # Add time-varying Dirichlet term: M_FB * (T_Γ^{n+1} - T_Γ^n)/dt
    # For homogeneous Dirichlet BC (T_Γ = 0), this term is zero
    # For time-varying BC, this term is needed
    if torch.allclose(T_boundary_current, T_boundary_prev):
        # Homogeneous or time-constant BC: no time-varying term
        time_varying_bc_term = torch.zeros_like(predictions_free)
    else:
        # Time-varying BC: include M_FB coupling
        mass_coupling_matrix = torch.tensor(
            fem_solver.mass_matrix_full[np.ix_(free_indices, bc_indices)].toarray(),
            dtype=torch.float32, device=device
        )
        time_varying_bc_term = torch.matmul(mass_coupling_matrix, 
                                           (T_boundary_current - T_boundary_prev)) / dt
    
    # Total residual for free test functions
    residual = time_derivative_free + diffusion_term_free + boundary_coupling_term + time_varying_bc_term
    
    return residual


def compute_fem_residual(predictions, T_prev, mass_matrix, stiff_matrix, dt, 
                        boundary_mask=None, boundary_values=None):
    """
    Compute FEM residual for physics-informed loss.
    
    Legacy function maintained for compatibility.
    For strict BC enforcement, use compute_fem_residual_strict_bc instead.
    
    Based on equation (8) from the paper
    
    For backward Euler discretization:
    Residual = M(T^{n+1} - T^n)/dt + K*T^{n+1}
    
    Args:
        predictions: Predicted temperature values (N,) or (N, 1)
        T_prev: Previous temperature values (N,)
        mass_matrix: Mass matrix (scipy sparse or torch tensor)
        stiff_matrix: Stiffness matrix (scipy sparse or torch tensor)
        dt: Time step size
        boundary_mask: Boolean mask for boundary nodes
        boundary_values: Boundary temperature values
    
    Returns:
        residual: FEM residual for each test function
    """
    device = predictions.device if hasattr(predictions, 'device') else 'cpu'
    
    # Ensure predictions is 1D
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    
    # Convert to torch tensors if needed
    if not isinstance(T_prev, torch.Tensor):
        T_prev = torch.tensor(T_prev, dtype=torch.float32, device=device)
    
    # Convert matrices to torch tensors if needed
    if not isinstance(mass_matrix, torch.Tensor):
        if sp.issparse(mass_matrix):
            mass_matrix = torch.tensor(mass_matrix.toarray(), dtype=torch.float32, device=device)
        else:
            mass_matrix = torch.tensor(mass_matrix, dtype=torch.float32, device=device)
    
    if not isinstance(stiff_matrix, torch.Tensor):
        if sp.issparse(stiff_matrix):
            stiff_matrix = torch.tensor(stiff_matrix.toarray(), dtype=torch.float32, device=device)
        else:
            stiff_matrix = torch.tensor(stiff_matrix, dtype=torch.float32, device=device)
    
    # Apply boundary conditions to predictions if specified
    if boundary_mask is not None and boundary_values is not None:
        predictions = predictions.clone()
        predictions[boundary_mask] = boundary_values[boundary_mask]
    
    # Compute residual: M(T^{n+1} - T^n)/dt + K*T^{n+1}
    time_derivative = torch.matmul(mass_matrix, (predictions - T_prev)) / dt
    diffusion_term = torch.matmul(stiff_matrix, predictions)
    residual = time_derivative + diffusion_term
    
    return residual


def compute_fem_loss(residuals):
    """
    Compute FEM loss as MSE of residuals.
    
    Based on equation (9) from the paper
    
    For single time step (no temporal bundling):
    L_FEM = 1/N_φ * Σ_{m=1}^{N_φ} (ε_{FEM,m})^2
    """
    return torch.mean(residuals ** 2)

def compute_fem_loss_bundle(model_output_bundle, T_start, fem_solver, dt, boundary_values):
    """
    Compute FEM loss over a temporal bundle of predictions.

    This implements the temporal bundling loss from the paper:
    L_FEM^TB = (1 / (N_TB * N_φ,F)) * Σ_{k=1}^{N_TB} ||ε_FEM^{n+k}||^2

    Args:
        model_output_bundle: Model predictions for all DOFs over the bundle (N_total, N_TB)
        T_start: Temperature state at the beginning of the bundle (N_total,)
        fem_solver: FEMSolver instance
        dt: Time step size
        boundary_values: Boundary condition values for ALL nodes (N_total,)

    Returns:
        loss: The mean squared residual loss over the entire bundle.
    """
    device = model_output_bundle.device
    n_tb = model_output_bundle.shape[1]
    all_residuals = []

    T_prev = T_start.clone()

    for k in range(n_tb):
        # Get the model's full prediction for the current step in the bundle
        T_pred_full = model_output_bundle[:, k]

        # Extract free DOF predictions for loss calculation
        predictions_free = extract_free_dofs(T_pred_full, fem_solver)

        # Reconstruct the full solution for this step to handle boundary terms correctly
        T_full_current = reconstruct_full_solution(predictions_free, boundary_values, fem_solver)

        # Compute residual for this step
        # T_prev is the state from the previous step in the bundle
        residual_k = compute_fem_residual_strict_bc(
            predictions=predictions_free,
            T_prev=T_prev,
            fem_solver=fem_solver,
            dt=dt,
            T_full=T_full_current
        )
        all_residuals.append(residual_k)

        # Update T_prev for the next iteration in the bundle
        T_prev = T_full_current.detach() # Detach to stop gradients flowing through time

    # Concatenate all residuals and compute the final loss
    total_residuals = torch.cat(all_residuals)
    loss = torch.mean(total_residuals ** 2)

    return loss

def get_boundary_conditions(problem, t):
    """
    Get boundary condition values for a specific time.
    
    For this heat equation problem, we use homogeneous Dirichlet boundary conditions:
    T = 0 on all boundaries for all time t.
    
    This function can be extended to support time-varying boundary conditions
    by modifying the return values based on problem parameters and time.
    
    Args:
        problem: MeshProblem instance containing problem information
        t: Current time
    
    Returns:
        boundary_values: Values for ALL nodes (N_total,) with correct BC values at boundary
    """
    n_nodes = problem.graph_data['pos'].shape[0]
    boundary_mask = problem.graph_data['boundary_mask']
    
    # Initialize with zeros (homogeneous Dirichlet BC)
    boundary_values = np.zeros(n_nodes)
    
    # For homogeneous Dirichlet BC, boundary values are always 0
    # This could be extended for time-varying or non-homogeneous cases:
    # boundary_values[boundary_mask] = some_function_of_time_and_position(t, positions[boundary_mask])
    
    return boundary_values


def get_boundary_values_tensor(problem, t, device='cpu'):
    """
    Get boundary condition values as a PyTorch tensor.
    
    Args:
        problem: MeshProblem instance
        t: Current time
        device: PyTorch device
    
    Returns:
        torch.Tensor: Boundary values for all nodes (N_total,)
    """
    boundary_values = get_boundary_conditions(problem, t)
    return torch.tensor(boundary_values, dtype=torch.float32, device=device)
    """
    Apply Dirichlet boundary conditions to predictions.
    
    Args:
        predictions: Predicted values (N,)
        boundary_mask: Boolean mask for boundary nodes (N,)
        boundary_values: Boundary values (N,)
    
    Returns:
        predictions: Modified predictions with boundary conditions applied
    """
    predictions = predictions.clone()
    predictions[boundary_mask] = boundary_values[boundary_mask]
    return predictions


def apply_dirichlet_bc(predictions, boundary_mask, boundary_values):
    """
    Apply Dirichlet boundary conditions to predictions.
    
    NOTE: This function should only be used in legacy mode.
    With strict BC enforcement, boundary values are handled through 
    matrix structure and reconstruction functions.
    
    Args:
        predictions: Predicted values (N,)
        boundary_mask: Boolean mask for boundary nodes (N,)
        boundary_values: Boundary values (N,)
    
    Returns:
        predictions: Modified predictions with boundary conditions applied
    """
    predictions = predictions.clone()
    predictions[boundary_mask] = boundary_values[boundary_mask]
    return predictions


def reconstruct_full_solution(predictions_free, problem, t, fem_solver, device='cpu'):
    """
    Reconstruct full solution from free DOF predictions and time-varying boundary conditions.
    
    This implements the paper's approach: T_v^{n+1} = T_{v,F}^{n+1} + T_{v,Γ}^{n+1}
    where T_{v,F}^{n+1} is non-zero on free nodes and T_{v,Γ}^{n+1} is non-zero on boundary.
    
    Args:
        predictions_free: Predicted values for free DOFs only (N_free,)
        problem: MeshProblem instance for boundary condition specification
        t: Current time for time-varying boundary conditions
        fem_solver: FEMSolver instance with free DOF information
        device: PyTorch device
    
    Returns:
        full_solution: Complete solution for all DOFs (N_total,)
    """
    if not isinstance(predictions_free, torch.Tensor):
        predictions_free = torch.tensor(predictions_free, dtype=torch.float32, device=device)
    
    # Get time-appropriate boundary values
    boundary_values = get_boundary_values_tensor(problem, t, device)
    
    # Initialize full solution with boundary values
    full_solution = boundary_values.clone()
    
    # Use stored free DOF indices
    free_indices = fem_solver.free_indices
    
    # Set free DOF values from predictions
    full_solution[free_indices] = predictions_free
    
    return full_solution


def extract_free_dofs(full_values, fem_solver):
    """
    Extract values for free degrees of freedom only.
    
    Args:
        full_values: Values for all DOFs (N_total,)
        fem_solver: FEMSolver instance with free DOF information
    
    Returns:
        free_values: Values for free DOFs only (N_free,)
    """
    device = full_values.device if hasattr(full_values, 'device') else 'cpu'
    
    if not isinstance(full_values, torch.Tensor):
        full_values = torch.tensor(full_values, dtype=torch.float32, device=device)
    
    # Use stored free DOF indices
    free_indices = fem_solver.free_indices
    
    return full_values[free_indices]


def compute_fem_residual_strict_bc_time_varying(predictions, T_prev, fem_solver, dt, 
                                               T_full_current, T_full_prev):
    """
    Compute FEM residual with strict Dirichlet BC enforcement for time-varying boundary conditions.
    
    This extends compute_fem_residual_strict_bc to handle time-varying Dirichlet boundary conditions
    by adding the time-varying boundary term: M_FB * (T_Γ^{n+1} - T_Γ^n)/dt
    
    Args:
        predictions: Predicted temperature values for FREE DOFs only (N_free,)
        T_prev: Previous temperature values for ALL DOFs (N_total,)
        fem_solver: FEMSolver instance with assembled matrices
        dt: Time step size
        T_full_current: Full reconstructed vector for current time step (N_total,)
        T_full_prev: Full reconstructed vector for previous time step (N_total,)
    
    Returns:
        residual: FEM residual for free test functions only (N_free,)
    """
    device = predictions.device if hasattr(predictions, 'device') else 'cpu'
    
    # Ensure predictions is 1D and only for free DOFs
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    
    # Convert to torch tensors if needed
    if not isinstance(T_prev, torch.Tensor):
        T_prev = torch.tensor(T_prev, dtype=torch.float32, device=device)
    if not isinstance(T_full_current, torch.Tensor):
        T_full_current = torch.tensor(T_full_current, dtype=torch.float32, device=device)
    if not isinstance(T_full_prev, torch.Tensor):
        T_full_prev = torch.tensor(T_full_prev, dtype=torch.float32, device=device)
    
    # Get free and boundary DOF indices
    free_indices = fem_solver.free_indices
    bc_indices = fem_solver.bc_indices
    
    # Extract free DOF values from previous state
    T_prev_free = T_prev[free_indices]
    
    # Convert reduced matrices to torch tensors
    mass_matrix_reduced = torch.tensor(
        fem_solver.mass_matrix_reduced.toarray(), 
        dtype=torch.float32, device=device
    )
    stiff_matrix_reduced = torch.tensor(
        fem_solver.stiff_matrix_reduced.toarray(),
        dtype=torch.float32, device=device
    )
    stiff_coupling_matrix = torch.tensor(
        fem_solver.stiff_coupling_matrix.toarray(),
        dtype=torch.float32, device=device
    )
    
    # For time-varying BC, we also need M_FB coupling matrix
    # This is the mass matrix coupling free DOFs to boundary DOFs
    mass_coupling_matrix = torch.tensor(
        fem_solver.mass_matrix_full[np.ix_(free_indices, bc_indices)].toarray(),
        dtype=torch.float32, device=device
    )

    # Compute base residual: M_FF(T_F^{n+1} - T_F^n)/dt + K_FF*T_F^{n+1}
    time_derivative = torch.matmul(mass_matrix_reduced, (predictions - T_prev_free)) / dt
    diffusion_term = torch.matmul(stiff_matrix_reduced, predictions)
    
    # Add boundary coupling term: K_FB @ T_Γ^{n+1}
    T_boundary_current = T_full_current[bc_indices]
    boundary_coupling_term = torch.matmul(stiff_coupling_matrix, T_boundary_current)
    
    # Add time-varying Dirichlet term: M_FB * (T_Γ^{n+1} - T_Γ^n)/dt
    T_boundary_prev = T_full_prev[bc_indices]
    time_varying_bc_term = torch.matmul(mass_coupling_matrix, 
                                       (T_boundary_current - T_boundary_prev)) / dt
    
    residual = time_derivative + diffusion_term + boundary_coupling_term + time_varying_bc_term
    
    return residual
