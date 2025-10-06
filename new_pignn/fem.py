import numpy as np
from containers import MeshConfig, MeshProblem, TimeConfig
import ngsolve as ng
from typing import Optional, List, Tuple
import torch
import scipy.sparse as sp
import os
from mesh_utils import create_rectangular_mesh, build_graph_from_mesh


class FEMSolver:
    def __init__(self, mesh: ng.Mesh, order=1, problem: Optional[MeshProblem] = None):
        self.mesh = mesh
        self.order = order
        self.fes = ng.H1(mesh, order=order, dirichlet="left|right|top|bottom")
        self.problem = problem
        if problem is not None:
            self.init_matrices()

    def init_matrices(self):
        if self.problem is None:
            raise ValueError("Problem must be set before initializing matrices")
        # Initialize stiffness matrix and mass matrix
        u = self.fes.TrialFunction()
        v = self.fes.TestFunction()

        self.stiffness_matrix = ng.BilinearForm(self.fes, symmetric=True)
        self.stiffness_matrix += self.problem.alpha * ng.grad(u) * ng.grad(v) * ng.dx
        self.stiffness_matrix.Assemble()

        self.mass_matrix = ng.BilinearForm(self.fes, symmetric=True)
        self.mass_matrix += u * v * ng.dx
        self.mass_matrix.Assemble()

        # Store full matrices
        self.stiffness_matrix_mat = self._ngsolve_to_torch(self.stiffness_matrix.mat)
        self.mass_matrix_mat = self._ngsolve_to_torch(self.mass_matrix.mat)
        
        # Extract free DOFs matrices for efficient residual computation
        self._init_free_dofs_matrices()

    def _init_free_dofs_matrices(self):
        """Extract submatrices corresponding to free DOFs only for efficient residual computation."""
        # Get free DOFs mask
        free_dofs_bitarray = self.fes.FreeDofs()
        free_dofs_indices = [i for i in range(len(free_dofs_bitarray)) if free_dofs_bitarray[i]]
        
        # Convert to numpy for easier indexing
        self.free_dofs_indices = np.array(free_dofs_indices)
        self.n_free_dofs = len(free_dofs_indices)
        
        # Extract submatrices for free DOFs only
        # Convert to scipy sparse first for easier submatrix extraction
        stiffness_scipy = self._ngsolve_to_scipy(self.stiffness_matrix.mat)
        mass_scipy = self._ngsolve_to_scipy(self.mass_matrix.mat)
        
        # Extract submatrices: A_ff (free-to-free block)
        stiffness_free = stiffness_scipy[np.ix_(free_dofs_indices, free_dofs_indices)]
        mass_free = mass_scipy[np.ix_(free_dofs_indices, free_dofs_indices)]
        
        # Convert back to torch sparse tensors
        self.stiffness_matrix_free = self._scipy_to_torch(stiffness_free)
        self.mass_matrix_free = self._scipy_to_torch(mass_free)

    def _scipy_to_torch(self, scipy_matrix):
        """Convert scipy sparse matrix to torch sparse tensor."""
        coo = scipy_matrix.tocoo()
        indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float64)
        shape = coo.shape
        return torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float64)

    def _ngsolve_to_scipy(self, ngsolve_matrix):
        """Convert NGSolve matrix to scipy sparse matrix."""
        if hasattr(ngsolve_matrix, "COO"):
            rows, cols, vals = ngsolve_matrix.COO()
            return sp.csr_matrix(
                (vals, (rows, cols)),
                shape=(ngsolve_matrix.height, ngsolve_matrix.width),
            )
        else:
            # Fallback to dense conversion
            dense = np.array(ngsolve_matrix.ToDenseMatrix(), dtype=np.float64)
            return sp.csr_matrix(dense)

    def _ngsolve_to_torch(self, ngsolve_matrix):
        """Convert NGSolve matrix to torch tensor."""
        rows, cols, vals = ngsolve_matrix.COO()
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float64)
        shape = (ngsolve_matrix.height, ngsolve_matrix.width)
        return torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float64)

    def solve_steady_state(self):
        if self.problem is None:
            raise ValueError("No problem defined for FEMSolver.")
        # Create grid function for solution
        fes = ng.H1(self.mesh, order=self.order, dirichlet="left|right|top|bottom")

        # Trial and test functions
        u = fes.TrialFunction()
        v = fes.TestFunction()

        # Create grid function for solution
        gfu = ng.GridFunction(fes)

        boundary_cf = self.mesh.BoundaryCF(self.problem.boundary_values, default=0)
        gfu.Set(boundary_cf, ng.BND)

        print(f"Boundary conditions set: {self.problem.boundary_values}")

        # Assemble stiffness matrix
        a = ng.BilinearForm(fes, symmetric=True)
        a += self.problem.alpha * ng.grad(u) * ng.grad(v) * ng.dx
        a.Assemble()

        # Assemble RHS (zero for no source)
        f = ng.LinearForm(fes)
        source_function = self.problem.source_function
        if source_function is None:
            # Homogeneous case: f = 0 - no terms added to linear form
            pass
        else:
            gfu_source = ng.GridFunction(fes)
            gfu_source.vec.FV().NumPy()[:] = source_function
            f += gfu_source * v * ng.dx
        f.Assemble()

        # The key correction: We need to modify the RHS to account for non-zero Dirichlet BC
        # The equation is: K * u = f - K * u_D
        # where u_D contains the Dirichlet values

        # Create a temporary grid function with boundary values
        gfu_bc = ng.GridFunction(fes)
        gfu_bc.Set(boundary_cf, ng.BND)

        # Modify RHS: f_modified = f - K * u_D
        f.vec.data -= a.mat * gfu_bc.vec

        # Now solve for the correction: K * u_corr = f_modified
        # The solution will be u_total = u_corr + u_D
        gfu_correction = ng.GridFunction(fes)
        gfu_correction.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

        # Add the correction to the boundary values
        gfu.vec.data += gfu_correction.vec

        return gfu.vec.FV().NumPy().copy()

    def solve_transient_problem(self, problem: MeshProblem) -> List[np.ndarray]:

        dt = problem.time_config.dt
        t_final = problem.time_config.t_final

        states = []
        u, v = self.fes.TnT()
        mform = u * v * ng.dx
        aform = ng.grad(u) * ng.grad(v) * ng.dx

        m = ng.BilinearForm(mform).Assemble()
        a = ng.BilinearForm(aform).Assemble()
        mstar = ng.BilinearForm(mform + dt * aform).Assemble()
        mstarinv = mstar.mat.Inverse(freedofs=self.fes.FreeDofs())
        f = ng.LinearForm(self.fes).Assemble()

        gfu = ng.GridFunction(self.fes)
        gfu.vec.FV().NumPy()[:] = problem.initial_condition
        states.append(gfu.vec.FV().NumPy().copy())
        for j in range(int(t_final / dt)):
            res = f.vec - a.mat * gfu.vec
            w = mstarinv * res
            gfu.vec.data += dt * w
            states.append(gfu.vec.FV().NumPy().copy())
        return states

    def compute_residual(
        self, t_pred_next: torch.Tensor, t_prev: torch.Tensor, problem: MeshProblem
    ):
        """
        Compute the FEM residual for physics-informed training.

        R(T_pred_next, T_prev) = (1/dt*M + K)*T_pred_next - (1/dt*M)*T_prev - Q(T_pred_next) - g

        This is the left-hand side of the discretized weak form that should equal zero
        for an exact solution. Following the paper's Eq. (9), we only evaluate the residual
        on free DOFs (not constrained by Dirichlet boundary conditions).
        """
        # Set problem if not already set
        if self.problem is None:
            self.problem = problem
            self.init_matrices()

        dt = problem.time_config.dt
        # Ensure tensors are on the same device and dtype
        device = t_pred_next.device

        free_dofs_bitarray = self.fes.FreeDofs()

        # Convert BitArray to torch boolean mask
        free_dofs_mask = torch.tensor(
            [free_dofs_bitarray[i] for i in range(len(free_dofs_bitarray))],
            dtype=torch.bool,
            device=device,
        )

        t_pred_next[free_dofs_mask == False] = 0.0  # Zero out boundary DOFs

        self.stiffness_matrix_mat = self.stiffness_matrix_mat.to(device)
        self.mass_matrix_mat = self.mass_matrix_mat.to(device)

        # Convert sparse tensors to float32 to match input precision
        mass_mat = self.mass_matrix_mat.to(dtype=t_pred_next.dtype)
        stiff_mat = self.stiffness_matrix_mat.to(dtype=t_pred_next.dtype)

        dt_mass_mat = mass_mat / dt  # (1/dt)*M
        mass_plus_stiff = dt_mass_mat + stiff_mat  # (1/dt*M + K)

        # Source term Q(T_pred_next) + g
        if problem.source_function is None:
            source_term = torch.zeros_like(
                t_pred_next, device=device, dtype=t_pred_next.dtype
            )
        else:
            raise NotImplementedError(
                "Non-zero source terms not implemented in residual computation."
            )

        t_pred_term = torch.sparse.mm(
            mass_plus_stiff, t_pred_next.unsqueeze(1)
        ).squeeze()  # [N_dofs]
        t_prev_term = torch.sparse.mm(
            dt_mass_mat, t_prev.unsqueeze(1)
        ).squeeze()  # [N_dofs]

        # Residual calculation: should be zero for exact solution
        # residual = mass_plus_stiff * t_pred_next - (1/dt)*mass_mat * t_prev - source_term
        residual = torch.add(t_pred_term, -t_prev_term)
        residual = residual.squeeze()  # [N_dofs]

        # Apply free DOFs mask - only compute loss on free DOFs (following Eq. 9 in paper)
        # NGSolve's FreeDofs() returns a BitArray indicating which DOFs are free

        # Mask residual to only include free DOFs
        residual_free = residual[free_dofs_mask]

        return residual_free

    def compute_residual_free_dofs_only(
        self, t_pred_next_free: torch.Tensor, t_prev: torch.Tensor, problem: MeshProblem, current_time: int = 0
    ):
        """
        Compute the FEM residual for physics-informed training using only free DOFs.
        
        This is more efficient as it works directly with reduced matrices and vectors.
        
        Args:
            t_pred_next_free: Predicted solution at next time step for FREE DOFs only [N_free_dofs]
            t_prev: Previous solution at all DOFs [N_dofs] 
            problem: MeshProblem instance
            
        Returns:
            residual_free: Residual computed only on free DOFs [N_free_dofs]
        """
        # Set problem if not already set
        if self.problem is None:
            self.problem = problem
            self.init_matrices()

        dt = problem.time_config.dt

        # Ensure tensors are on the same device and dtype
        device = t_pred_next_free.device
        dtype = t_pred_next_free.dtype
        
        # Move free DOFs matrices to correct device and dtype
        mass_mat_free = self.mass_matrix_free.to(device=device, dtype=dtype)
        stiff_mat_free = self.stiffness_matrix_free.to(device=device, dtype=dtype)

        # Extract previous solution for free DOFs only
        free_dofs_mask = torch.tensor(
            [self.fes.FreeDofs()[i] for i in range(len(self.fes.FreeDofs()))],
            dtype=torch.bool,
            device=device,
        )
        t_prev_free = t_prev[free_dofs_mask].to(dtype=dtype)

        # Compute residual components
        dt_mass_mat_free = mass_mat_free / dt  # (1/dt)*M_ff
        mass_plus_stiff_free = dt_mass_mat_free + stiff_mat_free  # (1/dt*M_ff + K_ff)

        # Matrix-vector multiplications (only on free DOFs)
        t_pred_term = torch.sparse.mm(
            mass_plus_stiff_free, t_pred_next_free.unsqueeze(1)
        ).squeeze()  # [N_free_dofs]
        
        t_prev_term = torch.sparse.mm(
            dt_mass_mat_free, t_prev_free.unsqueeze(1)
        ).squeeze()  # [N_free_dofs]

        # Residual calculation: should be zero for exact solution
        # residual = (1/dt*M_ff + K_ff) * t_pred_next_free - (1/dt)*M_ff * t_prev_free
        residual_free = t_pred_term - t_prev_term

        # # normalize to max absolute value
        # max_abs = torch.max(torch.abs(residual_free))
        # if max_abs > 0:
        #     residual_free = residual_free / max_abs

        # Source term (if needed)
        if problem.source_function is not None:
            raise NotImplementedError(
                "Non-zero source terms not implemented in free DOFs residual computation."
            )

        return residual_free.squeeze()

    def extract_free_dofs(self, full_solution: torch.Tensor) -> torch.Tensor:
        """
        Extract free DOFs from a full solution vector.
        
        Args:
            full_solution: Solution vector for all DOFs [N_dofs]
            
        Returns:
            free_solution: Solution vector for free DOFs only [N_free_dofs]
        """
        if not hasattr(self, 'free_dofs_indices'):
            self._init_free_dofs_matrices()
            
        device = full_solution.device
        free_dofs_mask = torch.tensor(
            [self.fes.FreeDofs()[i] for i in range(len(self.fes.FreeDofs()))],
            dtype=torch.bool,
            device=device,
        )
        return full_solution[free_dofs_mask]
    
    def expand_to_full_dofs(self, free_solution: torch.Tensor, boundary_values: torch.Tensor = None) -> torch.Tensor:
        """
        Expand a free DOFs solution to a full DOFs solution by inserting boundary values.
        
        Args:
            free_solution: Solution vector for free DOFs only [N_free_dofs]
            boundary_values: Values for boundary DOFs [N_boundary_dofs]. If None, uses zero.
            
        Returns:
            full_solution: Solution vector for all DOFs [N_dofs]
        """
        if not hasattr(self, 'free_dofs_indices'):
            self._init_free_dofs_matrices()
            
        device = free_solution.device
        dtype = free_solution.dtype
        
        # Create full solution vector
        n_total_dofs = len(self.fes.FreeDofs())
        full_solution = torch.zeros(n_total_dofs, device=device, dtype=dtype)
        
        # Get free DOFs mask
        free_dofs_mask = torch.tensor(
            [self.fes.FreeDofs()[i] for i in range(len(self.fes.FreeDofs()))],
            dtype=torch.bool,
            device=device,
        )
        
        # Insert free DOFs values
        full_solution[free_dofs_mask] = free_solution
        
        # Insert boundary values if provided
        if boundary_values is not None:
            boundary_mask = ~free_dofs_mask
            full_solution[boundary_mask] = boundary_values
            
        return full_solution

    def export_to_vtk(
        self, array_true, array_pred, time_steps, filename="results/vtk/results.vtk"
    ):
        """
        Export solutions to VTK file for visualization in Paraview.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        gfu_true = ng.GridFunction(self.fes)
        gfu_pred = ng.GridFunction(self.fes)
        gfu_diff = ng.GridFunction(self.fes)
        gfu_true.vec.FV().NumPy()[:] = array_true[0]
        gfu_pred.vec.FV().NumPy()[:] = array_pred[0]
        # relative error
        gfu_diff.vec.FV().NumPy()[:] = (array_true[0] - array_pred[0]) / (
            np.max(np.abs(array_true[0])) - np.min(np.abs(array_true[0]))
        )
        vtk_out = ng.VTKOutput(
            self.mesh,
            coefs=[gfu_true, gfu_pred, gfu_diff],
            names=["ExactSolution", "PredictedSolution", "Difference"],
            filename=str(filename),
        )
        for idx, time in enumerate(time_steps):
            gfu_true.vec.FV().NumPy()[:] = array_true[idx]
            gfu_pred.vec.FV().NumPy()[:] = array_pred[idx]
            gfu_diff.vec.FV().NumPy()[:] = (
                (array_true[idx] - array_pred[idx])
                / (np.max(np.abs(array_true[idx])) - np.min(np.abs(array_true[idx])))
                * 100
            )
            vtk_out.Do(time=time)
        print(f"VTK file saved as {filename}")


if __name__ == "__main__":

    import ngsolve as ng
    from mesh_utils import build_graph_from_mesh, create_gaussian_initial_condition
    from containers import MeshProblem
    from graph_creator import GraphCreator

    # Create a simple mesh
    mesh = create_rectangular_mesh(
        width=1,
        height=1,
        maxh=0.3,
    )

    # Convert mesh to graph data
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=["left", "right", "top", "bottom"],
        neumann_names=[],
        connectivity_method="fem",
    )
    graph_data, aux = graph_creator.create_graph()

    # Define problem parameters
    alpha = 1.0  # Diffusion coefficient
    x = graph_data["pos"][:, 0]
    y = graph_data["pos"][:, 1]
    initial_condition = create_gaussian_initial_condition(
        pos=graph_data["pos"],
        num_gaussians=1,
        amplitude_range=(0.5, 1.0),
        sigma_fraction_range=(0.1, 0.2),
        seed=42,
        centered=True,
        enforce_boundary_conditions=True,
    )

    graph_data, aux = graph_creator.create_graph(
        T_current=initial_condition, t_scalar=0.0
    )

    free_graph, node_mapping, new_aux = graph_creator.create_free_node_subgraph(
        graph_data, aux
    )

    mesh_config = MeshConfig(
        maxh=0.1,
        order=1,
        dim=2,
        dirichlet_boundaries=["left", "right", "top", "bottom"],
        mesh_type="rectangle",
    )

    # Create a MeshProblem instance
    problem = MeshProblem(
        mesh,
        graph_data,
        initial_condition,
        alpha,
        time_config=TimeConfig(dt=0.1, t_final=1.0),
        mesh_config=mesh_config,
        problem_id=0,
    )

    # Initialize FEM solver
    fem_solver = FEMSolver(mesh, order=1, problem=problem)

    # Solve steady-state problem
    transient_solution = fem_solver.solve_transient_problem(problem)

    # test residual computation
    t_prev = torch.tensor(transient_solution[2], dtype=torch.float64)
    t_pred_next = torch.tensor(transient_solution[3], dtype=torch.float64)
    
    # Test original residual computation (all DOFs then mask)
    residual_original = fem_solver.compute_residual(t_pred_next, t_prev, problem)
    print(f"Original residual (mean): {np.mean(residual_original.numpy()):.2e}")
    assert np.mean(residual_original.numpy()) < 1e-8, "Original residual is too high!"
    
    # Test new free DOFs only residual computation
    t_pred_next_free = fem_solver.extract_free_dofs(t_pred_next)
    residual_free_only = fem_solver.compute_residual_free_dofs_only(t_pred_next_free, t_prev, problem)
    print(f"Free DOFs only residual (mean): {np.mean(residual_free_only.numpy()):.2e}")
    assert np.mean(residual_free_only.numpy()) < 1e-8, "Free DOFs only residual is too high!"
    
    # Verify both methods give the same result
    residual_diff = torch.norm(residual_original - residual_free_only)
    print(f"Difference between methods: {residual_diff.item():.2e}")
    assert residual_diff < 1e-12, "Residual methods should give identical results!"
    
    print("All residual computation tests passed!")
    
    # Demonstrate efficiency gain
    print(f"Total DOFs: {len(t_pred_next)}")
    print(f"Free DOFs: {len(t_pred_next_free)}")
    print(f"Efficiency gain: {len(t_pred_next) / len(t_pred_next_free):.2f}x fewer operations")

    fem_solver.export_to_vtk(
        np.array(transient_solution),
        np.array(transient_solution),
        problem.time_config.time_steps,
        filename="results/vtk/transient_solution.vtk",
    )
