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
        
        # Set up finite element space with proper Dirichlet boundaries
        if problem is not None and problem.mesh_config.dirichlet_pipe:
            dirichlet_string = problem.mesh_config.dirichlet_pipe
        else:
            # Default fallback - use all boundaries as Dirichlet
            dirichlet_string = "left|right|top|bottom"
            
        self.fes = ng.H1(
            mesh,
            order=order,
            dirichlet=dirichlet_string
        )
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

        self.neumann_matrix = ng.LinearForm(self.fes)
        for name, value in self.problem.neumann_values.items():
            self.neumann_matrix += value * v * ng.ds(definedon=name)
        self.neumann_matrix.Assemble()

        # Store full matrices
        self.stiffness_matrix_mat = self._ngsolve_to_torch(self.stiffness_matrix.mat)
        self.mass_matrix_mat = self._ngsolve_to_torch(self.mass_matrix.mat)
        self.neumann_vec = torch.tensor(self.neumann_matrix.vec.FV().NumPy().copy(), dtype=torch.float64)

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
        for name, value in problem.neumann_values.items():
            f += value * v * ng.ds(definedon=name)
        f.Assemble()

        gfu = ng.GridFunction(self.fes)
        gfu_initial = ng.GridFunction(self.fes)

        # Set initial condition on the interior
        gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

        # Set Dirichlet boundary conditions
        boundary_cf = self.mesh.BoundaryCF(problem.boundary_values, default=0)
        gfu.Set(boundary_cf, definedon=self.mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

        # Copy initial condition values for free DOFs only
        free_dofs = self.fes.FreeDofs()
        for dof in range(self.fes.ndof):
            if free_dofs[dof]:
                gfu.vec[dof] = gfu_initial.vec[dof]
        
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
        Compute the FEM residual for physics-informed training with inhomogeneous Dirichlet BCs.

        For the transient heat equation with inhomogeneous Dirichlet boundary conditions:
        The discrete system is: (1/dt*M + K)*u_n+1 = (1/dt*M)*u_n + f + g_N
        
        For free DOFs, the residual is:
        R_free = (1/dt*M + K)_free,free * u_free_n+1 + (1/dt*M + K)_free,bc * u_bc - (1/dt*M)_free,free * u_free_n - (1/dt*M)_free,bc * u_bc_n - f_free - g_N_free
        
        But since we're using predictions for the entire solution, we compute the residual differently:
        R_free = [(1/dt*M + K) * u_pred_full - (1/dt*M) * u_prev_full - f - g_N]_free
        
        The residual should be zero for an exact solution on free DOFs only.
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

        # Create full solution vector including boundary values
        t_pred_full = t_pred_next.clone()
        t_prev_full = t_prev.clone()
        
        # Set Dirichlet boundary values from the problem in both prediction and previous state
        boundary_dofs_vector = self._create_boundary_dofs_vector(problem, device, t_pred_next.dtype)
        t_pred_full[~free_dofs_mask] = boundary_dofs_vector[~free_dofs_mask]
        t_prev_full[~free_dofs_mask] = boundary_dofs_vector[~free_dofs_mask]

        self.stiffness_matrix_mat = self.stiffness_matrix_mat.to(device)
        self.mass_matrix_mat = self.mass_matrix_mat.to(device)

        # Convert sparse tensors to match input precision
        mass_mat = self.mass_matrix_mat.to(dtype=t_pred_next.dtype)
        stiff_mat = self.stiffness_matrix_mat.to(dtype=t_pred_next.dtype)

        dt_mass_mat = mass_mat / dt  # (1/dt)*M
        mass_plus_stiff = dt_mass_mat + stiff_mat  # (1/dt*M + K)

        self.neumann_vec = self.neumann_vec.to(device)
        neumann_vec = self.neumann_vec.to(dtype=t_pred_next.dtype)

        # Source term Q(T_pred_next) + g
        if problem.source_function is None:
            source_term = torch.zeros_like(
                t_pred_full, device=device, dtype=t_pred_next.dtype
            )
        else:
            raise NotImplementedError(
                "Non-zero source terms not implemented in residual computation."
            )

        # Compute matrix-vector products for the full vectors
        t_pred_term = torch.sparse.mm(
            mass_plus_stiff, t_pred_full.unsqueeze(1)
        ).squeeze()  # [N_dofs]
        t_prev_term = torch.sparse.mm(
            dt_mass_mat, t_prev_full.unsqueeze(1)
        ).squeeze()  # [N_dofs]

        # Residual calculation for the full system:
        # R = (1/dt*M + K)*T_pred_full - (1/dt*M)*T_prev_full - source_term - neumann_vec
        residual = torch.add(t_pred_term, -t_prev_term)
        residual = torch.add(residual, -source_term)
        residual = torch.add(residual, -neumann_vec)
        residual = residual.squeeze()  # [N_dofs]

        # Apply free DOFs mask - only compute loss on free DOFs (following Eq. 9 in paper)
        # The residual on boundary DOFs should be zero by construction since we enforce the BC
        residual_free = residual[free_dofs_mask]

        return residual_free

    def _create_boundary_dofs_vector(self, problem: MeshProblem, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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
        if hasattr(problem, 'boundary_values') and problem.boundary_values:
            # Create a grid function to set boundary values
            gfu_boundary = ng.GridFunction(self.fes)
            
            # Set boundary values using NGSolve's BoundaryCF
            try:
                boundary_cf = self.mesh.BoundaryCF(problem.boundary_values, default=0)
                gfu_boundary.Set(boundary_cf, ng.BND)
                
                # Extract the boundary values as a numpy array and convert to torch
                boundary_values_np = gfu_boundary.vec.FV().NumPy().copy()
                boundary_vector = torch.tensor(boundary_values_np, device=device, dtype=dtype)
                
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
        maxh=0.1,
    )

    dirichlet_boundaries = ["bottom", "right", "top", "left"]
    neumann_boundaries = []
    dirichlet_boundaries_dict = {"bottom": 0, "right": 0, "top": 0, "left": 0}
    neumann_boundaries_dict = {}

    # Convert mesh to graph data
    graph_creator = GraphCreator(
        mesh=mesh,
        n_neighbors=2,
        dirichlet_names=dirichlet_boundaries,
        neumann_names=neumann_boundaries,
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
    # initial_condition = np.zeros_like(initial_condition)

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
        dirichlet_boundaries=dirichlet_boundaries,
        neumann_boundaries=neumann_boundaries,
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
    problem.set_neumann_values(neumann_boundaries_dict)
    problem.set_dirichlet_values(dirichlet_boundaries_dict)
    problem.set_source_function(None)

    # Initialize FEM solver
    fem_solver = FEMSolver(mesh, order=1, problem=problem)

    # Solve transient problem
    transient_solution = fem_solver.solve_transient_problem(problem)

    # Test residual computation with inhomogeneous Dirichlet BCs
    t_prev = torch.tensor(transient_solution[2], dtype=torch.float64)
    t_pred_next = torch.tensor(transient_solution[3], dtype=torch.float64)

    residual = fem_solver.compute_residual(t_pred_next, t_prev, problem)
    print(f"Residual (mean): {np.mean(residual.numpy()):.2e}")
    assert np.mean(residual.numpy()) < 1e-8, "Residual is too high!"

    fem_solver.export_to_vtk(
        np.array(transient_solution),
        np.array(transient_solution),
        problem.time_config.time_steps,
        filename="results/vtk/transient_solution.vtk",
    )

