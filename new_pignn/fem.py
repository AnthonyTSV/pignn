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

        self.stiffness_matrix_mat = self._ngsolve_to_torch(self.stiffness_matrix.mat)

        self.mass_matrix = ng.BilinearForm(self.fes, symmetric=True)
        self.mass_matrix += u * v * ng.dx
        self.mass_matrix.Assemble()

        self.mass_matrix_mat = self._ngsolve_to_torch(self.mass_matrix.mat)

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
        u,v = self.fes.TnT()
        mform = u*v*ng.dx
        aform = ng.grad(u)*ng.grad(v)*ng.dx

        m = ng.BilinearForm(mform).Assemble()
        a = ng.BilinearForm(aform).Assemble()
        mstar = ng.BilinearForm(mform+dt*aform).Assemble()
        mstarinv = mstar.mat.Inverse(freedofs=self.fes.FreeDofs())
        f = ng.LinearForm(self.fes).Assemble()

        gfu = ng.GridFunction(self.fes)
        gfu.vec.FV().NumPy()[:] = problem.initial_condition
        states.append(gfu.vec.FV().NumPy().copy())
        for j in range(int(t_final/dt)):
            res = f.vec - a.mat * gfu.vec
            w = mstarinv * res
            gfu.vec.data += dt*w
            states.append(gfu.vec.FV().NumPy().copy())
        return states

    def compute_residual(self, t_pred_next: torch.Tensor, t_prev: torch.Tensor, problem: MeshProblem):
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
        self.stiffness_matrix_mat = self.stiffness_matrix_mat.to(device)
        self.mass_matrix_mat = self.mass_matrix_mat.to(device)
        
        # Convert sparse tensors to float32 to match input precision
        mass_mat = self.mass_matrix_mat.to(dtype=t_pred_next.dtype)
        stiff_mat = self.stiffness_matrix_mat.to(dtype=t_pred_next.dtype)
        
        dt_mass_mat = mass_mat / dt  # (1/dt)*M
        mass_plus_stiff = dt_mass_mat + stiff_mat  # (1/dt*M + K)

        # Source term Q(T_pred_next) + g
        if problem.source_function is None:
            source_term = torch.zeros_like(t_pred_next, device=device, dtype=t_pred_next.dtype)
        else:
            raise NotImplementedError("Non-zero source terms not implemented in residual computation.")
        
        t_pred_term = torch.sparse.mm(mass_plus_stiff, t_pred_next.unsqueeze(1)).squeeze()  # [N_dofs]
        t_prev_term = torch.sparse.mm(dt_mass_mat, t_prev.unsqueeze(1)).squeeze()  # [N_dofs]

        # Residual calculation: should be zero for exact solution
        # residual = mass_plus_stiff * t_pred_next - (1/dt)*mass_mat * t_prev - source_term
        residual = torch.add(t_pred_term, -t_prev_term)
        residual = residual.squeeze()  # [N_dofs]
        
        # Apply free DOFs mask - only compute loss on free DOFs (following Eq. 9 in paper)
        # NGSolve's FreeDofs() returns a BitArray indicating which DOFs are free
        free_dofs_bitarray = self.fes.FreeDofs()
        
        # Convert BitArray to torch boolean mask
        free_dofs_mask = torch.tensor([free_dofs_bitarray[i] for i in range(len(free_dofs_bitarray))], 
                                     dtype=torch.bool, device=device)
        
        # Mask residual to only include free DOFs
        residual_free = residual[free_dofs_mask]

        return residual_free
    
    def export_to_vtk(self, array_true, array_pred, time_steps, filename="results/vtk/results.vtk"):
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
        gfu_diff.vec.FV().NumPy()[:] = (array_true[0] - array_pred[0]) / (np.max(np.abs(array_true[0])) - np.min(np.abs(array_true[0])))
        vtk_out = ng.VTKOutput(self.mesh, coefs=[gfu_true, gfu_pred, gfu_diff], names=["ExactSolution", "PredictedSolution", "Difference"], filename=str(filename))
        for idx, time in enumerate(time_steps):
            gfu_true.vec.FV().NumPy()[:] = array_true[idx]
            gfu_pred.vec.FV().NumPy()[:] = array_pred[idx]
            gfu_diff.vec.FV().NumPy()[:] = (array_true[idx] - array_pred[idx]) / (np.max(np.abs(array_true[idx])) - np.min(np.abs(array_true[idx]))) * 100
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
    graph_creator = GraphCreator(mesh=mesh)
    graph_data, aux = graph_creator.create_graph()

    # Define problem parameters
    alpha = 1.0  # Diffusion coefficient
    x = graph_data['pos'][:,0]
    y = graph_data['pos'][:,1]
    initial_condition = create_gaussian_initial_condition(
        pos=graph_data['pos'],
        num_gaussians=1,
        amplitude_range=(0.5, 1.0),
        sigma_fraction_range=(0.1, 0.2),
        seed=42,
        centered=True,
        enforce_boundary_conditions=True
    )

    graph_data, aux = graph_creator.create_graph(T_current=initial_condition, t_scalar=0.0)

    mesh_config = MeshConfig(
        maxh=0.1,
        order=1,
        dim=2,
        dirichlet_boundaries=["left", "right", "top", "bottom"],
        mesh_type="rectangle"
    )

    # Create a MeshProblem instance
    problem = MeshProblem(mesh, graph_data, initial_condition, alpha, time_config=TimeConfig(dt=0.1, t_final=1.0), mesh_config=mesh_config, problem_id=0)
    
    # Initialize FEM solver
    fem_solver = FEMSolver(mesh, order=1, problem=problem)
    
    # Solve steady-state problem
    transient_solution = fem_solver.solve_transient_problem(problem)

    # test residual computation
    t_prev = torch.tensor(transient_solution[2], dtype=torch.float64)
    t_pred_next = torch.tensor(transient_solution[3], dtype=torch.float64)
    residual = fem_solver.compute_residual(t_pred_next, t_prev, problem)
    assert np.mean(residual.numpy()) < 1e-8, "Residual is too high!"
    print("Residual computation test passed!")

    fem_solver.export_to_vtk(np.array(transient_solution), np.array(transient_solution), problem.time_config.time_steps, filename="results/vtk/transient_solution.vtk")