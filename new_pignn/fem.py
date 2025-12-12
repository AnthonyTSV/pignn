import numpy as np
from containers import MeshConfig, MeshProblem, TimeConfig
import ngsolve as ng
from typing import Optional, List, Tuple
import torch
import scipy.sparse as sp
import os
from pathlib import Path
from mesh_utils import create_rectangular_mesh, build_graph_from_mesh


class FEMSolver:
    def __init__(self, mesh: ng.Mesh, order=1, problem: Optional[MeshProblem] = None):
        self.mesh = mesh
        self.order = order
        
        # Set up finite element space with proper Dirichlet boundaries
        dirichlet_string = None
        if problem is not None and problem.mesh_config.dirichlet_pipe:
            dirichlet_string = problem.mesh_config.dirichlet_pipe
            
        # Create H1 space - only pass dirichlet if we have boundaries
        if dirichlet_string:
            self.fes = ng.H1(
                mesh,
                order=order,
                dirichlet=dirichlet_string
            )
        else:
            # No Dirichlet boundaries - all DOFs are free
            self.fes = ng.H1(
                mesh,
                order=order
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
        self.stiffness_matrix += self._alpha_weight() * ng.grad(u) * ng.grad(v) * ng.dx
        
        # Add Robin BC contribution to stiffness matrix: + integral(h * u * v) ds
        if hasattr(self.problem, 'robin_values'):
            for name, (h, _) in self.problem.robin_values.items():
                self.stiffness_matrix += h * u * v * ng.ds(definedon=name)
        
        self.stiffness_matrix.Assemble()

        self.mass_matrix = ng.BilinearForm(self.fes, symmetric=True)
        self.mass_matrix += u * v * ng.dx
        self.mass_matrix.Assemble()

        self.neumann_matrix = ng.LinearForm(self.fes)
        for name, value in self.problem.neumann_values.items():
            self.neumann_matrix += value * v * ng.ds(definedon=name)
            
        # Add Robin BC contribution to RHS: + integral(h * T_amb * v) ds
        if hasattr(self.problem, 'robin_values'):
            for name, (h, t_amb) in self.problem.robin_values.items():
                self.neumann_matrix += h * t_amb * v * ng.ds(definedon=name)
                
        self.neumann_matrix.Assemble()

        # Store full matrices
        self.stiffness_matrix_mat = self._ngsolve_to_torch(self.stiffness_matrix.mat)
        self.mass_matrix_mat = self._ngsolve_to_torch(self.mass_matrix.mat)
        self.neumann_vec = torch.tensor(self.neumann_matrix.vec.FV().NumPy().copy(), dtype=torch.float64)

    def _alpha_weight(self):
        if self.problem is None:
            return 1.0
        alpha_cf = getattr(self.problem, "alpha_coefficient", None)
        if alpha_cf is not None:
            return alpha_cf
        return getattr(self.problem, "alpha", 1.0)

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

    def solve_transient_problem(self, problem: MeshProblem) -> List[np.ndarray]:

        dt = problem.time_config.dt
        t_final = problem.time_config.t_final

        states = []
        u, v = self.fes.TnT()
        mform = u * v * ng.dx
        aform = self._alpha_weight() * ng.grad(u) * ng.grad(v) * ng.dx
        
        # Add Robin BC contribution to stiffness matrix
        if hasattr(problem, 'robin_values'):
            for name, (h, _) in problem.robin_values.items():
                aform += h * u * v * ng.ds(definedon=name)

        m = ng.BilinearForm(mform).Assemble()
        a = ng.BilinearForm(aform).Assemble()
        mstar = ng.BilinearForm(mform + dt * aform).Assemble()
        mstarinv = mstar.mat.Inverse(freedofs=self.fes.FreeDofs())

        f = ng.LinearForm(self.fes)
        for name, value in problem.neumann_values.items():
            f += value * v * ng.ds(definedon=name)
            
        # Add Robin BC contribution to RHS
        if hasattr(problem, 'robin_values'):
            for name, (h, t_amb) in problem.robin_values.items():
                f += h * t_amb * v * ng.ds(definedon=name)
                
        # add source term if provided
        if problem.source_function is not None:
            gfu_source = ng.GridFunction(self.fes)
            gfu_source.vec.FV().NumPy()[:] = problem.source_function
            f += gfu_source * v * ng.dx
        f.Assemble()
        base_force_vec = f.vec.CreateVector()
        base_force_vec.data = f.vec

        gfu = ng.GridFunction(self.fes)
        gfu_initial = ng.GridFunction(self.fes)

        # Set initial condition on the interior
        gfu_initial.vec.FV().NumPy()[:] = problem.initial_condition

        # Set Dirichlet boundary conditions (only if we have Dirichlet boundaries)
        boundary_cf = None
        has_dirichlet = problem.mesh_config.dirichlet_pipe and problem.boundary_values
        if has_dirichlet:
            boundary_cf = self.mesh.BoundaryCF(problem.boundary_values, default=0)
            gfu.Set(boundary_cf, definedon=self.mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

        # Copy initial condition values for free DOFs only
        free_dofs = self.fes.FreeDofs()
        for dof in range(self.fes.ndof):
            if free_dofs[dof]:
                gfu.vec[dof] = gfu_initial.vec[dof]
        
        states.append(gfu.vec.FV().NumPy().copy())
        for j in range(int(t_final / dt)):
            current_time = (j + 1) * dt

            total_force = base_force_vec.CreateVector()
            total_force.data = base_force_vec

            nonlinear_source = self._compute_nonlinear_source_vector(
                problem,
                temperature_values=gfu.vec.FV().NumPy(),
                current_time=current_time,
            )
            if nonlinear_source is not None:
                total_force.FV().NumPy()[:] += nonlinear_source

            rhs_vec = total_force.CreateVector()
            rhs_vec.data = m.mat * gfu.vec
            rhs_vec.data += dt * total_force

            gfu_next = ng.GridFunction(self.fes)
            
            # Only set Dirichlet BCs if we have them
            if has_dirichlet and boundary_cf is not None:
                gfu_next.Set(boundary_cf, definedon=self.mesh.Boundaries(problem.mesh_config.dirichlet_pipe))

            rhs_vec.data -= mstar.mat * gfu_next.vec
            gfu_next.vec.data += mstarinv * rhs_vec

            gfu = gfu_next
            states.append(gfu.vec.FV().NumPy().copy())
        return states

    def compute_residual(
        self,
        t_pred_next: torch.Tensor,
        t_prev: torch.Tensor,
        problem: MeshProblem,
        time_scalar: Optional[float] = None,
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

        source_term = self._compute_source_term(
            problem=problem,
            temperature_field=t_pred_full,
            time_scalar=time_scalar,
            mass_mat=mass_mat,
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

        return residual_free ** 2

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

    def _compute_source_term(
        self,
        problem: MeshProblem,
        temperature_field: torch.Tensor,
        time_scalar: Optional[float],
        mass_mat: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the source term, supporting nonlinear heating."""
        device = temperature_field.device
        dtype = temperature_field.dtype

        if problem.source_function is not None:
            if isinstance(problem.source_function, torch.Tensor):
                source_nodal = problem.source_function.detach().clone().to(device=device, dtype=dtype)
            else:
                source_nodal = torch.tensor(problem.source_function, device=device, dtype=dtype)
            return torch.sparse.mm(mass_mat, source_nodal.unsqueeze(1)).squeeze()

        return torch.zeros_like(temperature_field, device=device, dtype=dtype)

    def _compute_nonlinear_source_vector(
        self,
        problem: MeshProblem,
        temperature_values: np.ndarray,
        current_time: float,
    ) -> Optional[np.ndarray]:
        params = getattr(problem, "nonlinear_source_params", None)
        material_fraction = getattr(problem, "material_fraction_field", None)
        if not params or material_fraction is None:
            return None

        if self.mass_matrix_mat is None:
            raise ValueError("Mass matrix not initialized for nonlinear source computation")

        device = self.mass_matrix_mat.device
        dtype = self.mass_matrix_mat.dtype
        temp_tensor = torch.tensor(temperature_values, dtype=dtype, device=device)
        vf_tensor = torch.tensor(material_fraction, dtype=dtype, device=device)

        q0 = params.get("q0", 0.0)
        t0 = max(params.get("t0", 1.0), 1e-8)
        C = params.get("C", 0.0)
        q_tilde = q0 * (1.0 - vf_tensor)
        exponent = torch.exp(-C * temp_tensor * (current_time / t0))
        source_density = q_tilde * exponent * C * temp_tensor

        mass_mat = self.mass_matrix_mat.to(device)
        source_vec = torch.sparse.mm(mass_mat, source_density.unsqueeze(1)).squeeze()
        return source_vec.cpu().numpy()

    def export_to_vtk(
        self,
        array_true,
        array_pred,
        time_steps,
        filename="results/vtk/results.vtk",
        material_field: Optional[np.ndarray] = None,
        material_name: str = "MaterialDistribution",
    ):
        """
        Export solutions to VTK file for visualization in Paraview.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        gfu_true = ng.GridFunction(self.fes)
        gfu_pred = ng.GridFunction(self.fes)
        gfu_diff = ng.GridFunction(self.fes)
        material_gfu = None
        gfu_true.vec.FV().NumPy()[:] = array_true[0]
        gfu_pred.vec.FV().NumPy()[:] = array_pred[0]
        # relative error
        gfu_diff.vec.FV().NumPy()[:] = (array_true[0] - array_pred[0]) / (
            np.max(np.abs(array_true[0])) - np.min(np.abs(array_true[0]))
        )
        coefs = [gfu_true, gfu_pred, gfu_diff]
        names = ["ExactSolution", "PredictedSolution", "Difference, %"]

        if material_field is not None:
            material_array = np.array(material_field, dtype=np.float64)
            if material_array.ndim > 1:
                material_array = material_array.reshape(-1)
            if material_array.size == self.fes.ndof:
                material_gfu = ng.GridFunction(self.fes)
                material_gfu.vec.FV().NumPy()[:] = material_array
                coefs.append(material_gfu)
                names.append(material_name)
            else:
                print(
                    f"Warning: Material field length {material_array.size} does not match number of DOFs {self.fes.ndof}. Skipping export."
                )

        vtk_out = ng.VTKOutput(
            self.mesh,
            coefs=coefs,
            names=names,
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
    from mesh_utils import build_graph_from_mesh, create_gaussian_initial_condition
    from containers import MeshProblem
    from graph_creator import GraphCreator
    from train_problems import create_test_problem, create_mms_problem, create_industrial_heating_problem, create_source_test_problem

    problem, time_config = create_source_test_problem(maxh=0.1, alpha=1)

    # Initialize FEM solver
    fem_solver = FEMSolver(problem.mesh, order=1, problem=problem)

    # Solve transient problem
    transient_solution = fem_solver.solve_transient_problem(problem)

    for step_idx, time_value in enumerate(problem.time_config.time_steps, start=1):
        t_prev = torch.tensor(transient_solution[step_idx - 1], dtype=torch.float64)
        t_pred_next = torch.tensor(transient_solution[step_idx], dtype=torch.float64)

        residual = fem_solver.compute_residual(
            t_pred_next,
            t_prev,
            problem,
            time_scalar=float(time_value),
        )
        print(
            f"Time step {step_idx}, Residual (mean): {np.abs(torch.mean(residual).item()):.2e}"
        )
    assert np.mean(residual.numpy()) < 1e-8, "Residual is too high!"

    fem_solver.export_to_vtk(
        np.array(transient_solution),
        np.array(transient_solution),
        problem.time_config.time_steps,
        filename="results/fem_tests/vtk/result",
    )

