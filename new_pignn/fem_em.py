import numpy as np
from containers import MeshConfig, MeshProblem, MeshProblemEM
import ngsolve as ng
from typing import Optional, List, Tuple
import torch
import scipy.sparse as sp
import os
from pathlib import Path
from mesh_utils import create_rectangular_mesh

r_star = 70 * 1e-3  # m
A_star = 4.8 * 1e-4  # Wb/m
mu_star = 4 * 3.1415926535e-7 # H/m
J_star = A_star / (r_star**2 * mu_star)

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

    def _ngsolve_to_torch(self, ngsolve_matrix):
        """Convert NGSolve matrix to torch tensor."""
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
        # a += 1j * self.problem.omega * sigma * r * A * v * ng.dx
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


if __name__ == "__main__":

    import ngsolve as ng
    from containers import MeshProblemEM
    from graph_creator import GraphCreator
    from train_problems import create_em_problem

    problem = create_em_problem()

    # Initialize FEM solver
    fem_solver = FEMSolverEM(problem.mesh, order=1, problem=problem)

    gfA = fem_solver.solve(problem)
    # random_solution = np.random.rand(len(curl_gfa)) + 1j * np.random.rand(len(curl_gfa))
    # curl_gfa_noisy = curl_gfa + 1e-6 * (
    #     np.random.rand(len(curl_gfa)) + 1j * np.random.rand(len(curl_gfa))
    # )

    # npz_filename = "results/physics_informed/test_em_problem/results_data/results.npz"
    # data = np.load(npz_filename)
    # gfA = data["predicted"] * 1e-6

    residual = fem_solver.compute_residual(gfA)
    residuals_abs = np.absolute(residual.cpu().numpy())
    print(residuals_abs)
    print(f"Mean residual: {np.mean(residuals_abs)}")

    # fem_solver.export_to_vtk(
    #     curl_gfa,
    #     curl_gfa,
    #     filename="results/fem_tests_em/vtk/result",
    # )
