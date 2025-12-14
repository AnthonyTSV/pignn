import numpy as np
from containers import MeshConfig, MeshProblem, MeshProblemEM
import ngsolve as ng
from typing import Optional, List, Tuple
import torch
import scipy.sparse as sp
import os
from pathlib import Path
from mesh_utils import create_rectangular_mesh


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

        profile_width = 7 * 1e-3  # m
        profile_height = 7 * 1e-3  # m

        mu0 = 4 * 3.1415926535e-7  # Permeability of free space
        mu_r_workpiece = 100  # Relative permeability of workpiece
        mu_r_air = 1.0
        mu_r_coil = 1.0

        sigma_workpiece = 6250000.0  # S/m
        sigma_air = 0.0
        sigma_coil = 58823529.0

        # Coil parameters
        N_turns = 1  # Number of turns
        I_coil = 1000  # A
        coil_area = profile_width * profile_height  # Cross-sectional area
        frequency = 1000  # Hz
        omega = 2 * ng.pi * frequency  # rad/s

        # Define material properties as coefficient functions
        mu_r = self.mesh.MaterialCF(
            {
                "mat_workpiece": mu_r_workpiece,
                "mat_air": mu_r_air,
                "mat_coil": mu_r_coil,
            },
            default=1.0,
        )

        sigma = self.mesh.MaterialCF(
            {
                "mat_workpiece": sigma_workpiece,
                "mat_air": sigma_air,
                "mat_coil": sigma_coil,
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

        nu = 1.0 / (mu0 * mu_r)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + inv_r * dr_rA * dr_rv) * ng.dx
        a += 1j * omega * sigma * r * A * v * ng.dx

        Acoil = profile_width * profile_height
        Js_phi = N_turns * I_coil / Acoil
        f = ng.LinearForm(fes)
        f += r * Js_phi * v * ng.dx("mat_coil")

        a.Assemble()
        f.Assemble()

        self.bilinear_form = self._ngsolve_to_torch(a.mat)
        self.linear_form = torch.tensor(
            f.vec.FV().NumPy().copy(), dtype=torch.complex128
        ).to(self.device)
        self.fes = fes

    def _ngsolve_to_torch(self, ngsolve_matrix):
        """Convert NGSolve matrix to torch tensor."""
        rows, cols, vals = ngsolve_matrix.COO()
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.complex128)
        shape = (ngsolve_matrix.height, ngsolve_matrix.width)
        tensor = torch.sparse_coo_tensor(
            indices, values, size=shape, dtype=torch.complex128
        )
        return tensor.to(self.device)

    def solve(self, problem):
        profile_width = 7 * 1e-3  # m
        profile_height = 7 * 1e-3  # m

        mu0 = 4 * 3.1415926535e-7  # Permeability of free space
        mu_r_workpiece = 100  # Relative permeability of workpiece
        mu_r_air = 1.0
        mu_r_coil = 1.0

        sigma_workpiece = 6250000.0  # S/m
        sigma_air = 0.0
        sigma_coil = 58823529.0

        # Coil parameters
        N_turns = 1  # Number of turns
        I_coil = 1000  # A
        coil_area = profile_width * profile_height  # Cross-sectional area
        frequency = 1000  # Hz
        omega = 2 * ng.pi * frequency  # rad/s

        # Define material properties as coefficient functions
        mu_r = self.mesh.MaterialCF(
            {
                "mat_workpiece": mu_r_workpiece,
                "mat_air": mu_r_air,
                "mat_coil": mu_r_coil,
            },
            default=1.0,
        )

        sigma = self.mesh.MaterialCF(
            {
                "mat_workpiece": sigma_workpiece,
                "mat_air": sigma_air,
                "mat_coil": sigma_coil,
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

        nu = 1.0 / (mu0 * mu_r)

        a = ng.BilinearForm(fes, symmetric=False)
        a += nu * (r * dzA * dzv + inv_r * dr_rA * dr_rv) * ng.dx
        a += 1j * omega * sigma * r * A * v * ng.dx

        Acoil = profile_width * profile_height
        Js_phi = N_turns * I_coil / Acoil
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
            pred_sol_tensor = torch.tensor(pred_sol, dtype=torch.complex128)
        else:
            # It's already a tensor - convert dtype while preserving gradients
            pred_sol_tensor = pred_sol.to(dtype=torch.complex128)

        pred_sol_tensor = pred_sol_tensor.to(self.device)

        Ax = torch.sparse.mm(self.bilinear_form, pred_sol_tensor.unsqueeze(1)).squeeze(
            1
        )
        res = Ax - self.linear_form

        return res

    def export_to_vtk(
        self,
        array_true,
        array_pred,
        filename="results/vtk/results.vtk",
    ):
        """
        Export solutions to VTK file for visualization in Paraview.
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Create GridFunctions for true and predicted solutions (magnitude)
        gfu_true_mag = ng.GridFunction(self.fes)
        gfu_pred_mag = ng.GridFunction(self.fes)

        # Set complex values properly
        gfu_true_mag.vec.FV().NumPy()[:] = array_true
        gfu_pred_mag.vec.FV().NumPy()[:] = array_pred

        # Create magnitude versions for visualization
        gfu_true_abs = ng.GridFunction(ng.H1(self.mesh, order=self.order))
        gfu_pred_abs = ng.GridFunction(ng.H1(self.mesh, order=self.order))
        gfu_true_abs.vec.FV().NumPy()[:] = np.abs(array_true)
        gfu_pred_abs.vec.FV().NumPy()[:] = np.abs(array_pred)

        coefs = [
            gfu_true_abs,
            gfu_pred_abs,
        ]
        names = ["ExactSolution_magnitude", "PredictedSolution_magnitude"]

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

    curl_gfa = fem_solver.solve()
    # random_solution = np.random.rand(len(curl_gfa)) + 1j * np.random.rand(len(curl_gfa))
    # curl_gfa_noisy = curl_gfa + 1e-6 * (
    #     np.random.rand(len(curl_gfa)) + 1j * np.random.rand(len(curl_gfa))
    # )
    residual = fem_solver.compute_residual(curl_gfa)
    print(f"FEM solution computed with residual: {residual:.6e}")

    # fem_solver.export_to_vtk(
    #     curl_gfa,
    #     curl_gfa,
    #     filename="results/fem_tests_em/vtk/result",
    # )
