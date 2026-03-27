import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "new_pignn"))

from new_pignn.ih_geometry_and_mesh import (
    IHGeometryAndMesh,
    BilletParams,
    TubeParams,
    SteppedShaftParams,
    MultiBilletParams,
    CircularInductorParams,
    RectangularInductorParams,
)
from ngsolve import Draw, VOL
import ngsolve as ng
import numpy as np
# import netgen.gui

mm = 1e-3

def create_mesh():
    builder = IHGeometryAndMesh(
        BilletParams(diameter=60 * mm, height=500 * mm),
        RectangularInductorParams(
            coil_inner_diameter=78 * mm, coil_height=500 * mm,
            winding_count=10, profile_width=20 * mm, profile_height=40 * mm,
            is_hollow=True, wall_thickness=3 * mm
        ),
        h_workpiece=0.5 * mm, h_coil=2 * mm, h_air=100 * mm,
        air_width = 300 * mm,
        air_height_factor=2.0,
    )
    mesh = builder.generate()
    return mesh

mesh = create_mesh()

# Draw(mesh)
# input()

# Physical parameters
mu0 = 4 * 3.1415926535e-7
mu_r_workpiece = 100
mu_r_air = 1.0
mu_r_coil = 1.0

sigma_workpiece = 4761904
sigma_air = 0.0
sigma_coil = 58823529

# Coil parameters
N_turns = 1
I_coil = 4950
frequency = 2000
omega = 2 * ng.pi * frequency
full_coil_area = 20 * mm * 40 * mm
hollow_coil_area = (20 - 2 * 3) * mm * (40 - 2 * 3) * mm
area_coil = full_coil_area - hollow_coil_area
fill_factor = 1

rho = 8030
cp = 494
lambda_ = 48.6
initial_temperature = 22

T_c = 770
emissivity = 0.8
stefan_boltzmann = 5.670374419e-8

def temp_func(T):
    return ng.IfPos(T_c - T, 1 - (T / T_c) ** 6, 0.0)

def build_materials(T):
    temperature_cf = mesh.MaterialCF({"mat_workpiece": T}, default=initial_temperature)
    mu_r_temp = mesh.MaterialCF(
        {
            "mat_workpiece": 1 + (mu_r_workpiece - 1) * temp_func(temperature_cf),
            "mat_air": mu_r_air,
            "mat_coil": mu_r_coil,
        },
        default=1.0,
    )
    sigma_temp = mesh.MaterialCF(
        {
            "mat_workpiece": sigma_workpiece,
            "mat_air": sigma_air,
            "mat_coil": sigma_coil,
        },
        default=0.0,
    )
    return mu_r_temp, sigma_temp

def solve_em(T):
    # A_phi space
    mu_r_temp, sigma_temp = build_materials(T)

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

    nu = 1.0 / (mu0 * mu_r_temp)

    a = ng.BilinearForm(fes, symmetric=False)
    a += nu * (r * dzA * dzv + r1 * dr_rA * dr_rv) * ng.dx

    A_eff = A + phi_coil * r1
    v_eff = v + psi * r1
    a += 1j * omega * sigma_temp * r * A_eff * v_eff * ng.dx

    I_spec = N_turns * I_coil
    Js_phi = I_spec / (area_coil * fill_factor)

    f = ng.LinearForm(fes)
    f += (-Js_phi) * psi * ng.dx("mat_coil")

    a.Assemble()
    f.Assemble()

    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="pardiso") * f.vec

    return gfA, gfPhi, r1, sigma_temp

def build_transient_stepper(
    fes,
    dt,
    thermal_conductivity=1.0,
    density=1.0,
    specific_heat=1.0,
):
    u, v = fes.TnT()

    rho_c = density * specific_heat
    k = thermal_conductivity

    h_conv = 10
    T_amb = 22
    T_amb_kelvin = T_amb + 273.15

    convective_bc = "bc_workpiece_right|bc_workpiece_top|bc_workpiece_bottom"
    jac = ng.x

    M = ng.BilinearForm(fes)
    M += rho_c * jac * u * v * ng.dx
    M.Assemble()
    rhs = M.mat.CreateVector()

    def advance(gfu, heat_source):
        T_kelvin = gfu + 273.15
        radiation_jacobian = 4.0 * emissivity * stefan_boltzmann * T_kelvin ** 3
        radiation_flux = emissivity * stefan_boltzmann * (
            T_kelvin ** 4 - T_amb_kelvin ** 4
        )
        radiation_constant = radiation_flux - radiation_jacobian * gfu

        S = ng.BilinearForm(fes)
        S += rho_c * jac * u * v * ng.dx
        S += dt * (
            k * jac * ng.InnerProduct(ng.grad(u), ng.grad(v)) * ng.dx
            + h_conv * jac * u * v * ng.ds(convective_bc)
            + radiation_jacobian * jac * u * v * ng.ds(convective_bc)
        )
        S.Assemble()

        F = ng.LinearForm(fes)
        F += jac * heat_source * v * ng.dx
        F += h_conv * jac * T_amb * v * ng.ds(convective_bc)
        F += -radiation_constant * jac * v * ng.ds(convective_bc)
        F.Assemble()

        invS = S.mat.Inverse(freedofs=fes.FreeDofs(), inverse="pardiso")
        rhs.data = M.mat * gfu.vec + dt * F.vec
        gfu.vec.data = invS * rhs

    return advance

def curl(u):
    gradu = ng.grad(u)
    return ng.CF((-gradu[1], gradu[0] + ng.IfPos(ng.x, u / ng.x, gradu[0])))

def solve_coupled_problem(dt, t_final):
    fes_t = ng.H1(mesh, order=1, definedon="mat_workpiece")
    gfuT = ng.GridFunction(fes_t)
    gfuT.Set(initial_temperature)
    _, sigma_export = build_materials(gfuT)

    advance_temperature = build_transient_stepper(
        fes_t,
        dt,
        thermal_conductivity=lambda_,
        density=rho,
        specific_heat=cp,
    )

    nsteps = int(round(t_final / dt))
    time_steps = [step * dt for step in range(nsteps + 1)]
    temperature_states = []
    final_fields = {}
    vtk_out = None
    export_gfA = None
    export_gfPhi = None
    export_Q = None
    export_current_density = None
    export_E_phi = None
    export_curl_gfA = None

    for step_idx, time in enumerate(time_steps):
        gfA, gfPhi, r1, sigma_temp = solve_em(gfuT)
        E_phi = -1j * omega * (gfA + gfPhi * r1)
        J_phi = sigma_temp * E_phi
        Q = 0.5 * sigma_temp * ng.Norm(E_phi) ** 2
        current_density = ng.Norm(J_phi)
        curl_gfA = curl(gfA)

        temperature_states.append(gfuT.vec.FV().NumPy().copy())

        if vtk_out is None:
            export_gfA = gfA
            export_gfPhi = gfPhi
            export_E_phi = -1j * omega * (export_gfA + export_gfPhi * r1)
            export_Q = 0.5 * sigma_export * ng.Norm(export_E_phi) ** 2
            export_current_density = ng.Norm(sigma_export * export_E_phi)
            export_curl_gfA = ng.Norm(curl(export_gfA))
            vtk_out = ng.VTKOutput(
                mesh,
                coefs=[
                    gfuT,
                    export_curl_gfA,
                    ng.Norm(export_gfA),
                    export_Q,
                    export_current_density,
                    ng.Norm(export_gfPhi),
                    ng.Norm(export_E_phi),
                ],
                names=[
                    "Temperature",
                    "curl_gfa",
                    "gfa",
                    "Joule_heating",
                    "Current_density",
                    "gfPhi",
                    "E_phi",
                ],
                filename="fem_tests/team_36_simple/result",
                order=1,
            )
        else:
            export_gfA.vec.data = gfA.vec
            export_gfPhi.vec.data = gfPhi.vec

        vtk_out.Do(time=time)

        final_fields = {
            "temperature": gfuT,
            "magnetic_potential": gfA,
            "coil_potential": gfPhi,
            "joule_heating": Q,
            "current_density": current_density,
            "electric_field": ng.Norm(E_phi),
            "curl_magnetic_potential": ng.Norm(curl_gfA),
        }

        if step_idx < nsteps:
            advance_temperature(gfuT, Q)

    return temperature_states, final_fields

if __name__ == "__main__":
    dt = 0.5
    t_final = 25

    solve_coupled_problem(dt, t_final)
