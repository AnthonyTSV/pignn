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
import netgen.gui


def test_case(name, wp, ind, **kwargs):
    print(f"\n--- {name} ---")
    builder = IHGeometryAndMesh(wp, ind, **kwargs)
    mesh = builder.generate()
    print(f"  Materials:  {set(mesh.GetMaterials())}")
    print(f"  Boundaries: {set(mesh.GetBoundaries())}")
    print(f"  Elements: {mesh.ne}, Vertices: {mesh.nv}")
    return mesh

kw = dict(h_workpiece=2e-3, h_air=60e-3, h_coil=1e-3, workpiece_boundary_layer_thicknesses=[1e-3, 2e-3, 4e-3])

# 1) Billet + rectangular inductor
print("Test 1...")
mesh = test_case(
    "Billet + Rectangular",
    BilletParams(diameter=0.030, height=0.070),
    RectangularInductorParams(
        coil_inner_diameter=0.050, coil_height=0.040,
        winding_count=1, profile_width=0.007, profile_height=0.007
    ),
    **kw,
)

# 2) Billet + circular inductor
# print("Test 2...")
# mesh = test_case(
#     "Billet + Circular",
#     BilletParams(diameter=0.030, height=0.070),
#     CircularInductorParams(
#         coil_inner_diameter=0.050, coil_height=0.040,
#         winding_count=1, profile_diameter=0.007,
#     ),
#     **kw,
# )

# 3) Tube + rectangular inductor
# print("Test 3...")
# mesh = test_case(
#     "Tube + Rectangular",
#     TubeParams(outer_diameter=0.03, inner_diameter=0.016, height=0.070),
#     RectangularInductorParams(
#         coil_inner_diameter=0.050, coil_height=0.040,
#         winding_count=1, profile_width=0.007, profile_height=0.007,
#     ),
#     **kw,
# )

# 4) Stepped shaft
# print("Test 4...")
# mesh = test_case(
#     "Stepped Shaft + Rectangular",
#     SteppedShaftParams(diameter1=0.03, height1=0.035, diameter2=0.02, height2=0.035),
#     RectangularInductorParams(
#         coil_inner_diameter=0.050, coil_height=0.040,
#         winding_count=1, profile_width=0.007, profile_height=0.007,
#     ),
#     **kw,
# )

# 5) Multi-billet (2 billets)
# print("Test 5...")
# mesh = test_case(
#     "Multi-billet + Rectangular",
#     MultiBilletParams(diameter=0.03, height=0.030, count=2, gap=0.005),
#     RectangularInductorParams(
#         coil_inner_diameter=0.050, coil_height=0.040,
#         winding_count=1, profile_width=0.007, profile_height=0.007,
#     ),
#     **kw,
# )

# 6) Hollow rectangular inductor
# print("Test 6...")
# mesh = test_case(
#     "Billet + Hollow Rectangular",
#     BilletParams(diameter=0.03, height=0.070),
#     RectangularInductorParams(
#         coil_inner_diameter=0.050, coil_height=0.040,
#         winding_count=1, profile_width=0.010, profile_height=0.010,
#         is_hollow=True, wall_thickness=0.002,
#     ),
#     **kw,
# )

# print("\nAll tests passed!")

cf = mesh.RegionCF(VOL, dict(mat_workpiece=0, mat_air=4, mat_coil=7))

Draw(cf, mesh, "regioncf")

input()