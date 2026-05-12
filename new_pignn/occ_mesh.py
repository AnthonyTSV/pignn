from netgen import occ
import netgen
import ngsolve as ng
from ngsolve import Draw
# import netgen.gui

width = 2.0
height = 1.0
maxh = 0.15
thicknesses = [0.01, 0.015, 0.02, 0.03]

shape = occ.Rectangle(width, height).Face()

shape.edges.Min(occ.X).name = "left"
shape.edges.Max(occ.X).name = "right"
shape.edges.Min(occ.Y).name = "bottom"
shape.edges.Max(occ.Y).name = "top"

geo = occ.OCCGeometry(shape, dim=2)

ngmesh = geo.GenerateMesh(maxh=maxh)
ngmesh.BoundaryLayer2(1, thicknesses, False)

mesh = ng.Mesh(ngmesh)

print(mesh.GetBoundaries())

# Draw(mesh)
# input()
