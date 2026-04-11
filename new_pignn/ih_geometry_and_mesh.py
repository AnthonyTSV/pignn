from __future__ import annotations
from typing import List, Union
from pydantic import BaseModel

from ngsolve import Mesh
from netgen.geom2d import CSG2d, Rectangle, Circle


class BilletParams(BaseModel):
    """Solid cylindrical workpiece (simple finite rod)."""
    diameter: float
    height: float


class TubeParams(BaseModel):
    """Hollow cylindrical workpiece."""
    outer_diameter: float
    inner_diameter: float
    height: float

    def __post_init__(self):
        if self.inner_diameter >= self.outer_diameter:
            raise ValueError("inner_diameter must be less than outer_diameter")


class SteppedShaftParams(BaseModel):
    """Two-diameter cylindrical workpiece.  The step is at the junction."""
    diameter1: float # upper section diameter
    height1: float # upper section height
    diameter2: float # lower section diameter
    height2: float # lower section height


class MultiBilletParams(BaseModel):
    """Multiple identical billets stacked along the axis."""
    diameter: float # per-billet diameter
    height: float # per-billet height
    count: int # number of billets
    gap: float # axial gap between billets

    def __post_init__(self):
        if self.count < 1:
            raise ValueError("count must be >= 1")
        if self.gap < 0:
            raise ValueError("gap must be >= 0")


class CircularInductorParams(BaseModel):
    """Inductor with circular cross-section windings."""
    coil_inner_diameter: float # inner diameter of the coil (gap to workpiece)
    coil_height: float # total axial span of the winding area
    winding_count: int # number of windings
    profile_diameter: float # diameter of one winding cross-section
    is_hollow: bool = False
    wall_thickness: float = 0.001 # used when is_hollow=True
    y_offset: float = 0.0 # axial offset of the coil centre


class RectangularInductorParams(BaseModel):
    """Inductor with rectangular cross-section windings."""
    coil_inner_diameter: float # inner diameter of the coil (gap to workpiece)
    coil_height: float # total axial span of the winding area
    winding_count: int
    profile_width: float  # radial extent of one winding [m]
    profile_height: float # axial extent of one winding  [m]
    is_hollow: bool = False
    wall_thickness: float = 0.001
    y_offset: float = 0.0


WorkpieceParams = Union[BilletParams, TubeParams, SteppedShaftParams, MultiBilletParams]
InductorParams = Union[CircularInductorParams, RectangularInductorParams]

# Internal scale factor: CSG2d boolean ops with Circle hang at metre scale
# (netgen bug). We build geometry in mm and scale the mesh back to metres.
_SCALE = 1000.0  # m -> mm


class IHGeometryAndMesh:
    """Creates 2D axisymmetric induction-heating meshes.

    Usage::

        wp  = BilletParams(diameter=0.015, height=0.070)
        ind = RectangularInductorParams(
            coil_inner_diameter=0.025, coil_height=0.070,
            winding_count=1, profile_width=0.007, profile_height=0.007,
        )
        builder = IHGeometryAndMesh(wp, ind)
        mesh = builder.generate()
    """

    def __init__(
        self,
        workpiece: WorkpieceParams,
        inductor: InductorParams,
        *,
        air_width: float = 0.12,
        air_height_factor: float = 3.0,
        h_workpiece: float = 5e-4,
        h_coil: float = 1e-3,
        h_air: float = 60e-3,
    ):
        self.wp = workpiece
        self.ind = inductor
        self.air_width = air_width
        self.air_height_factor = air_height_factor
        self.h_workpiece = h_workpiece
        self.h_coil = h_coil
        self.h_air = h_air
        self._S = _SCALE  # shortcut

    def _workpiece_total_height(self) -> float:
        wp = self.wp
        if isinstance(wp, BilletParams):
            return wp.height
        if isinstance(wp, TubeParams):
            return wp.height
        if isinstance(wp, SteppedShaftParams):
            return wp.height1 + wp.height2
        if isinstance(wp, MultiBilletParams):
            return wp.count * wp.height + max(0, wp.count - 1) * wp.gap
        raise TypeError(f"Unknown workpiece type: {type(wp)}")

    def _build_workpiece(self, y_center: float) -> list:
        """Return a list of CSG2d Solid2d objects for the workpiece."""
        wp = self.wp
        if isinstance(wp, BilletParams):
            return [self._make_billet(wp, y_center)]
        if isinstance(wp, TubeParams):
            return [self._make_tube(wp, y_center)]
        if isinstance(wp, SteppedShaftParams):
            return [self._make_stepped_shaft(wp, y_center)]
        if isinstance(wp, MultiBilletParams):
            return self._make_multi_billet(wp, y_center)
        raise TypeError(f"Unknown workpiece type: {type(wp)}")

    def _make_billet(self, wp: BilletParams, y_center: float):
        S = self._S
        r = wp.diameter / 2.0 * S
        hh = wp.height / 2.0 * S
        yc = y_center * S
        rect = Rectangle(
            pmin=(0, yc - hh),
            pmax=(r, yc + hh),
            mat="mat_workpiece",
            left="bc_workpiece_left",
            right="bc_workpiece_right",
            top="bc_workpiece_top",
            bottom="bc_workpiece_bottom",
        ).Mat("mat_workpiece").Maxh(self.h_workpiece * S)
        
        return rect

    def _make_tube(self, wp: TubeParams, y_center: float):
        S = self._S
        r_inner = wp.inner_diameter / 2.0 * S
        r_outer = wp.outer_diameter / 2.0 * S
        hh = wp.height / 2.0 * S
        yc = y_center * S
        return (
            Rectangle(
                pmin=(r_inner, yc - hh),
                pmax=(r_outer, yc + hh),
                mat="mat_workpiece",
                left="bc_workpiece_left",
                right="bc_workpiece_right",
                top="bc_workpiece_top",
                bottom="bc_workpiece_bottom",
            )
            .Mat("mat_workpiece")
            .Maxh(self.h_workpiece * S)
        )

    def _make_stepped_shaft(self, wp: SteppedShaftParams, y_center: float):
        S = self._S
        r1 = wp.diameter1 / 2.0 * S
        r2 = wp.diameter2 / 2.0 * S
        yc = y_center * S
        # Upper section (diameter1 / height1) above the step
        rect_upper = Rectangle(
            pmin=(0, yc),
            pmax=(r1, yc + wp.height1 * S),
            mat="mat_workpiece",
            left="bc_workpiece_left",
            right="bc_workpiece_right",
            top="bc_workpiece_top",
            bottom="bc_workpiece_bottom",
        )
        # Lower section (diameter2 / height2) below the step
        rect_lower = Rectangle(
            pmin=(0, yc - wp.height2 * S),
            pmax=(r2, yc),
            mat="mat_workpiece",
            left="bc_workpiece_left",
            right="bc_workpiece_right",
            top="bc_workpiece_top",
            bottom="bc_workpiece_bottom",
        )
        return (rect_upper + rect_lower).Mat("mat_workpiece").Maxh(self.h_workpiece * S)

    def _make_multi_billet(self, wp: MultiBilletParams, y_center: float) -> list:
        S = self._S
        r = wp.diameter / 2.0 * S
        total_h = self._workpiece_total_height()
        y_start = (y_center - total_h / 2) * S

        solids = []
        for i in range(wp.count):
            y_bot = y_start + i * (wp.height + wp.gap) * S
            solid = (
                Rectangle(
                    pmin=(0, y_bot),
                    pmax=(r, y_bot + wp.height * S),
                    mat="mat_workpiece",
                    left="bc_workpiece_left",
                    right="bc_workpiece_right",
                    top="bc_workpiece_top",
                    bottom="bc_workpiece_bottom",
                )
                .Mat("mat_workpiece")
                .Maxh(self.h_workpiece * S)
            )
            solids.append(solid)
        return solids

    def _winding_y_positions(self, y_center: float) -> List[float]:
        """Compute the Y-centre of each winding (top to bottom), in metres."""
        ind = self.ind
        if ind.winding_count == 1:
            return [y_center + ind.y_offset]

        # Axial extent of one winding profile
        if isinstance(ind, CircularInductorParams):
            winding_size = ind.profile_diameter
        else:
            winding_size = ind.profile_height

        top_y = y_center + ind.y_offset + ind.coil_height / 2 - winding_size / 2
        spacing = (ind.coil_height - winding_size) / (ind.winding_count - 1)
        return [top_y - i * spacing for i in range(ind.winding_count)]

    def _build_inductor(self, y_center: float) -> list:
        """Return a list of CSG2d Solid2d objects for the inductor windings."""
        y_positions = self._winding_y_positions(y_center)
        solids = []
        for y_pos in y_positions:
            if isinstance(self.ind, CircularInductorParams):
                solids.append(self._make_circular_winding(self.ind, y_pos))
            else:
                solids.append(self._make_rectangular_winding(self.ind, y_pos))
        return solids

    def _make_circular_winding(self, ind: CircularInductorParams, y_pos: float):
        S = self._S
        cx = (ind.coil_inner_diameter / 2 + ind.profile_diameter / 2) * S
        r = ind.profile_diameter / 2 * S
        yp = y_pos * S

        outer = (
            Circle(center=(cx, yp), radius=r, mat="mat_coil", bc="bc_coil")
            .Mat("mat_coil")
            .Maxh(self.h_coil * S)
        )
        if ind.is_hollow:
            r_inner = (ind.profile_diameter / 2 - ind.wall_thickness) * S
            inner = Circle(center=(cx, yp), radius=r_inner)
            return (outer - inner).Mat("mat_coil").Maxh(self.h_coil * S)
        return outer

    def _make_rectangular_winding(self, ind: RectangularInductorParams, y_pos: float):
        S = self._S
        cx = (ind.coil_inner_diameter / 2 + ind.profile_width / 2) * S
        hw = ind.profile_width / 2 * S
        hh = ind.profile_height / 2 * S
        yp = y_pos * S

        outer = (
            Rectangle(
                pmin=(cx - hw, yp - hh),
                pmax=(cx + hw, yp + hh),
                mat="mat_coil",
                bc="bc_coil",
            )
            .Mat("mat_coil")
            .Maxh(self.h_coil * S)
        )
        if ind.is_hollow:
            t = ind.wall_thickness * S
            inner = Rectangle(
                pmin=(cx - hw + t, yp - hh + t),
                pmax=(cx + hw - t, yp + hh - t),
            )
            return (outer - inner).Mat("mat_coil").Maxh(self.h_coil * S)
        return outer

    @staticmethod
    def _scale_mesh_to_metres(mesh: Mesh) -> None:
        """Scale mesh coordinates from mm back to metres (in-place)."""
        factor = 1.0 / _SCALE
        for p in mesh.ngmesh.Points():
            p[0] = p[0] * factor
            p[1] = p[1] * factor
        mesh.ngmesh.Update()

    def generate(self) -> Mesh:
        """Build the full geometry and return an NGSolve Mesh (in metres)."""
        S = self._S
        total_h = self._workpiece_total_height()
        air_height = total_h * self.air_height_factor
        y_center = air_height / 2.0  # in metres — helpers convert to mm

        geo = CSG2d()

        # Air bounding box (in mm)
        air = (
            Rectangle(
                pmin=(0, 0),
                pmax=(self.air_width * S, air_height * S),
                mat="mat_air",
                bc="bc_air",
                left="bc_axis",
            )
            .Mat("mat_air")
            .Maxh(self.h_air * S)
        )

        # Collect inner-domain solids (already in mm from helpers)
        wp_solids = self._build_workpiece(y_center)
        ind_solids = self._build_inductor(y_center)
        inner_solids = wp_solids + ind_solids

        # Intersect each solid with the air box
        clipped = [s * air for s in inner_solids]

        # Subtract all inner domains from air
        remaining_air = air
        for s in clipped:
            remaining_air = remaining_air - s

        # Add all domains
        geo.Add(remaining_air)
        for s in clipped:
            geo.Add(s)

        ngmesh = geo.GenerateMesh()
        mesh = Mesh(ngmesh)

        # Scale mesh coordinates back from mm to metres
        self._scale_mesh_to_metres(mesh)
        return mesh
