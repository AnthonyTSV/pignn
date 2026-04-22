from __future__ import annotations
from typing import List, Sequence, Union
from pydantic import BaseModel

from ngsolve import Mesh
from netgen import occ


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

        # With workpiece skin layers:
        builder = IHGeometryAndMesh(
            wp, ind, workpiece_boundary_layer_thicknesses=[1e-5, 2e-5, 4e-5],
        )
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
        workpiece_boundary_layer_thicknesses: Sequence[float] | None = None,
    ):
        self.wp = workpiece
        self.ind = inductor
        self.air_width = air_width
        self.air_height_factor = air_height_factor
        self.h_workpiece = h_workpiece
        self.h_coil = h_coil
        self.h_air = h_air
        self.workpiece_boundary_layer_thicknesses = (
            None
            if workpiece_boundary_layer_thicknesses is None
            else list(workpiece_boundary_layer_thicknesses)
        )

    @staticmethod
    def _set_face_properties(shape, material: str, maxh: float) -> None:
        """Set OCC face material and mesh size on a face or compound."""
        shape.name = material
        shape.maxh = maxh
        for face in shape.faces:
            face.name = material
            face.maxh = maxh

    @staticmethod
    def _make_rectangle(
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        *,
        material: str,
        maxh: float,
        left: str | None = None,
        right: str | None = None,
        top: str | None = None,
        bottom: str | None = None,
        bc: str | None = None,
    ):
        """Create an OCC rectangular face with optional edge names."""
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("rectangle dimensions must be positive")

        rect = (
            occ.Rectangle(xmax - xmin, ymax - ymin)
            .Face()
            .Move(occ.Vec(xmin, ymin, 0))
        )
        IHGeometryAndMesh._set_face_properties(rect, material, maxh)

        if bc is not None:
            rect.edges.name = bc
        if left is not None:
            rect.edges.Min(occ.X).name = left
        if right is not None:
            rect.edges.Max(occ.X).name = right
        if top is not None:
            rect.edges.Max(occ.Y).name = top
        if bottom is not None:
            rect.edges.Min(occ.Y).name = bottom
        return rect

    @staticmethod
    def _make_circle(
        cx: float,
        cy: float,
        radius: float,
        *,
        material: str,
        maxh: float,
        bc: str | None = None,
    ):
        """Create an OCC circular face with optional edge name."""
        if radius <= 0:
            raise ValueError("circle radius must be positive")

        circle = occ.WorkPlane().Circle(cx, cy, radius).Face()
        IHGeometryAndMesh._set_face_properties(circle, material, maxh)
        if bc is not None:
            circle.edges.name = bc
        return circle

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
        """Return a list of OCC faces/compounds for the workpiece."""
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
        r = wp.diameter / 2.0
        hh = wp.height / 2.0
        rect = self._make_rectangle(
            0,
            y_center - hh,
            r,
            y_center + hh,
            material="mat_workpiece",
            maxh=self.h_workpiece,
            left="bc_workpiece_left",
            right="bc_workpiece_right",
            top="bc_workpiece_top",
            bottom="bc_workpiece_bottom",
        )
        return rect

    def _make_tube(self, wp: TubeParams, y_center: float):
        r_inner = wp.inner_diameter / 2.0
        r_outer = wp.outer_diameter / 2.0
        hh = wp.height / 2.0
        return self._make_rectangle(
            r_inner,
            y_center - hh,
            r_outer,
            y_center + hh,
            material="mat_workpiece",
            maxh=self.h_workpiece,
            left="bc_workpiece_left",
            right="bc_workpiece_right",
            top="bc_workpiece_top",
            bottom="bc_workpiece_bottom",
        )

    def _make_stepped_shaft(self, wp: SteppedShaftParams, y_center: float):
        r1 = wp.diameter1 / 2.0
        r2 = wp.diameter2 / 2.0
        # Upper section (diameter1 / height1) above the step
        rect_upper = self._make_rectangle(
            0,
            y_center,
            r1,
            y_center + wp.height1,
            material="mat_workpiece",
            maxh=self.h_workpiece,
            left="bc_workpiece_left",
            right="bc_workpiece_right",
            top="bc_workpiece_top",
            bottom="bc_workpiece_bottom",
        )
        # Lower section (diameter2 / height2) below the step
        rect_lower = self._make_rectangle(
            0,
            y_center - wp.height2,
            r2,
            y_center,
            material="mat_workpiece",
            maxh=self.h_workpiece,
            left="bc_workpiece_left",
            right="bc_workpiece_right",
            top="bc_workpiece_top",
            bottom="bc_workpiece_bottom",
        )
        stepped = rect_upper + rect_lower
        self._set_face_properties(stepped, "mat_workpiece", self.h_workpiece)
        return stepped

    def _make_multi_billet(self, wp: MultiBilletParams, y_center: float) -> list:
        r = wp.diameter / 2.0
        total_h = self._workpiece_total_height()
        y_start = y_center - total_h / 2

        solids = []
        for i in range(wp.count):
            y_bot = y_start + i * (wp.height + wp.gap)
            solid = self._make_rectangle(
                0,
                y_bot,
                r,
                y_bot + wp.height,
                material="mat_workpiece",
                maxh=self.h_workpiece,
                left="bc_workpiece_left",
                right="bc_workpiece_right",
                top="bc_workpiece_top",
                bottom="bc_workpiece_bottom",
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
        """Return a list of OCC faces/compounds for the inductor windings."""
        y_positions = self._winding_y_positions(y_center)
        solids = []
        for y_pos in y_positions:
            if isinstance(self.ind, CircularInductorParams):
                solids.append(self._make_circular_winding(self.ind, y_pos))
            else:
                solids.append(self._make_rectangular_winding(self.ind, y_pos))
        return solids

    def _make_circular_winding(self, ind: CircularInductorParams, y_pos: float):
        cx = ind.coil_inner_diameter / 2 + ind.profile_diameter / 2
        r = ind.profile_diameter / 2

        outer = self._make_circle(
            cx,
            y_pos,
            r,
            material="mat_coil",
            maxh=self.h_coil,
            bc="bc_coil",
        )
        if ind.is_hollow:
            r_inner = ind.profile_diameter / 2 - ind.wall_thickness
            inner = self._make_circle(
                cx,
                y_pos,
                r_inner,
                material="mat_air",
                maxh=self.h_air,
                bc="bc_coil",
            )
            winding = outer - inner
            self._set_face_properties(winding, "mat_coil", self.h_coil)
            winding.edges.name = "bc_coil"
            return winding
        return outer

    def _make_rectangular_winding(self, ind: RectangularInductorParams, y_pos: float):
        cx = ind.coil_inner_diameter / 2 + ind.profile_width / 2
        hw = ind.profile_width / 2
        hh = ind.profile_height / 2

        outer = self._make_rectangle(
            cx - hw,
            y_pos - hh,
            cx + hw,
            y_pos + hh,
            material="mat_coil",
            maxh=self.h_coil,
            bc="bc_coil",
        )
        if ind.is_hollow:
            t = ind.wall_thickness
            inner = self._make_rectangle(
                cx - hw + t,
                y_pos - hh + t,
                cx + hw - t,
                y_pos + hh - t,
                material="mat_air",
                maxh=self.h_air,
                bc="bc_coil",
            )
            winding = outer - inner
            self._set_face_properties(winding, "mat_coil", self.h_coil)
            winding.edges.name = "bc_coil"
            return winding
        return outer

    @staticmethod
    def _workpiece_domain_indices(ngmesh) -> list[int]:
        return sorted(
            {
                elem.index
                for elem in ngmesh.Elements2D()
                if ngmesh.GetMaterial(elem.index) == "mat_workpiece"
            }
        )

    def _add_workpiece_boundary_layers(self, ngmesh) -> None:
        thicknesses = self.workpiece_boundary_layer_thicknesses
        if not thicknesses:
            return
        if any(thickness <= 0 for thickness in thicknesses):
            raise ValueError("workpiece boundary layer thicknesses must be positive")

        domain_indices = self._workpiece_domain_indices(ngmesh)
        if not domain_indices:
            raise RuntimeError("Cannot add boundary layers: no mat_workpiece domain found")

        for domain_index in domain_indices:
            ngmesh.BoundaryLayer2(domain_index, thicknesses, False)

    def generate(self) -> Mesh:
        """Build the full geometry and return an NGSolve Mesh (in metres)."""
        total_h = self._workpiece_total_height()
        air_height = total_h * self.air_height_factor
        y_center = air_height / 2.0

        air = self._make_rectangle(
            0,
            0,
            self.air_width,
            air_height,
            material="mat_air",
            maxh=self.h_air,
            bc="bc_air",
            left="bc_axis",
        )

        wp_solids = self._build_workpiece(y_center)
        ind_solids = self._build_inductor(y_center)
        inner_solids = wp_solids + ind_solids

        clipped = [s * air for s in inner_solids]

        remaining_air = air
        for s in clipped:
            remaining_air = remaining_air - s
        self._set_face_properties(remaining_air, "mat_air", self.h_air)

        shape = occ.Compound([remaining_air, *clipped])
        geo = occ.OCCGeometry(shape, dim=2)

        ngmesh = geo.GenerateMesh(maxh=self.h_air, grading=0.5)
        self._add_workpiece_boundary_layers(ngmesh)
        return Mesh(ngmesh)
