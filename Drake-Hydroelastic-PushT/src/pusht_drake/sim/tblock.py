"""The Push-T slider ("T") geometry, defined parametrically.

The T is a crossbar and a stem, extruded along z. Two frames matter and they are
not the same:

- Pickle frame: origin at the crossbar centre. This is the frame the upstream
  load_primitive_info reads and the one TBlock.primitive_boxes emits, so it
  matches Zeng's small_t_pusher.pkl convention exactly.
- Body frame: origin at the area centroid. create_arbitrary_shape_sdf_file emits
  global_translation = -com_offset, so this is the frame Drake reports the
  slider pose in, and the one TBlock.outline returns.

Every derived quantity here is computed, never transcribed. Nothing in this module
imports pydrake: it is the shared definition that drivers and checks read so
the T's dimensions live in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PUSHT_T", "TBlock"]


@dataclass(frozen=True)
class TBlock:
    """A Push-T slider. All lengths in meters.

    stem_length is derived rather than stored so the five stored numbers cannot
    disagree with each other.
    """

    total_width: float
    total_height: float
    bar_width: float
    crossbar_height: float
    thickness: float

    def __post_init__(self) -> None:
        if self.stem_length <= 0.0:
            raise ValueError(
                f"crossbar_height {self.crossbar_height} leaves no stem "
                f"in a total_height of {self.total_height}"
            )
        if not 0.0 < self.bar_width <= self.total_width:
            raise ValueError(
                f"bar_width {self.bar_width} does not fit total_width {self.total_width}"
            )
        if self.thickness <= 0.0:
            raise ValueError(f"thickness {self.thickness} must be positive")

    # -- derived lengths ---------------------------------------------------------

    @property
    def stem_length(self) -> float:
        """Length of the vertical stem, below the crossbar.

        Rounded to picometers: the stored dimensions are exact decimals in mm, so the
        difference is binary subtraction noise (0.120 - 0.030 lands 1.4e-17 low), and
        it would otherwise be written verbatim into the pickle and the generated SDF.
        """
        return round(self.total_height - self.crossbar_height, 12)

    @property
    def crossbar_area(self) -> float:
        return self.total_width * self.crossbar_height

    @property
    def stem_area(self) -> float:
        return self.bar_width * self.stem_length

    @property
    def area(self) -> float:
        """Cross-sectional area of the T, in m^2."""
        return self.crossbar_area + self.stem_area

    @property
    def volume(self) -> float:
        return self.area * self.thickness

    # -- frames ------------------------------------------------------------------

    @property
    def stem_centre_y(self) -> float:
        """Stem box centre in the pickle frame (origin = crossbar centre)."""
        return -(self.crossbar_height + self.stem_length) / 2.0

    @property
    def com_offset_y(self) -> float:
        """Area centroid in the pickle frame, i.e. how far it sits below the crossbar centre.

        Uniform density, so this is the area-weighted mean of the two box centres,
        the same quantity compute_com_from_uniform_density derives from the pickle.
        """
        return self.stem_area * self.stem_centre_y / self.area

    @property
    def com_offset(self) -> np.ndarray:
        """[x, y] of the area centroid in the pickle frame."""
        return np.array([0.0, self.com_offset_y])

    def outline(self) -> np.ndarray:
        """The 8 outline vertices in the body frame, counter-clockwise, shape (8, 2)."""
        half_w, half_bar = self.total_width / 2.0, self.bar_width / 2.0
        top = self.crossbar_height / 2.0 - self.com_offset_y
        junction = -self.crossbar_height / 2.0 - self.com_offset_y
        bottom = junction - self.stem_length
        return np.array(
            [
                [-half_w, top],
                [half_w, top],
                [half_w, junction],
                [half_bar, junction],
                [half_bar, bottom],
                [-half_bar, bottom],
                [-half_bar, junction],
                [-half_w, junction],
            ]
        )

    @property
    def max_face_distance(self) -> float:
        """Largest perpendicular distance from the body origin to an outer face.

        For a T this is the stem tip. It is what the workspace analyses mean by "face
        distance from the centroid"; the farthest vertex is slightly further out.
        """
        outline = self.outline()
        return float(max(outline[:, 1].max(), -outline[:, 1].min(), self.total_width / 2.0))

    @property
    def circumradius(self) -> float:
        """Distance from the body origin to the farthest outline vertex."""
        return float(np.linalg.norm(self.outline(), axis=1).max())

    def contact_ring_radius(self, pusher_radius: float) -> float:
        """Radius of the smallest circle containing every pusher-centre contact position.

        Derived from the circumradius, not the face distance. A pusher approaching
        along a corner-ward bearing touches the corner, which sits 1.76 mm further
        out than the farthest face, so the face distance makes a quantity labelled
        "worst case" 1.69 mm optimistic, in the one direction a conservative bound
        must never err.

        This is only a scalar bound. The real envelope is not a circle at all: over
        the T's outline the centre-to-pusher distance runs 0.034-0.116 m, so
        anything that cares about coverage or margin needs the full envelope.
        """
        return self.circumradius + pusher_radius

    def inertia_about_com(self, mass: float):
        """Uniform-density rotational inertia about the COM, body axes, as a 3x3 list.

        The COM coincides with the body origin (the area centroid, mid-thickness),
        so this is exactly the tensor the generated SDF needs at <pose>0</pose>.
        Parallel-axis sum over the two boxes; products of inertia vanish by the T's
        left/right symmetry.

        The diag(1e-5) placeholder inherited from the upstream Drake station
        (Michaelszeng/diffusion-policy-drake; see NOTICE.md) is isotropic
        (impossible for a T) and 19.3x too small in yaw at m = 0.1 kg, which
        simulates a 5.2 g slider for rotation and a 100 g one for gravity and
        friction.
        """

        boxes = self.primitive_boxes()
        total_area = sum(b["size"][0] * b["size"][1] for b in boxes)
        ixx = iyy = izz = 0.0
        for b in boxes:
            w, h, d = b["size"]
            m = mass * (w * h) / total_area
            dy = b["transform"][1, 3] - self.com_offset_y
            ixx += m * (h**2 + d**2) / 12.0 + m * dy**2
            iyy += m * (w**2 + d**2) / 12.0
            izz += m * (w**2 + h**2) / 12.0 + m * dy**2
        return [[float(ixx), 0.0, 0.0], [0.0, float(iyy), 0.0], [0.0, 0.0, float(izz)]]

    # -- the simulator payload ---------------------------------------------------

    def primitive_boxes(self) -> list[dict]:
        """The two boxes in the pickle frame, in the exact shape load_primitive_info expects."""
        stem_transform = np.eye(4)
        stem_transform[1, 3] = self.stem_centre_y
        return [
            {
                "name": "box",
                "size": [self.total_width, self.crossbar_height, self.thickness],
                "transform": np.eye(4),
            },
            {
                "name": "box",
                "size": [self.bar_width, self.stem_length, self.thickness],
                "transform": stem_transform,
            },
        ]


#: The slider this repo simulates: a millimetre-for-pixel replica of canonical
#: Push-T. real-stanford/diffusion_policy's pusht_env.py builds its T with
#: add_tee(..., scale=30) and length = 4, giving 120 x 120 with a 30-wide bar and
#: a 90 stem, and its agent with add_circle(..., 15). Taking one pixel as one
#: millimetre reproduces that exactly, including the pusher: our 15 mm radius against
#: the 30 mm bar is the original's radius-15 circle against its 30-wide bar.
#:
#: The XY proportions are therefore the canonical 4 : 4 : 1 : 1 : 3
#: (width : height : bar width : crossbar height : stem length). Thickness has no
#: counterpart in a 2-D pymunk world and is set independently, at 30 mm.
#:
#: The name carries no size. The size lives in the fields below and in every
#: dataset sidecar, so resizing the T renames nothing.
PUSHT_T = TBlock(
    total_width=0.120,
    total_height=0.120,
    bar_width=0.030,
    crossbar_height=0.030,
    thickness=0.030,
)
