"""Generate the T's SDF, into a content-addressed cache.

The slider is the one model that cannot ship as a static file: its inertia
depends on the configured mass, and its contact properties are configuration.
So it is generated, and where and how is what this module is about.

Upstream regenerated ``simulation/models/arbitrary_shape.sdf`` inside
the package's own source tree on every config load, which had three
consequences that content addressing removes:

- it writes into ``site-packages`` on an installed wheel, which may be read-only;
- concurrent evaluation workers race on the bytes, so every entry point needed
  an ``fcntl`` lock and an ``os.chdir`` to the repository root first;
- a stale file from a different configuration is reused with no error.

The path here is derived from a hash of the generated text, so a given
configuration always maps to its own file, writing is idempotent by
construction, and two processes racing on the same path write identical bytes.

Only boxes are emitted. The upstream generator handled spheres, ellipsoids,
cylinders and external visual meshes through a pickle format; the T is two
boxes, and their dimensions come from :data:`pusht_drake.sim.tblock.PUSHT_T`
rather than from a pickle that has to agree with it.

The emitted numbers are bit-for-bit the ones the upstream generator emitted:
the box poses, the centroid offset and the inertia round-trip through
``repr(float)`` exactly as they did through upstream's f-strings.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from pusht_drake.sim.env_config import SliderConfig


def _cache_root() -> Path:
    """``$PUSHT_DRAKE_CACHE``, else ``$XDG_CACHE_HOME``, else ``~/.cache``."""
    override = os.environ.get("PUSHT_DRAKE_CACHE")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "pusht-drake"


def _fmt(value: float) -> str:
    """Full-precision float, so the emitted bytes round-trip the config exactly."""
    return repr(float(value))


def _box(kind: str, index: int, size, pose_y: float, body: str) -> str:
    sx, sy, sz = (_fmt(v) for v in size)
    return (
        f'    <{kind} name="{kind}_{index}">\n'
        f"      <pose>0 {_fmt(pose_y)} 0 0 0 0</pose>\n"
        f"      <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>\n"
        f"{body}"
        f"    </{kind}>\n"
    )


def slider_sdf_source(slider: SliderConfig) -> str:
    """The SDF text for the configured T. Pure: same input, same bytes.

    The body frame origin is the T's area centroid, not the crossbar centre,
    achieved by translating both boxes by ``-com_offset``. It is the easiest
    off-by-one to make in this task: Drake reports the slider pose in the body
    frame, so every consumer of that pose (coverage, the observation) inherits
    this choice.

    Visuals are emitted before collisions, which is also the order Drake's
    SDFormat parser registers them in, so the geometry ids fall the same way
    they did upstream.
    """
    block = slider.block
    inertia = slider.inertia
    offset = float(block.com_offset_y)
    r, g, b, a = (_fmt(v) for v in slider.rgba)

    material = f"      <material><diffuse>{r} {g} {b} {a}</diffuse></material>\n"
    # Drake's discrete (SAP) solver reads only the dynamic coefficient, but both
    # are emitted so the parsed contact material is unambiguous.
    # hunt_crossley_dissipation is not emitted at all: SAP ignores it outright,
    # and carrying it would invite a reader to transcribe a value that does
    # nothing. The upstream generator emitted the same five tags.
    proximity = (
        "      <drake:proximity_properties>\n"
        "        <drake:compliant_hydroelastic/>\n"
        f"        <drake:hydroelastic_modulus>{_fmt(slider.hydroelastic_modulus)}"
        "</drake:hydroelastic_modulus>\n"
        f"        <drake:mesh_resolution_hint>{_fmt(slider.mesh_resolution_hint)}"
        "</drake:mesh_resolution_hint>\n"
        f"        <drake:mu_dynamic>{_fmt(slider.mu_dynamic)}</drake:mu_dynamic>\n"
        f"        <drake:mu_static>{_fmt(slider.mu_static)}</drake:mu_static>\n"
        "      </drake:proximity_properties>\n"
    )

    # primitive_boxes() places the crossbar at the origin; shifting by
    # -com_offset moves the body frame onto the area centroid.
    poses = [float(np.asarray(box["transform"])[1, 3]) - offset for box in block.primitive_boxes()]
    sizes = [box["size"] for box in block.primitive_boxes()]

    parts = [
        '<?xml version="1.0"?>\n',
        '<sdf version="1.7">\n',
        f'  <model name="{slider.name}">\n',
        f'    <link name="{slider.name}">\n',
        "      <inertial>\n",
        "        <pose>0 0 0 0 0 0</pose>\n",
        f"        <mass>{_fmt(slider.mass)}</mass>\n",
        "        <inertia>\n",
        f"          <ixx>{_fmt(inertia[0][0])}</ixx>"
        f"<ixy>{_fmt(inertia[0][1])}</ixy>"
        f"<ixz>{_fmt(inertia[0][2])}</ixz>\n",
        f"          <iyy>{_fmt(inertia[1][1])}</iyy><iyz>{_fmt(inertia[1][2])}</iyz>\n",
        f"          <izz>{_fmt(inertia[2][2])}</izz>\n",
        "        </inertia>\n",
        "      </inertial>\n",
    ]
    for index, (size, pose_y) in enumerate(zip(sizes, poses)):
        parts.append(_box("visual", index, size, pose_y, material))
    for index, (size, pose_y) in enumerate(zip(sizes, poses)):
        parts.append(_box("collision", index, size, pose_y, proximity))
    parts += ["    </link>\n", "  </model>\n", "</sdf>\n"]
    return "".join(parts)


def slider_sdf_sha256(slider: SliderConfig) -> str:
    return hashlib.sha256(slider_sdf_source(slider).encode("utf-8")).hexdigest()


def ensure_slider_sdf(slider: SliderConfig, cache_dir: Path | None = None) -> Path:
    """Write the T's SDF if it is not already cached, and return its path.

    Safe to call concurrently from any number of processes: the path is derived
    from the content, so racing writers write identical bytes, and the write
    itself is atomic (temp file plus ``os.replace``) so no reader can ever see a
    partial file.
    """
    source = slider_sdf_source(slider)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    directory = Path(cache_dir) if cache_dir is not None else _cache_root()
    path = directory / f"slider-{digest}.sdf"
    if path.exists():
        return path

    directory.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".slider-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(source)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path
