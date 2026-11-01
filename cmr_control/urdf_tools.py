"""Generate UR7e URDF variants with a tool body rigidly attached to ``tool0``.

The course URDF (``ur_simulation/assets/urdf/ur7e.urdf``) references meshes with relative
paths, so generated files rewrite them to absolute paths and are cached under
``~/.cache/cmr_control`` (or $CMR_CACHE).

Tools
-----
"none"     : bare arm (identical dynamics to the course URDF)
"payload"  : a point-like payload of ``payload_mass`` kg, 5 cm below tool0
"peg"      : a square peg (side ``peg_side``, length ``peg_length``) + a massless ``peg_tip``
             frame at its lower end. The peg joint gets a F/T sensor in the simulation.
"""
from __future__ import annotations

import hashlib
import os
import re
from importlib.resources import files
from pathlib import Path

ASSET_DIR = Path(files("ur_simulation.assets"))
BASE_URDF = ASSET_DIR / "urdf" / "ur7e.urdf"

CACHE_DIR = Path(os.environ.get("CMR_CACHE", Path.home() / ".cache" / "cmr_control"))

PEG_SIDE = 0.020     # m, square cross-section
PEG_LENGTH = 0.080   # m
PEG_MASS = 0.10      # kg


def _box_inertia(m, x, y, z):
    return (m * (y * y + z * z) / 12.0, m * (x * x + z * z) / 12.0, m * (x * x + y * y) / 12.0)


def _tool_xml(tool: str, payload_mass: float, peg_side: float, peg_length: float, peg_mass: float) -> str:
    if tool == "none":
        return ""
    if tool == "payload":
        m = max(payload_mass, 1e-4)
        ixx = iyy = izz = max(m * 0.03 ** 2 * 0.4, 1e-6)  # small solid sphere r=3cm
        return f"""
  <link name="payload">
    <inertial>
      <mass value="{m}"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual><origin xyz="0 0 0.05"/><geometry><sphere radius="0.03"/></geometry>
      <material name="payload_mat"><color rgba="0.9 0.6 0.1 1"/></material></visual>
  </link>
  <joint name="tool0-payload" type="fixed">
    <parent link="tool0"/><child link="payload"/><origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
"""
    if tool == "peg":
        s, L, m = peg_side, peg_length, peg_mass
        ixx, iyy, izz = _box_inertia(m, s, s, L)
        return f"""
  <link name="peg">
    <inertial>
      <mass value="{m}"/>
      <origin xyz="0 0 {L / 2}" rpy="0 0 0"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual><origin xyz="0 0 {L / 2}"/><geometry><box size="{s} {s} {L}"/></geometry>
      <material name="peg_mat"><color rgba="0.2 0.4 0.9 1"/></material></visual>
    <collision><origin xyz="0 0 {L / 2}"/><geometry><box size="{s} {s} {L}"/></geometry></collision>
  </link>
  <joint name="tool0-peg" type="fixed">
    <parent link="tool0"/><child link="peg"/><origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <link name="peg_tip"/>
  <joint name="peg-peg_tip" type="fixed">
    <parent link="peg"/><child link="peg_tip"/><origin xyz="0 0 {L}" rpy="0 0 0"/>
  </joint>
"""
    raise ValueError(f"unknown tool '{tool}'")


def make_urdf(
    tool: str = "none",
    payload_mass: float = 0.0,
    link_mass_scale: float = 1.0,
    peg_side: float = PEG_SIDE,
    peg_length: float = PEG_LENGTH,
    peg_mass: float = PEG_MASS,
) -> str:
    """Return the path of a generated URDF.

    ``link_mass_scale`` multiplies all arm link masses and inertias (used to create a
    deliberately wrong *nominal* model for robustness experiments).
    """
    text = BASE_URDF.read_text(encoding="utf-8")
    mesh_root = (ASSET_DIR / "meshes").as_posix()
    text = text.replace('filename="../meshes', f'filename="{mesh_root}')

    if link_mass_scale != 1.0:
        k = link_mass_scale

        def scale_mass(mo):
            return f'<mass value="{float(mo.group(1)) * k}"/>'

        def scale_inertia(mo):
            vals = [float(v) * k for v in mo.groups()]
            return ('<inertia ixx="{}" ixy="{}" ixz="{}" iyy="{}" iyz="{}" izz="{}"/>'.format(*vals))

        text = re.sub(r'<mass value="([-0-9.eE+]+)"\s*/>', scale_mass, text)
        num = r'"([-0-9.eE+]+)"'
        text = re.sub(
            rf"<inertia ixx={num} ixy={num} ixz={num} iyy={num} iyz={num} izz={num}\s*/>", scale_inertia, text
        )

    text = text.replace("</robot>", _tool_xml(tool, payload_mass, peg_side, peg_length, peg_mass) + "\n</robot>")

    # pybullet assigns mass = 1 kg and unit inertia to every link without an <inertial> tag.
    # The course URDF has three such frames on the wrist (ft_frame, flange, tool0), i.e. 3 kg of
    # phantom payload. Give all frame-only links a negligible inertia instead.
    text = re.sub(
        r'<link name="([^"]+)"\s*/>',
        r'<link name="\1"><inertial><mass value="1e-4"/><origin xyz="0 0 0" rpy="0 0 0"/>'
        r'<inertia ixx="1e-8" ixy="0" ixz="0" iyy="1e-8" iyz="0" izz="1e-8"/></inertial></link>',
        text,
    )

    key = hashlib.md5(text.encode()).hexdigest()[:10]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"ur7e_{tool}_{key}.urdf"
    if not out.exists():
        tmp = out.with_suffix(f".{os.getpid()}.tmp")  # atomic write: safe with many Slurm array tasks
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out)
    return out.as_posix()
