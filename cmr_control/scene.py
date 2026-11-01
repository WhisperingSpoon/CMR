"""Peg-in-hole fixture: a square hole with a 45 degree chamfer, built as one static compound body.

Geometry (fixture frame, origin = centre of the hole bottom):
    hole half-width a = peg_side/2 + clearance, depth H, chamfer width c, outer half-width W.
Each of the four sides consists of
    - a lower wall  (inner face at |x| = a,     z in [0, H-c]),
    - an upper wall (inner face at |x| = a + c, z in [H-c, H]),
    - a 45 deg rotated box ("diamond") whose upper face is the chamfer from (a+c, H) to (a, H-c).
The fixture rests on a base plate of thickness ``plate`` on the floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cmr_control.urdf_tools import PEG_SIDE


@dataclass
class HoleSpec:
    center_xy: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.55]))
    clearance: float = 0.0005   # m per side (1 mm total)
    depth: float = 0.030        # m
    chamfer: float = 0.004      # m
    outer: float = 0.06         # m, outer half-width
    plate: float = 0.02         # m, base plate thickness (hole bottom height above floor)
    friction: float = 0.5
    peg_side: float = PEG_SIDE

    @property
    def a(self):
        return self.peg_side / 2 + self.clearance

    @property
    def bottom(self):
        """World position of the hole bottom centre."""
        return np.array([self.center_xy[0], self.center_xy[1], self.plate])

    @property
    def top(self):
        return self.bottom + np.array([0, 0, self.depth])


def _rotz(k):
    c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def fixture_boxes(spec: HoleSpec):
    """List of (half_extents, position, quaternion) in the fixture frame."""
    a, H, c, W, P = spec.a, spec.depth, spec.chamfer, spec.outer, spec.plate
    boxes = [(np.array([W, W, P / 2]), np.array([0, 0, -P / 2]), np.array([0, 0, 0, 1.0]))]  # base plate
    s = c / np.sqrt(2)
    q_y45 = np.array([0, np.sin(np.pi / 8), 0, np.cos(np.pi / 8)])  # 45 deg about y
    for k in range(4):
        R = _rotz(k)
        qz = np.array([0, 0, np.sin(k * np.pi / 4), np.cos(k * np.pi / 4)])
        local = [
            # lower wall
            (np.array([(W - a) / 2, W, (H - c) / 2]), np.array([(a + W) / 2, 0, (H - c) / 2]), np.array([0, 0, 0, 1.0])),
            # upper wall
            (np.array([(W - a - c) / 2, W, c / 2]), np.array([(a + c + W) / 2, 0, H - c / 2]), np.array([0, 0, 0, 1.0])),
            # chamfer diamond
            (np.array([s, a + c, s]), np.array([a + c, 0, H - c]), q_y45),
        ]
        for he, pos, q in local:
            if np.allclose(q, [0, 0, 0, 1]):  # axis-aligned box: swap x/y extents on the +-y sides
                boxes.append((np.abs(R @ he), R @ pos, np.array([0, 0, 0, 1.0])))
            else:                             # chamfer diamond: rotate the box itself about z
                boxes.append((he, R @ pos, _quat_mul(qz, q)))
    return boxes


def build_fixture(pc, spec: HoleSpec, rgba=(0.55, 0.55, 0.6, 1.0)) -> int:
    """Create the fixture as one static multibody; returns its body id."""
    boxes = fixture_boxes(spec)
    n = len(boxes)
    col = pc.createCollisionShapeArray(
        shapeTypes=[pc.GEOM_BOX] * n,
        halfExtents=[b[0].tolist() for b in boxes],
        collisionFramePositions=[b[1].tolist() for b in boxes],
        collisionFrameOrientations=[b[2].tolist() for b in boxes],
    )
    vis = pc.createVisualShapeArray(
        shapeTypes=[pc.GEOM_BOX] * n,
        halfExtents=[b[0].tolist() for b in boxes],
        visualFramePositions=[b[1].tolist() for b in boxes],
        visualFrameOrientations=[b[2].tolist() for b in boxes],
        rgbaColors=[list(rgba)] * n,
    )
    bid = pc.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                             basePosition=spec.bottom.tolist())
    pc.changeDynamics(bid, -1, lateralFriction=spec.friction)
    return bid


def contact_force(pc, robot_id, fixture_id) -> np.ndarray:
    """Total contact force (world frame) exerted by the fixture on the robot (ground truth)."""
    f = np.zeros(3)
    for cp in pc.getContactPoints(bodyA=robot_id, bodyB=fixture_id):
        n = np.array(cp[7])  # contact normal on B pointing towards A
        f += cp[9] * n
        f += cp[10] * np.array(cp[11]) + cp[12] * np.array(cp[13])  # friction
    return f
