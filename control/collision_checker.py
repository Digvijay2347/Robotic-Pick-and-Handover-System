"""
Collision and Clearance Monitoring for Dual-Arm Manipulation.

Monitors:
- Minimum Euclidean distance between Arm 1 and Arm 2 links.
- Inter-arm proximity warnings.
- Unintended collision contact detection between robot links and static environment obstacles.
"""

from typing import List, Tuple, Dict
import numpy as np
import mujoco

class CollisionChecker:
    """Monitors collision state and safety clearances in the MuJoCo simulation."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, min_safe_distance: float = 0.05):
        self.model = model
        self.data = data
        self.min_safe_distance = min_safe_distance

        # Collect Arm 1 and Arm 2 geom IDs based on body ownership
        self.arm1_geom_ids = []
        self.arm2_geom_ids = []
        self.shelf_geom_ids = []
        self.bin_geom_ids = []

        for i in range(model.ngeom):
            b_id = model.geom_bodyid[i]
            b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or ""
            g_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""

            if "arm1_" in b_name or "arm1_" in g_name:
                self.arm1_geom_ids.append(i)
            elif "arm2_" in b_name or "arm2_" in g_name:
                self.arm2_geom_ids.append(i)
            elif "shelf" in b_name or "shelf" in g_name:
                self.shelf_geom_ids.append(i)
            elif "bin" in b_name or "bin" in g_name:
                self.bin_geom_ids.append(i)

    def _get_geom_name(self, geom_id: int) -> str:
        """Returns readable geom or parent body name, avoiding None."""
        g_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if g_name:
            return g_name
        b_id = self.model.geom_bodyid[geom_id]
        b_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b_id)
        return b_name or f"geom_{geom_id}"

    def check_inter_arm_collision(self) -> Tuple[bool, List[Tuple[str, str]]]:
        """
        Checks if there are active contacts between Arm 1 and Arm 2.
        Returns:
            Tuple: (is_colliding: bool, list_of_colliding_geom_pairs)
        """
        colliding_pairs = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.arm1_geom_ids and g2 in self.arm2_geom_ids) or \
               (g2 in self.arm1_geom_ids and g1 in self.arm2_geom_ids):
                name1 = self._get_geom_name(g1)
                name2 = self._get_geom_name(g2)
                colliding_pairs.append((name1, name2))

        return len(colliding_pairs) > 0, colliding_pairs

    def check_shelf_collision(self, arm: int) -> Tuple[bool, List[Tuple[str, str]]]:
        """Checks for unintended collision between arm links (excluding fingertips) and shelf."""
        arm_geoms = self.arm1_geom_ids if arm == 1 else self.arm2_geom_ids
        colliding_pairs = []

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in arm_geoms and g2 in self.shelf_geom_ids) or \
               (g2 in arm_geoms and g1 in self.shelf_geom_ids):
                name1 = self._get_geom_name(g1)
                name2 = self._get_geom_name(g2)
                # Ignore contacts from finger pads if they touch shelf floor gently
                if "pad" not in name1.lower() and "pad" not in name2.lower():
                    colliding_pairs.append((name1, name2))

        return len(colliding_pairs) > 0, colliding_pairs
