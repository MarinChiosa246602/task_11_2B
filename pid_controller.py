"""
PID Controller for Opentrons OT-2
"""

import numpy as np
import pybullet as p


class PIDController:
    def __init__(self, kp, ki, kd, dt, output_limits=None, integral_limit=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits
        self.integral_limit = integral_limit

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output


class FilteredPosition:
    """Exponential moving average filter for noisy position readings."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered = None

    def update(self, raw_pos):
        raw_pos = np.array(raw_pos, dtype=float)
        if self.filtered is None:
            self.filtered = raw_pos.copy()
        else:
            self.filtered = self.alpha * raw_pos + (1 - self.alpha) * self.filtered
        return self.filtered.copy()


def fix_baseplane(sim):
    """Fix the baseplane: lock joints, disable collision, hide it.

    The sim loads ot_2_simulation_v6.urdf twice:
      - baseplaneId (body 0) at origin — overlaps the robot
      - robotId (body 1) at [0,0,0.03] — the controllable robot

    We move it far away, lock its joints, and disable collision.
    """
    base_id = sim.baseplaneId

    p.resetBasePositionAndOrientation(base_id, [0, 0, -10], [0, 0, 0, 1])

    for joint_idx in range(3):
        p.setJointMotorControl2(
            base_id, joint_idx, p.VELOCITY_CONTROL,
            targetVelocity=0, force=10000
        )

    for robot_id in sim.robotIds:
        p.setCollisionFilterPair(base_id, robot_id, -1, -1, enableCollision=0)
        for i in range(3):
            for j in range(3):
                p.setCollisionFilterPair(base_id, robot_id, i, j, enableCollision=0)
            p.setCollisionFilterPair(base_id, robot_id, -1, j, enableCollision=0)
            p.setCollisionFilterPair(base_id, robot_id, i, -1, enableCollision=0)

    num_joints = p.getNumJoints(base_id)
    p.changeVisualShape(base_id, -1, rgbaColor=[0, 0, 0, 0])
    for link_idx in range(num_joints):
        p.changeVisualShape(base_id, link_idx, rgbaColor=[0, 0, 0, 0])


def teleport_pipette(sim, target_x, target_y, target_z):
    """Directly set joint states to place pipette at target position.

    From get_states():
        pipette_x = base_x - joint_0 + 0.073
        pipette_y = base_y - joint_1 + 0.0895
        pipette_z = base_z + joint_2 + 0.0895

    Therefore:
        joint_0 = base_x + 0.073 - target_x
        joint_1 = base_y + 0.0895 - target_y
        joint_2 = target_z - base_z - 0.0895
    """
    for robotId in sim.robotIds:
        base_pos = p.getBasePositionAndOrientation(robotId)[0]

        j0 = base_pos[0] + sim.pipette_offset[0] - target_x
        j1 = base_pos[1] + sim.pipette_offset[1] - target_y
        j2 = target_z - base_pos[2] - sim.pipette_offset[2]

        j0 = np.clip(j0, -0.18, 0.26)
        j1 = np.clip(j1, -0.13, 0.26)
        j2 = np.clip(j2, 0.05, 0.17)

        p.resetJointState(robotId, 0, targetValue=j0)
        p.resetJointState(robotId, 1, targetValue=j1)
        p.resetJointState(robotId, 2, targetValue=j2)


# Tuned gains
TUNED_GAINS = {
    'x': (3.0, 1.0, 0.05),
    'y': (3.0, 1.0, 0.05),
    'z': (3.0, 1.0, 0.05)
}

OUTPUT_LIMITS = {
    'x': (-0.2, 0.2),
    'y': (-0.2, 0.2),
    'z': (-0.2, 0.2)
}

SUBSTEPS = 10
