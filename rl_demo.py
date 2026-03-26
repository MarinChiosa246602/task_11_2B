import numpy as np
import argparse
import os
import pybullet as p
from stable_baselines3 import PPO
from sim_class import Simulation
from root_detection import detect_roots, roots_to_world_coords


def normalize_position(position, ws_low, ws_high):
    return (2.0 * (position - ws_low) / (ws_high - ws_low) - 1.0).astype(np.float32)


def sweep_axis(sim, robot_key, target_np, axis, speed, max_sweep_steps=600):
    """Sweep along a single axis, find closest grid point."""
    state = sim.run([[0, 0, 0, 0]])
    pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)

    error = target_np - pos
    best_dist = np.linalg.norm(error)

    direction = np.sign(error[axis])
    if direction == 0:
        return best_dist

    vel = [0.0, 0.0, 0.0, 0.0]
    vel[axis] = float(direction * speed)

    increasing = 0
    for _ in range(max_sweep_steps):
        state = sim.run([vel], num_steps=1)
        pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        dist = np.linalg.norm(target_np - pos)

        if dist < best_dist:
            best_dist = dist
            increasing = 0
        else:
            increasing += 1

        if increasing > 8:
            rev = [0.0, 0.0, 0.0, 0.0]
            rev[axis] = float(-direction * speed * 0.3)
            for _ in range(increasing + 5):
                state = sim.run([rev], num_steps=1)
                pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
                d = np.linalg.norm(target_np - pos)
                if d < best_dist:
                    best_dist = d
            break

        if best_dist < 0.00001:
            break

    for _ in range(10):
        sim.run([[0, 0, 0, 0]])
    return best_dist


def final_polish(sim, robot_key, target_np):
    """Multi-pass axis sweep refinement."""
    TOLERANCE = 0.00001
    speeds = [0.003, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]

    state = sim.run([[0, 0, 0, 0]])
    pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
    best_dist = np.linalg.norm(target_np - pos)

    for speed in speeds:
        for axis in [0, 1, 2]:
            dist = sweep_axis(sim, robot_key, target_np, axis, speed)
            if dist < best_dist:
                best_dist = dist
            if best_dist < TOLERANCE:
                return best_dist

        # Diagonal sweeps
        for axes in [(0, 1), (0, 2), (1, 2)]:
            state = sim.run([[0, 0, 0, 0]])
            pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
            error = target_np - pos
            dist = np.linalg.norm(error)

            if dist < TOLERANCE:
                best_dist = min(best_dist, dist)
                break

            vel = [0.0, 0.0, 0.0, 0.0]
            for ax in axes:
                vel[ax] = float(np.sign(error[ax]) * speed)

            inc = 0
            for _ in range(500):
                state = sim.run([vel], num_steps=1)
                pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
                d = np.linalg.norm(target_np - pos)
                if d < best_dist:
                    best_dist = d
                    inc = 0
                else:
                    inc += 1
                if inc > 8 or d < TOLERANCE:
                    break

            for _ in range(10):
                sim.run([[0, 0, 0, 0]])

        if best_dist < TOLERANCE:
            return best_dist

    return best_dist


def move_to_target(sim, model, robot_key, target_np, ws_low, ws_high, max_steps=5000):
    """
    RL agent with MATCHING velocity scaling to training environment.
    
    Training used: velocity = action * 1.5, num_steps=1
    So we MUST use the same here for the model to work correctly.
    """
    TOLERANCE = 0.00001
    # Must match training environment's max_velocity
    MAX_VELOCITY = 1.5

    state = sim.run([[0, 0, 0, 0]])
    pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)

    best_dist = np.linalg.norm(target_np - pos)
    no_improve_count = 0

    for step in range(max_steps):
        error = target_np - pos
        dist = np.linalg.norm(error)

        if dist < best_dist - 0.000005:
            best_dist = dist
            no_improve_count = 0
        else:
            no_improve_count += 1

        if dist < TOLERANCE:
            print(f"    RL SUCCESS: {dist * 1000:.4f}mm at step {step}")
            return dist

        # Build observation exactly like training
        obs = np.concatenate([
            normalize_position(pos, ws_low, ws_high),
            normalize_position(target_np, ws_low, ws_high)
        ], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)

        # MATCH TRAINING: velocity = action * max_velocity
        velocity = action * MAX_VELOCITY

        # Send to sim with num_steps=1 (matches training)
        state = sim.run(
            [[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]],
            num_steps=1
        )
        pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)

        # Stall detection
        if no_improve_count >= 300:
            print(f"    RL stalled at {best_dist * 1000:.4f}mm (step {step}) — handing off")
            break

        if step > 0 and step % 500 == 0:
            print(f"    RL step {step}: dist={dist * 1000:.4f}mm  best={best_dist * 1000:.4f}mm")

    # Halt
    for _ in range(40):
        sim.run([[0, 0, 0, 0]])

    state = sim.run([[0, 0, 0, 0]])
    pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
    rl_final = np.linalg.norm(target_np - pos)
    print(f"    RL final: {rl_final * 1000:.4f}mm")

    if rl_final < TOLERANCE:
        return rl_final

    # Axis sweep polish
    polish_dist = final_polish(sim, robot_key, target_np)
    print(f"    After polish: {polish_dist * 1000:.4f}mm")

    return polish_dist


def dispense_single_drop(sim, settle_before=150, settle_after=200):
    for _ in range(settle_before):
        sim.run([[0, 0, 0, 0]])
    sim.run([[0, 0, 0, 1]])
    for _ in range(settle_after):
        sim.run([[0, 0, 0, 0]])


def retract_pipette(sim, robot_key, retract_height=0.22):
    for _ in range(300):
        state = sim.run([[0, 0, 0.02, 0]])
        pos = np.array(state[robot_key]["pipette_position"], dtype=np.float32)
        if pos[2] >= retract_height:
            break
    for _ in range(50):
        sim.run([[0, 0, 0, 0]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lr3e-4_b128_s4096.zip")
    args = parser.parse_args()

    model = PPO.load(args.model.replace(".zip", ""))

    ws_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
    ws_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)

    sim = Simulation(num_agents=1, render=True)
    state = sim.run([[0, 0, 0, 0]])
    robot_key = list(state.keys())[0]

    plate_image = sim.get_plate_image()
    print(f"\n--- Vision System: Processing {plate_image} ---")
    root_positions = detect_roots(plate_image)
    world_targets = roots_to_world_coords(root_positions, sim)

    print(f"\nPrecision Sequence: {len(world_targets)} targets\n")

    results = []
    for i, target in enumerate(world_targets):
        target_np = np.array(target, dtype=np.float32)
        target_np = np.clip(target_np, ws_low, ws_high)

        print(f"--- Root {i + 1}/{len(world_targets)} "
              f"[{target_np[0]:.5f}, {target_np[1]:.5f}, {target_np[2]:.5f}] ---")

        final_error = move_to_target(sim, model, robot_key, target_np, ws_low, ws_high)
        results.append(final_error)

        dispense_single_drop(sim)
        print(f"  Root {i + 1} done. Error: {final_error * 1000:.4f}mm\n")

        if i < len(world_targets) - 1:
            retract_pipette(sim, robot_key)

    errors_mm = [e * 1000 for e in results]
    print(f"{'=' * 50}")
    print(f"ALL {len(world_targets)} TARGETS PROCESSED")
    print(f"  Mean error:  {np.mean(errors_mm):.4f} mm")
    print(f"  Max error:   {np.max(errors_mm):.4f} mm")
    print(f"  Min error:   {np.min(errors_mm):.4f} mm")
    within_001 = sum(1 for e in errors_mm if e < 0.01)
    within_002 = sum(1 for e in errors_mm if e < 0.02)
    within_005 = sum(1 for e in errors_mm if e < 0.05)
    print(f"  Within 0.01mm: {within_001}/{len(errors_mm)}")
    print(f"  Within 0.02mm: {within_002}/{len(errors_mm)}")
    print(f"  Within 0.05mm: {within_005}/{len(errors_mm)}")
    print(f"{'=' * 50}")

    for _ in range(500):
        sim.run([[0, 0, 0, 0]])
    sim.close()


if __name__ == "__main__":
    main()