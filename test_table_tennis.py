import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.absolute()))

from common.ctrlcomp import PolicyOutput, StateAndCmd
from policy.table_tennis.TableTennis import TableTennis


def build_dummy_state(num_joints: int, seed: int) -> StateAndCmd:
    rng = np.random.default_rng(seed)
    state_cmd = StateAndCmd(num_joints)
    state_cmd.base_pos = rng.uniform(low=-0.1, high=0.1, size=3).astype(np.float32)
    state_cmd.base_lin_vel = rng.uniform(low=-0.2, high=0.2, size=3).astype(np.float32)
    state_cmd.ball_pos = np.array([3.5, -0.2, 1.0], dtype=np.float32)
    state_cmd.q = rng.uniform(low=-0.2, high=0.2, size=num_joints).astype(np.float32)
    state_cmd.dq = rng.uniform(low=-0.5, high=0.5, size=num_joints).astype(np.float32)
    state_cmd.ang_vel = rng.uniform(low=-0.2, high=0.2, size=3).astype(np.float32)
    state_cmd.gravity_ori = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    state_cmd.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return state_cmd


def main():
    parser = argparse.ArgumentParser(description="Minimal offline test for TableTennis policy.")
    parser.add_argument("--steps", type=int, default=1, help="Number of run() calls to execute.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dummy robot state.")
    args = parser.parse_args()

    num_joints = 29
    state_cmd = build_dummy_state(num_joints, args.seed)
    policy_output = PolicyOutput(num_joints)

    policy = TableTennis(state_cmd, policy_output)
    policy.enter()

    print(f"policy_available: {policy.policy_available}")
    if not policy.policy_available:
        print(f"init_error: {policy.init_error}")

    for step in range(args.steps):
        policy.run()
        print(f"step {step}:")
        print("  actions shape:", policy_output.actions.shape)
        print("  kps shape:", policy_output.kps.shape)
        print("  kds shape:", policy_output.kds.shape)
        print("  actions min/max:", float(np.min(policy_output.actions)), float(np.max(policy_output.actions)))
        print("  has nan:", bool(np.isnan(policy_output.actions).any()))

        # Perturb the state slightly between steps to mimic a changing robot state.
        state_cmd.base_pos = (state_cmd.base_pos + 0.005 * state_cmd.base_lin_vel).astype(np.float32)
        state_cmd.ball_pos = (state_cmd.ball_pos + np.array([-0.02, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
        state_cmd.q = (state_cmd.q + 0.01 * np.sin(step + np.arange(num_joints))).astype(np.float32)
        state_cmd.dq = (state_cmd.dq * 0.95).astype(np.float32)

    policy.exit()


if __name__ == "__main__":
    main()
