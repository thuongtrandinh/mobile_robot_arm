#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import sys
import types


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--halo_drl_dir", required=True)
    parser.add_argument("--config", default="configs/mpc_rl.py")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--action_dim", type=int, default=9)
    parser.add_argument("--action_range", type=float, default=2.25)
    parser.add_argument("--v_pref", type=float, default=0.8)
    parser.add_argument("--robot_radius", type=float, default=0.25)
    return parser.parse_args()


def resolve_config_path(halo_drl_dir: str, config: str) -> str:
    if os.path.isabs(config):
        return config
    return os.path.join(halo_drl_dir, config)


def ensure_halo_runtime_stubs() -> None:
    if "ocp_planner_py" not in sys.modules:
        ocp_module = types.ModuleType("ocp_planner_py")

        class _DummyOcpPlanner:
            def __init__(self, *args, **kwargs):
                pass

        ocp_module.OcpPlanner = _DummyOcpPlanner
        sys.modules["ocp_planner_py"] = ocp_module

    if "rospy" not in sys.modules:
        rospy_module = types.ModuleType("rospy")

        class _DummyServiceProxy:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return None

        rospy_module.ServiceProxy = _DummyServiceProxy
        sys.modules["rospy"] = rospy_module

    if "ocp_planner" not in sys.modules:
        sys.modules["ocp_planner"] = types.ModuleType("ocp_planner")

    if "ocp_planner.srv" not in sys.modules:
        srv_module = types.ModuleType("ocp_planner.srv")

        class _DummySrv:
            pass

        class _DummyReq:
            pass

        srv_module.OcpLocalPlann = _DummySrv
        srv_module.OcpLocalPlannRequest = _DummyReq
        sys.modules["ocp_planner.srv"] = srv_module

    if "ocp_planner.msg" not in sys.modules:
        msg_module = types.ModuleType("ocp_planner.msg")

        class _DummyMsg:
            def __init__(self, *args, **kwargs):
                pass

        msg_module.HumanState = _DummyMsg
        msg_module.ObstacleState = _DummyMsg
        msg_module.PolyState = _DummyMsg
        msg_module.Point = _DummyMsg
        msg_module.WallState = _DummyMsg
        sys.modules["ocp_planner.msg"] = msg_module

    if "nav_msgs" not in sys.modules:
        sys.modules["nav_msgs"] = types.ModuleType("nav_msgs")
    if "nav_msgs.msg" not in sys.modules:
        nav_msg_module = types.ModuleType("nav_msgs.msg")

        class _DummyPath:
            pass

        nav_msg_module.Path = _DummyPath
        sys.modules["nav_msgs.msg"] = nav_msg_module

    if "geometry_msgs" not in sys.modules:
        sys.modules["geometry_msgs"] = types.ModuleType("geometry_msgs")
    if "geometry_msgs.msg" not in sys.modules:
        geo_msg_module = types.ModuleType("geometry_msgs.msg")

        class _DummyPoseStamped:
            pass

        geo_msg_module.PoseStamped = _DummyPoseStamped
        sys.modules["geometry_msgs.msg"] = geo_msg_module

    if "std_msgs" not in sys.modules:
        sys.modules["std_msgs"] = types.ModuleType("std_msgs")
    if "std_msgs.msg" not in sys.modules:
        std_msg_module = types.ModuleType("std_msgs.msg")

        class _DummyHeader:
            pass

        std_msg_module.Header = _DummyHeader
        sys.modules["std_msgs.msg"] = std_msg_module


def load_model(args):
    if args.halo_drl_dir not in sys.path:
        sys.path.insert(0, args.halo_drl_dir)

    ensure_halo_runtime_stubs()

    import gym
    import torch as th
    import crowd_sim.envs
    from algorithms.graph_ppo import joint_state_as_graph
    from algorithms.mpc_ppo import MpcPPO
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.state import FullState, JointState, ObstacleState, WallState, ObservableState
    from modules.policies import ExternalPolicy

    config_path = resolve_config_path(args.halo_drl_dir, args.config)
    if not os.path.isfile(config_path):
        raise RuntimeError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    env_config = config.EnvConfig(False)

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    env = gym.make("CrowdSim-v0", disable_env_checker=True)
    env.configure(env_config)
    env.set_phase(10)

    robot = Robot(env_config, "robot")
    robot.time_step = env.time_step
    robot.set_policy(ExternalPolicy())
    env.set_robot(robot)

    goal_range = (-args.action_range, args.action_range)
    env.num_actions_per_dim = args.action_dim
    env.goal_coord_range = goal_range
    env.action_space = gym.spaces.Discrete(args.action_dim * args.action_dim)
    env.use_AM = True
    env.use_action_mask = True
    env.use_PL = True

    model = MpcPPO(
        "GraphPolicy",
        env,
        device=device,
        action_dim=args.action_dim,
        goal_coord_range=goal_range,
        use_ros=False,
    )

    candidates = [args.model_path]
    if args.model_path.endswith(".zip"):
        candidates.append(args.model_path[:-4])
    else:
        candidates.append(f"{args.model_path}.zip")

    loaded = False
    for candidate in candidates:
        try:
            model.set_parameters(candidate, device=device)
            loaded = True
            break
        except Exception:
            pass

    if not loaded:
        raise RuntimeError(f"Cannot load model from candidates: {candidates}")

    return {
        "device": device,
        "model": model,
        "joint_state_as_graph": joint_state_as_graph,
        "FullState": FullState,
        "JointState": JointState,
        "ObstacleState": ObstacleState,
        "WallState": WallState,
        "ObservableState": ObservableState,
    }


def infer_sub_goal(payload, deps, v_pref, robot_radius):
    px = float(payload["px"])
    py = float(payload["py"])
    yaw = float(payload["yaw"])
    v_left = float(payload["v_left"])
    v_right = float(payload["v_right"])
    gx = float(payload["gx"])
    gy = float(payload["gy"])

    obstacle_tuples = payload.get("obstacles", [])
    wall_tuples = payload.get("walls", [])
    human_tuples = payload.get("humans", [])

    obstacles = [deps["ObstacleState"](float(x), float(y), float(r)) for x, y, r in obstacle_tuples]
    walls = [deps["WallState"](float(sx), float(sy), float(ex), float(ey)) for sx, sy, ex, ey in wall_tuples]

    humans = [
        deps["ObservableState"](float(hx), float(hy), float(hvx), float(hvy), float(hr))
        for hx, hy, hvx, hvy, hr in human_tuples
    ]

    full_state = deps["FullState"](
        px,
        py,
        v_left,
        v_right,
        robot_radius,
        gx,
        gy,
        v_pref,
        yaw,
    )

    observed_state = (humans, obstacles, walls, [])

    joint_state = deps["JointState"](full_state, observed_state)
    graph = deps["joint_state_as_graph"](joint_state, device=deps["device"])

    import torch as th
    with th.no_grad():
        action = deps["model"].policy.predict(graph, action_mask=None, deterministic=True).squeeze()
        action_idx = int(action.item() if hasattr(action, "item") else action)

    local_goal_x, local_goal_y = deps["model"].map_action_to_goal(action_idx)

    sub_goal_x = px + math.cos(yaw) * local_goal_x - math.sin(yaw) * local_goal_y
    sub_goal_y = py + math.sin(yaw) * local_goal_x + math.cos(yaw) * local_goal_y

    return sub_goal_x, sub_goal_y


def main():
    args = parse_args()
    deps = load_model(args)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            sub_goal_x, sub_goal_y = infer_sub_goal(payload, deps, args.v_pref, args.robot_radius)
            print(json.dumps({"sub_goal": [sub_goal_x, sub_goal_y]}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
