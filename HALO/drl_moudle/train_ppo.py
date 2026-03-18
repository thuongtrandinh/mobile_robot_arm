import argparse
import math
import os.path
import shutil
import sys

import gym
import importlib.util
import logging
import torch as th

from crowd_sim.envs.utils.robot import Robot
from modules.policies import ExternalPolicy

# from algorithms.graph_ppo import GraphPPO
from algorithms.mpc_ppo import MpcPPO
from modules.callbacks import CurriculumCallback, EvalCallback, CheckpointCallback, LastModelCheckpointCallback
from modules.evaluation import evaluate_policy
from modules.utils import get_linear_fn

def main(args):
    output_exists = os.path.exists(args.output_dir)
    is_resuming = args.resume and output_exists

    if output_exists and not args.resume:
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    output_config = os.path.join(args.output_dir, "config.py")
    # Resume: keep the existing config in output dir for consistency.
    if is_resuming and os.path.exists(output_config):
        args.config = output_config
    else:
        shutil.copy(args.config, output_config)
        args.config = output_config

    # configure logging
    log_file = os.path.join(args.output_dir, "output.log")
    file_handler = logging.FileHandler(log_file, mode='a' if args.resume else 'w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format='%(asctime)s, %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # load config module of simulation environment
    spec = importlib.util.spec_from_file_location("config", args.config)
    if spec is None:
        parser.error("Config file not found.")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env_config.reward.success_reward = args.re_arrival
    env_config.reward.goal_factor = args.goal_weight
    env_config.reward.collision_penalty = args.re_collision
    env_config.reward.discomfort_penalty_factor = args.safe_weight
    env_config.reward.re_theta = args.re_theta
    env_config.reward.re_rvo = args.re_rvo

    logging.info('Current random seed: {}'.format(args.randomseed))
    logging.info('Current safe_weight: {}'.format(args.safe_weight))
    logging.info('Current re_rvo_weight: {}'.format(args.re_rvo))
    logging.info('Current re_theta_weight: {}'.format(args.re_theta))
    logging.info('Current goal_weight: {}'.format(args.goal_weight))
    logging.info('Current re_collision: {}'.format(args.re_collision))
    logging.info('Current re_arrival: {}'.format(args.re_arrival))
    logging.info('Current n_steps: {}'.format(args.n_steps))
    logging.info('Current batch_size: {}'.format(args.batch_size))
    logging.info('Current n_epochs: {}'.format(args.n_epochs))
    device = th.device('cuda:0' if th.cuda.is_available() and args.gpu else 'cpu')
    logging.info('Using device: %s', device)

    env = gym.make("CrowdSim-v0", disable_env_checker=True)
    env.configure(env_config)
    env.set_phase(0)
    logging.info('FOV enabled: %s, FOV angle: %.1f deg, range: %.1f m',
                 env.use_fov, math.degrees(env.camera_fov_rad) * 2, env.camera_range)

    robot = Robot(env_config, "robot")
    robot.time_step = env.time_step
    robot.set_policy(ExternalPolicy())
    env.set_robot(robot)  # action space is determined here
    # action_range = 121
    # env.update_action_range(121)
    action_dim = args.action_dim
    goal_range = (-args.action_range, args.action_range)
    use_ros = args.use_ros
    # temp solution
    env.num_actions_per_dim = action_dim
    env.goal_coord_range = goal_range
    env.use_ros = use_ros
    env.action_space = gym.spaces.Discrete(action_dim * action_dim)
    env.use_AM = args.use_AM
    env.use_PL = args.use_PL
    env.PL_traj_length = args.PL_traj_length
    # Set trực tiếp lên crowd_sim (unwrapped) để _update_action_mask() dùng đúng giá trị
    env.unwrapped.num_actions_per_dim = action_dim
    env.unwrapped.goal_coord_range = goal_range
    env.unwrapped.use_AM = args.use_AM
    env.PL_traj_gamma = args.PL_traj_gamma

    lr_schedule = get_linear_fn(2.5e-4, 1.0e-4, 0.5)

    model = MpcPPO(
        "GraphPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=0.001,
        tensorboard_log=args.output_dir,
        seed=args.randomseed,
        device=device,
        action_dim=action_dim,
        goal_coord_range=goal_range,
        use_ros=use_ros,
    )

    if args.resume:
        last_model_path = os.path.join(args.output_dir, "last_model")
        best_model_path = os.path.join(args.output_dir, "best_model")
        last_episode_file = os.path.join(args.output_dir, "last_episode_num.txt")
        best_episode_file = os.path.join(args.output_dir, "episode_num.txt")

        resumed_ok = False
        if args.resume_from == "last":
            checkpoint_candidates = [(last_model_path, "last")]
        elif args.resume_from == "best":
            checkpoint_candidates = [(best_model_path, "best")]
        else:
            checkpoint_candidates = [(last_model_path, "last"), (best_model_path, "best")]

        for ckpt_path, ckpt_name in checkpoint_candidates:
            if not (os.path.exists(ckpt_path) or os.path.exists(ckpt_path + ".zip")):
                continue
            try:
                model.set_parameters(ckpt_path, device=device)
                logging.info("Resumed parameters from %s checkpoint: %s", ckpt_name, ckpt_path)
                resumed_ok = True
                break
            except Exception as e:
                logging.warning("Failed to load %s checkpoint at %s: %s", ckpt_name, ckpt_path, e)

        if not resumed_ok:
            logging.warning("Resume requested (%s) but no valid checkpoint found in %s. Training from scratch.",
                            args.resume_from, args.output_dir)

        resume_episode = args.start_episode
        if resume_episode == 0:
            if os.path.exists(last_episode_file):
                with open(last_episode_file, "r") as f:
                    resume_episode = int(f.read().strip())
            elif os.path.exists(best_episode_file):
                with open(best_episode_file, "r") as f:
                    resume_episode = int(f.read().strip())

        model._episode_num = resume_episode
        logging.info("Resumed from episode: %d", model._episode_num)
        ep = resume_episode
        if ep >= 19999:
            env.set_phase(3)
            logging.info("Restored curriculum: Phase 3 (ep >= 19999)")
        elif ep >= 11999:
            env.set_phase(2)
            logging.info("Restored curriculum: Phase 2 (ep >= 11999)")
        elif ep >= 3999:
            env.set_phase(1)
            logging.info("Restored curriculum: Phase 1 (ep >= 3999)")
        else:
            logging.info("Restored curriculum: Phase 0 (ep < 3999)")

    curriculum_callback = CurriculumCallback(verbose=0)
    eval_callback = EvalCallback(eval_env=env, n_eval_episodes=args.n_eval_episodes, eval_freq=args.eval_freq,
                                 best_model_save_path=args.output_dir, verbose=1)
    last_model_callback = LastModelCheckpointCallback(
        save_path=args.output_dir,
        save_freq_episodes=args.last_save_freq,
        verbose=1,
    )
    try:
        model.learn(int(args.total_timesteps), callback=[eval_callback, curriculum_callback, last_model_callback],
                    reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving last model checkpoint...")
    finally:
        last_model_path = os.path.join(args.output_dir, "last_model")
        tmp_last_model_path = os.path.join(args.output_dir, "last_model_tmp")
        model.save(tmp_last_model_path)
        tmp_zip = tmp_last_model_path + ".zip"
        final_zip = last_model_path + ".zip"
        if os.path.exists(tmp_zip):
            os.replace(tmp_zip, final_zip)
        if os.path.exists(tmp_last_model_path):
            os.remove(tmp_last_model_path)
        with open(os.path.join(args.output_dir, "last_episode_num.txt"), "w") as f:
            f.write(str(model._episode_num))
        logging.info("Saved last model to: %s.zip", last_model_path)
    # checkpt_callback = CheckpointCallback(save_freq=2000, save_path=args.checkpt_dir, name_prefix='PPO')
    # model.learn(int(5e6), callback=[eval_callback, curriculum_callback, checkpt_callback])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parse configuration file")
    parser.add_argument('--config', type=str, default='configs/mpc_rl.py')
    # parser.add_argument('--output_dir', type=str, default='data/output_eval_ipopt_theta_4_yaw_penalize')
    parser.add_argument('--output_dir', type=str, default='train_data/action_mask_eval_traj_3_4')
    
    # parser.add_argument('--output_dir', type=str, default='data/output_test')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_from', type=str, default='auto', choices=['auto', 'last', 'best'],
                        help='Checkpoint source for resume: auto tries last then best')
    parser.add_argument('--start_episode', type=int, default=0,
                        help='Episode number to resume from (use with --resume)')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=3)
    parser.add_argument('--safe_weight', type=float, default=0.5)
    parser.add_argument('--goal_weight', type=float, default=0.1)
    parser.add_argument('--re_collision', type=float, default=-0.25)
    parser.add_argument('--re_arrival', type=float, default=0.25)
    parser.add_argument('--re_rvo', type=float, default=0.01)
    parser.add_argument('--re_theta', type=float, default=0.01)
    parser.add_argument('--action_dim', type=int, default=9)
    parser.add_argument('--action_range', type=float, default=2.25)
    parser.add_argument('--use_ros', type=bool, default=False)
    parser.add_argument('--use_AM', type=bool, default=True, help="enable Action Mask")
    parser.add_argument('--use_PL', type=bool, default=True, help="enable Privileged Learning")
    parser.add_argument('--PL_traj_length', type=int, default=4)
    parser.add_argument('--PL_traj_gamma', type=float, default=0.9)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--total_timesteps', type=int, default=int(5e6))
    parser.add_argument('--eval_freq', type=int, default=500)
    parser.add_argument('--n_eval_episodes', type=int, default=100)
    parser.add_argument('--last_save_freq', type=int, default=50,
                        help='Auto-save last_model every N episodes')
    # parser.add_argument('--checkpt_dir', type=str, default='data/output_eval_ipopt_theta_4_yaw_penalize_pt1')


    sys_args = parser.parse_args()
    main(sys_args)
