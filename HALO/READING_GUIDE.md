# 📚 Hướng dẫn Đọc Source Code HALO

**HALO - Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation**

> 🎯 **Mục tiêu**: Hướng dẫn bạn đọc và hiểu source code HALO từ cơ bản đến nâng cao, tập trung vào quá trình training model.

---

## 📖 Mục lục

1. [Tổng quan về HALO](#1-tổng-quan-về-halo)
2. [Kiến trúc Hệ thống](#2-kiến-trúc-hệ-thống)
3. [Thứ tự Đọc Code Đề xuất](#3-thứ-tự-đọc-code-đề-xuất)
4. [Chi tiết từng Module](#4-chi-tiết-từng-module)
5. [Flow Training](#5-flow-training)
6. [Các Kỹ thuật Chính](#6-các-kỹ-thuật-chính)
7. [Tips Debugging](#7-tips-debugging)

---

## 1. Tổng quan về HALO

### 1.1 HALO là gì?

HALO là một hệ thống điều hướng robot trong môi trường động có đông người sử dụng **hybrid approach**:

- **Deep Reinforcement Learning (PPO)** với Graph Neural Network để học chiến lược high-level
- **Model Predictive Control (MPC)** để xử lý ràng buộc động học và đảm bảo an toàn

### 1.2 Ba Kỹ thuật Chính

| Kỹ thuật | Mục đích | Implementation |
|----------|----------|----------------|
| **Spatio-Temporal Path Search** | Tạo reference trajectory tốt hơn | `lookahead.cc/h`, `astar.cc/h` |
| **Action Masking (AM)** | Loại bỏ actions không khả thi | `MaskedActorCriticPolicy` |
| **Privileged Learning (PL)** | Học từ expert MPC trajectory | `collect_rollouts()` trong `mpc_ppo.py` |

### 1.3 Paper Reference

```bibtex
@misc{liu2025hierarchicallearningenhancedmpcsafe,
      title={Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation with Heterogeneous Constraints}, 
      author={Huajian Liu and Yixuan Feng and Wei Dong and Kunpeng Fan and Chao Wang and Yongzhuo Gao},
      year={2025},
      eprint={2506.09859},
      archivePrefix={arXiv},
}
```

---

## 2. Kiến trúc Hệ thống

### 2.1 Luồng xử lý chính

```
┌──────────────────────────────────────────────────────────────┐
│                  HALO SYSTEM FLOW                            │
└──────────────────────────────────────────────────────────────┘

    Input: Robot State + Humans + Obstacles + Goal
                      │
                      ▼
    ┌─────────────────────────────────────────────────────┐
    │  1. Observation Processing                          │
    │     - Convert to JointState                         │
    │     - Create Graph: Joint State → DGL Graph         │
    │       * Robot node: [px, py, vx, vy, r, gx, gy]    │
    │       * Human nodes: [px, py, vx, vy, r]           │
    │       * Obstacle nodes: geometric info              │
    └────────────────────┬────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  2. Graph Neural Network (GNN Policy)               │
    │     - Graph Extractor: process hetero graph         │
    │     - Actor Network: output action distribution     │
    │     - Critic Network: estimate state value          │
    │     → Output: Action probabilities (grid of goals)  │
    └────────────────────┬────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  3. Action Masking (AM)                             │
    │     - Call MPC solver for each candidate goal       │
    │     - Mark as INVALID if MPC fails                  │
    │     - Mask invalid actions in policy                │
    └────────────────────┬────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  4. Action Selection                                │
    │     - Sample from masked distribution               │
    │     → Selected local goal (x, y)                    │
    └────────────────────┬────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  5. Spatio-Temporal Path Search                     │
    │     - A* with time dimension                        │
    │     - Consider kinematic constraints                │
    │     - Avoid predicted human positions               │
    │     → Reference trajectory                          │
    └────────────────────┬────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  6. MPC Optimization                                │
    │     - Solve with IPOPT or FORCES PRO                │
    │     - Input: Reference trajectory                   │
    │     - Enforce: kinematic, safety constraints        │
    │     → Output: (left_acc, right_acc)                 │
    └────────────────────┬────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  7. Apply Action & Get Reward                       │
    │     Reward = goal_reward + safety_penalty           │
    │            + collision_penalty + rvo_reward         │
    └────────────────────┬────────────────────────────────┘
                         │
                         ▼
                 Store in Rollout Buffer
                         │
                         ▼
                (After n_steps collected)
                         │
                         ▼
    ┌─────────────────────────────────────────────────────┐
    │  8. PPO Training                                    │
    │     - Compute GAE advantages                        │
    │     - Update Actor (policy loss + entropy)          │
    │     - Update Critic (value loss)                    │
    │     - Clip gradients                                │
    └─────────────────────────────────────────────────────┘
```

### 2.2 Cấu trúc thư mục dự án

```
HALO/
├── drl_moudle/                      # 🔥 Module Training (ĐÂY LÀ PHẦN CHÍNH)
│   ├── train_ppo.py                 # ⭐ Script training chính - BẮT ĐẦU TỪ ĐÂY
│   ├── eval_ppo.py                  # Script evaluation & testing
│   │
│   ├── algorithms/                  # 🧠 Core training algorithms
│   │   ├── graph_ppo.py            # Base PPO với Graph NN
│   │   └── mpc_ppo.py              # MPC-integrated PPO (extends graph_ppo)
│   │
│   ├── configs/                     # ⚙️ Cấu hình training
│   │   ├── config.py               # Base config (env, reward, sim)
│   │   └── mpc_rl.py               # MPC-RL specific config
│   │
│   ├── modules/                     # 🎯 Core implementations
│   │   ├── graph_ppo_core.py       # GraphActorCriticPolicy, GraphRolloutBuffer
│   │   ├── mpc_ppo_core.py         # MaskedActorCriticPolicy, MaskedRolloutBuffer
│   │   ├── policies.py             # Policy base classes
│   │   ├── gnn_models.py           # Graph Neural Network models
│   │   ├── torch_layers.py         # Neural network layers (MLP, GNN)
│   │   ├── distributions.py        # Action distributions
│   │   ├── callbacks.py            # Training callbacks
│   │   ├── evaluation.py           # Evaluation utilities
│   │   └── utils.py                # Helper functions
│   │
│   ├── crowd_sim/                   # 🌍 Simulation Environment
│   │   └── envs/
│   │       ├── crowd_sim.py        # Main Gym environment
│   │       ├── policy/             # Human policies (ORCA, Social Force)
│   │       └── utils/              # State, Action, Agent classes
│   │
│   └── data/                        # 💾 Trained models
│
└── src/ocp_planner/                 # 🚀 MPC Solver (C++)
    ├── src/
    │   ├── planner.cc/h            # Main planner class
    │   ├── mpc.cc/h                # MPC formulation & solver
    │   ├── astar.cc/h              # A* path search
    │   ├── lookahead.cc/h          # Spatio-temporal search
    │   ├── smooth.cc/h             # Path smoothing
    │   └── ocp_bind.cc             # Python bindings
    └── msg/                        # ROS message definitions
```

---

## 3. Thứ tự Đọc Code Đề xuất

### 🎯 Roadmap cho người mới

#### **Level 1: Hiểu Flow Tổng thể (30-60 phút)**

1. **[train_ppo.py](drl_moudle/train_ppo.py)** ⭐ **BẮT ĐẦU NGAY TỪ ĐÂY**
   - Đây là entry point chính
   - Đọc hàm `main()` từ đầu đến cuối
   - Hiểu flow: Load config → Create env → Create model → Train
   
   **📝 Checklist khi đọc:**
   - [ ] Parse arguments (hyperparameters)
   - [ ] Load config file
   - [ ] Setup environment (`gym.make("CrowdSim-v0")`)
   - [ ] Create robot with external policy
   - [ ] Initialize `MpcPPO` model
   - [ ] Setup callbacks (Curriculum, Eval, Checkpoint)
   - [ ] Call `model.learn()`

2. **[configs/config.py](drl_moudle/configs/config.py)** + **[configs/mpc_rl.py](drl_moudle/configs/mpc_rl.py)**
   - Hiểu các hyperparameters
   - Reward function configuration
   - Environment settings
   
   **📝 Key configs:**
   - `time_step = 0.25` (250ms per step)
   - `human_num`, `obstacle_num`, `wall_num`
   - Reward weights: `success_reward`, `collision_penalty`, `goal_factor`, `discomfort_penalty_factor`

#### **Level 2: Hiểu Training Algorithm (1-2 giờ)**

3. **[algorithms/graph_ppo.py](drl_moudle/algorithms/graph_ppo.py)** 
   - Đây là base class cho training algorithm
   - **ĐỌC THEO THỨ TỰ:**
   
   **a) Class structure & init:**
   ```python
   class GraphPPO(BaseAlgorithm):
       def __init__(...):
           # Hyperparameters: n_steps, batch_size, gamma, gae_lambda
           # Clip range, entropy coefficient, value coefficient
   ```
   
   **b) Main training loop:**
   ```python
   def learn(...):
       while self.num_timesteps < total_timesteps:
           # 1. Collect rollouts (interact with env)
           self.collect_rollouts(...)
           # 2. Update policy (PPO update)
           self.train()
   ```
   
   **c) Rollout collection:**
   ```python
   def collect_rollouts(...):
       # Loop n_steps times:
       #   - Get action from policy
       #   - Execute in environment
       #   - Store transition in buffer
       #   - Bootstrap value if timeout
   ```
   
   **d) PPO update:**
   ```python
   def train(...):
       for epoch in n_epochs:
           for batch in rollout_buffer:
               # Compute advantages
               # Policy loss (clipped)
               # Value loss
               # Entropy loss
               # Backprop & update
   ```

4. **[algorithms/mpc_ppo.py](drl_moudle/algorithms/mpc_ppo.py)** 
   - Extends GraphPPO với MPC integration
   - **KEY ADDITIONS:**
   
   **a) MPC planner initialization:**
   ```python
   def __init__(...):
       super().__init__(...)
       # Initialize ocp_planner (C++ module)
       self.ocp_planner = ocp_planner_py.Planner(...)
   ```
   
   **b) Action masking trong collect_rollouts:**
   ```python
   def collect_rollouts(...):
       # For each candidate goal:
       #   - Call MPC solver
       #   - If MPC fails → mask = 0 (invalid)
       #   - If MPC succeeds → mask = 1 (valid)
   ```
   
   **c) Privileged learning (PL):**
   ```python
   # Store expert trajectory information
   # Use in reward shaping or auxiliary loss
   ```

#### **Level 3: Hiểu Network Architecture (1-2 giờ)**

5. **[modules/graph_ppo_core.py](drl_moudle/modules/graph_ppo_core.py)**
   - `GraphActorCriticPolicy`: Main policy class
   - `GraphRolloutBuffer`: Experience buffer
   
   **📝 Actor-Critic structure:**
   ```python
   GraphActorCriticPolicy:
       ├── features_extractor: GraphExtractor (GNN)
       │   └── Process hetero graph (robot, human, obstacle nodes)
       ├── action_net (Actor): MLP → action logits
       └── value_net (Critic): MLP → state value
   ```

6. **[modules/mpc_ppo_core.py](drl_moudle/modules/mpc_ppo_core.py)**
   - `MaskedActorCriticPolicy`: Policy với action masking
   - `MaskedRolloutBuffer`: Buffer lưu thêm action masks
   
   **Key difference:**
   ```python
   def forward(..., action_masks):
       # Standard forward
       distribution = super().forward(...)
       # Apply mask
       distribution.apply_masking(action_masks)
       return distribution
   ```

7. **[modules/gnn_models.py](drl_moudle/modules/gnn_models.py)** + **[modules/torch_layers.py](drl_moudle/modules/torch_layers.py)**
   - Graph Neural Network implementation
   - GraphExtractor: converts graph to features
   
   **GNN Architecture:**
   ```
   Input: DGL Hetero Graph
      ↓
   Node Embedding (separate for each node type)
      ↓
   Graph Convolution Layers (message passing)
      ↓
   Readout (aggregate to graph-level features)
      ↓
   Output: Feature vector
   ```

#### **Level 4: Hiểu Environment (1-2 giờ)**

8. **[crowd_sim/envs/crowd_sim.py](drl_moudle/crowd_sim/envs/crowd_sim.py)**
   - Main Gym environment
   - **ĐỌC THEO METHODS:**
   
   **a) Initialization:**
   ```python
   def configure(self, config):
       # Setup time_step, robot_sensor_range
       # Setup reward function
   ```
   
   **b) Reset:**
   ```python
   def reset(self, phase):
       # Generate new scenario (humans, obstacles, walls)
       # Reset robot position
       # Return initial observation
   ```
   
   **c) Step (QUAN TRỌNG NHẤT):**
   ```python
   def step(self, action: ActionDiff):
       # 1. Convert action (local goal) to trajectory via MPC
       # 2. Update robot state
       # 3. Update humans (ORCA policy)
       # 4. Check collisions, goal reaching
       # 5. Compute reward
       # 6. Return (obs, reward, done, info)
   ```
   
   **d) Reward computation:**
   ```python
   # success_reward: reach goal
   # collision_penalty: hit human/obstacle
   # goal_factor * distance_to_goal: progress reward
   # discomfort_penalty: too close to humans
   # rvo_reward: follow RVO rule
   # theta_reward: heading alignment
   ```

9. **[crowd_sim/envs/utils/](drl_moudle/crowd_sim/envs/utils/)**
   - `state.py`: JointState, FullState, ObservableState
   - `action.py`: ActionDiff (left_acc, right_acc)
   - `robot.py`, `human.py`: Agent classes
   - `utils.py`: Helper functions

#### **Level 5: MPC Solver (Optional, nếu quan tâm low-level)**

10. **[src/ocp_planner/src/planner.cc](src/ocp_planner/src/planner.cc)**
    - Main interface for MPC planning
    
11. **[src/ocp_planner/src/mpc.cc](src/ocp_planner/src/mpc.cc)**
    - MPC formulation (cost function, constraints)
    - IPOPT solver integration

12. **[src/ocp_planner/src/lookahead.cc](src/ocp_planner/src/lookahead.cc)**
    - Spatio-temporal path search
    - A* variant với time dimension

---

## 4. Chi tiết từng Module

### 4.1 Module: train_ppo.py

**Purpose:** Training script chính

**Key Components:**

```python
def main(args):
    # 1. Setup logging & output directory
    # 2. Load config module
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    
    # 3. Configure environment
    env_config = config.EnvConfig(args.debug)
    env_config.reward.success_reward = args.re_arrival
    env_config.reward.collision_penalty = args.re_collision
    # ... more reward weights
    
    # 4. Create Gym environment
    env = gym.make("CrowdSim-v0")
    env.configure(env_config)
    
    # 5. Setup robot with external policy
    robot = Robot(env_config, "robot")
    robot.set_policy(ExternalPolicy())  # Policy from RL
    env.set_robot(robot)
    
    # 6. Create PPO model
    model = MpcPPO(
        "GraphPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=2048,        # Collect 2048 samples before update
        batch_size=64,       # Mini-batch size
        ent_coef=0.001,      # Entropy coefficient
        device=device,
        action_dim=args.action_dim,  # 9x9 grid → 81 actions
        goal_coord_range=goal_range, # [-2.25, 2.25]
    )
    
    # 7. Setup callbacks
    curriculum_callback = CurriculumCallback()  # Increase difficulty
    eval_callback = EvalCallback(
        eval_freq=500,      # Evaluate every 500 steps
        n_eval_episodes=100 # 100 episodes per evaluation
    )
    
    # 8. Start training
    model.learn(
        total_timesteps=int(5e6),  # 5 million steps
        callback=[eval_callback, curriculum_callback]
    )
```

**Command-line Arguments:**

```bash
python train_ppo.py \
    --config configs/mpc_rl.py \
    --output_dir train_data/my_experiment \
    --randomseed 3 \
    --safe_weight 0.5 \
    --goal_weight 0.1 \
    --re_collision -0.25 \
    --re_arrival 0.25 \
    --action_dim 9 \
    --action_range 2.25 \
    --use_AM True \       # Enable Action Mask
    --use_PL True \       # Enable Privileged Learning
    --PL_traj_length 4    # PL trajectory length
```

---

### 4.2 Module: algorithms/graph_ppo.py

**Purpose:** Base PPO implementation với Graph NN

**Class Hierarchy:**
```
BaseAlgorithm (modules/base_class.py)
    └── GraphPPO
            └── MpcPPO (algorithms/mpc_ppo.py)
```

**Key Methods:**

#### `__init__()`
```python
def __init__(
    self,
    policy,              # "GraphPolicy" → GraphActorCriticPolicy
    env,                 # Gym environment
    learning_rate,       # Can be float or Schedule
    n_steps=2048,        # Rollout buffer size
    batch_size=64,       # Mini-batch size
    n_epochs=10,         # Epochs per update
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # GAE lambda
    clip_range=0.2,      # PPO clip epsilon
    ent_coef=0.0,        # Entropy coefficient
    vf_coef=0.5,         # Value loss coefficient
    max_grad_norm=0.5,   # Gradient clipping
):
    # Store hyperparameters
    # Initialize rollout buffer
    # Create policy network
```

#### `learn(total_timesteps, callback)`
**Main training loop:**

```python
def learn(self, total_timesteps, callback):
    # Setup: reset buffers, initialize observation
    iteration = 0
    
    while self.num_timesteps < total_timesteps:
        # PHASE 1: COLLECT ROLLOUTS
        continue_training = self.collect_rollouts(
            self.env, 
            callback, 
            self.rollout_buffer, 
            n_rollout_steps=self.n_steps  # 2048 steps
        )
        
        if not continue_training:
            break
        
        iteration += 1
        
        # Update learning rate, clip range (linear decay)
        self._update_current_progress_remaining(...)
        
        # Log episode statistics
        if iteration % log_interval == 0:
            self.logger.record("rollout/ep_ret_mean", ...)
            self.logger.record("rollout/ep_len_mean", ...)
        
        # PHASE 2: UPDATE POLICY
        self.train()
    
    return self
```

#### `collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)`
**Interaction với environment:**

```python
def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
    n_steps = 0
    rollout_buffer.reset()
    
    while n_steps < n_rollout_steps:
        # GET ACTION from policy
        with th.no_grad():
            actions, values, log_probs = self.policy(self._last_state)
        
        actions = actions.cpu().numpy()
        clipped_action = actions.squeeze()
        
        # EXECUTE ACTION in environment
        if isinstance(self.action_space, gym.spaces.Discrete):
            left_acc, right_acc = map_action_to_accel(clipped_action)
            new_ob, reward, done, info = env.step(
                ActionDiff(left_acc, right_acc)
            )
        
        self.num_timesteps += 1
        
        # Track returns for logging
        self.rewards.append(float(reward))
        
        if done:
            # Calculate discounted returns
            returns = [sum([gamma^t * r for t, r in enumerate(rewards[step:])]) 
                      for step in range(len(rewards))]
            
            ep_info = {"r": mean(returns), "l": len(rewards)}
            self.ep_info_buffer.append(ep_info)
            
            # Handle timeout: bootstrap with value function
            if isinstance(info, Timeout):
                terminal_state = joint_state_as_graph(
                    JointState(new_robot_state, new_ob)
                )
                terminal_value = self.policy.predict_values(terminal_state)
                reward += gamma * terminal_value
            
            # Reset environment
            new_ob = env.reset(phase="train")
        
        # STORE TRANSITION
        rollout_buffer.add(
            self._last_state,
            actions,
            reward,
            self._last_episode_start,
            values,
            log_probs,
        )
        
        # Update state
        self._last_state = joint_state_as_graph(
            JointState(new_robot_state, new_ob)
        )
        self._last_episode_start = done
        n_steps += 1
    
    # Compute GAE advantages
    with th.no_grad():
        values = self.policy.predict_values(self._last_state)
    
    rollout_buffer.compute_returns_and_advantage(
        last_values=values, 
        dones=done
    )
    
    return True
```

#### `train()`
**PPO policy update:**

```python
def train(self):
    self.policy.set_training_mode(True)
    
    # Get current clip range (linearly decayed)
    clip_range = self.clip_range(self._current_progress_remaining)
    
    # Train for n_epochs (default 10)
    for epoch in range(self.n_epochs):
        approx_kl_divs = []
        
        # Iterate over mini-batches
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            actions = rollout_data.actions
            
            # FORWARD PASS: get new values, log_probs, entropy
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, 
                actions
            )
            values = values.flatten()
            
            # NORMALIZE ADVANTAGES
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # POLICY LOSS (PPO clipped objective)
            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(
                ratio, 
                1 - clip_range,  # 0.8
                1 + clip_range   # 1.2
            )
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            
            # VALUE LOSS (MSE)
            if self.clip_range_vf is None:
                values_pred = values
            else:
                # Clip value function (optional)
                values_pred = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values,
                    -clip_range_vf,
                    clip_range_vf
                )
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            
            # ENTROPY LOSS (for exploration)
            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)
            
            # TOTAL LOSS
            loss = (policy_loss 
                   + self.ent_coef * entropy_loss      # 0.001
                   + self.vf_coef * value_loss)        # 0.5
            
            # Early stopping check
            with th.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                approx_kl_divs.append(approx_kl_div)
            
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # BACKPROPAGATION
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                self.max_grad_norm  # 0.5
            )
            self.policy.optimizer.step()
        
        self._n_updates += 1
    
    # Logging
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
```

---

### 4.3 Module: algorithms/mpc_ppo.py

**Purpose:** MPC-integrated PPO với Action Masking và Privileged Learning

**Key Additions to GraphPPO:**

#### 1. MPC Planner Integration

```python
class MpcPPO(GraphPPO):
    def __init__(self, ..., action_dim, goal_coord_range, use_ros):
        super().__init__(...)
        
        # Create action grid (e.g., 9x9 = 81 goals)
        self.action_dim = action_dim
        self.goal_coord_range = goal_coord_range
        self.goals_np = self._create_goal_grid()
        
        # Initialize MPC planner (C++ module)
        if not use_ros:
            self.ocp_planner = ocp_planner_py.Planner(...)
            self.ocp_planner.set_solver(solver_name="ipopt")
        else:
            # Use ROS service for MPC
            rospy.init_node('mpc_ppo_trainer')
            self.ocp_client = rospy.ServiceProxy(
                '/ocp_local_plann', 
                OcpLocalPlann
            )

def _create_goal_grid(self):
    """Create grid of local goals"""
    x_goals = np.linspace(
        self.goal_coord_range[0],  # -2.25
        self.goal_coord_range[1],  # +2.25
        self.action_dim            # 9
    )
    y_goals = np.linspace(...)
    
    goals = []
    for x in x_goals:
        for y in y_goals:
            goals.append([x, y])
    
    return np.array(goals)  # Shape: (81, 2)
```

#### 2. Action Masking in collect_rollouts()

```python
def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
    n_steps = 0
    rollout_buffer.reset()
    
    while n_steps < n_rollout_steps:
        # STEP 1: Generate action mask
        action_masks = self._generate_action_masks(env)
        # action_masks shape: (81,) with 0/1 values
        
        # STEP 2: Get action from policy (with mask)
        with th.no_grad():
            actions, values, log_probs = self.policy(
                self._last_state,
                action_masks=th.tensor(action_masks).to(self.device)
            )
        
        # STEP 3: Execute selected action
        selected_action = actions.cpu().numpy().squeeze()
        selected_goal = self.goals_np[selected_action]
        
        # Call MPC to get control
        left_acc, right_acc = self._mpc_solve(env, selected_goal)
        
        new_ob, reward, done, info = env.step(
            ActionDiff(left_acc, right_acc)
        )
        
        # STEP 4: Privileged Learning (optional)
        if env.use_PL:
            # Get expert trajectory from MPC
            expert_traj = self._get_expert_trajectory(env)
            # Store for auxiliary loss (implementation varies)
        
        # STEP 5: Store transition (with mask)
        rollout_buffer.add(
            self._last_state,
            actions,
            reward,
            self._last_episode_start,
            values,
            log_probs,
            action_masks,  # NEW: store mask
        )
        
        # Update
        self._last_state = joint_state_as_graph(...)
        n_steps += 1
    
    return True
```

#### 3. Action Mask Generation

```python
def _generate_action_masks(self, env):
    """
    Generate action mask by testing MPC feasibility
    
    Returns:
        np.ndarray: shape (num_actions,), 1 = valid, 0 = invalid
    """
    robot_state = env.robot.get_full_state()
    humans = env.humans
    obstacles = env.obstacles
    walls = env.walls
    
    action_masks = np.zeros(self.action_dim ** 2, dtype=np.float32)
    
    # Test each candidate goal
    for i, goal in enumerate(self.goals_np):
        # Convert to global frame
        global_goal = self._local_to_global(robot_state, goal)
        
        # Call MPC solver
        success = self._test_mpc_feasibility(
            env, 
            robot_state, 
            global_goal,
            humans,
            obstacles,
            walls
        )
        
        if success:
            action_masks[i] = 1.0  # Valid action
        else:
            action_masks[i] = 0.0  # Invalid action
    
    # Ensure at least one valid action
    if action_masks.sum() == 0:
        # Fallback: mark closest to current heading as valid
        action_masks[self.action_dim ** 2 // 2] = 1.0
    
    return action_masks

def _test_mpc_feasibility(self, env, robot_state, goal, humans, obstacles, walls):
    """Test if MPC can find a solution for this goal"""
    try:
        # Setup MPC problem
        self.ocp_planner.set_robot_state(robot_state)
        self.ocp_planner.set_goal(goal)
        self.ocp_planner.set_obstacles(humans + obstacles + walls)
        
        # Solve
        result = self.ocp_planner.solve()
        
        return result.success
    except:
        return False
```

#### 4. Privileged Learning

```python
def _get_expert_trajectory(self, env):
    """
    Get expert MPC trajectory for privileged learning
    
    This trajectory uses perfect information and serves as a "teacher"
    """
    # Solve MPC with full horizon
    expert_result = self.ocp_planner.solve_full_horizon()
    
    # Extract trajectory
    expert_traj = {
        'positions': expert_result.positions,  # (T, 2)
        'velocities': expert_result.velocities,  # (T, 2)
        'controls': expert_result.controls,  # (T, 2)
    }
    
    return expert_traj

# Can be used for:
# 1. Auxiliary loss: minimize distance to expert trajectory
# 2. Reward shaping: bonus for following expert
# 3. Imitation learning component
```

---

### 4.4 Module: modules/graph_ppo_core.py

**Purpose:** Policy và rollout buffer implementation

#### GraphActorCriticPolicy

```python
class GraphActorCriticPolicy(nn.Module):
    """
    Policy network for Graph PPO
    
    Architecture:
        Input: DGL Hetero Graph
          ↓
        GraphExtractor (GNN)
          ↓ 
        Features (latent_dim)
          ├→ Actor MLP → Action distribution
          └→ Critic MLP → State value
    """
    
    def __init__(
        self,
        action_space,
        lr_schedule,
        net_arch=None,           # [dict(pi=[64, 64], vf=[64, 64])]
        activation_fn=nn.Tanh,
        features_extractor_kwargs=None,
    ):
        super().__init__()
        
        # Feature extractor (GNN)
        self.features_extractor = GraphExtractor(
            **features_extractor_kwargs
        )
        
        latent_dim = self.features_extractor.features_dim
        
        # Actor network
        self.action_net = create_mlp(
            input_dim=latent_dim,
            output_dim=action_space.n,
            net_arch=[64, 64],
            activation_fn=activation_fn
        )
        
        # Critic network
        self.value_net = create_mlp(
            input_dim=latent_dim,
            output_dim=1,
            net_arch=[64, 64],
            activation_fn=activation_fn
        )
        
        # Action distribution
        self.action_dist = CategoricalDistribution(action_space.n)
        
        # Optimizer
        self.optimizer = th.optim.Adam(
            self.parameters(), 
            lr=lr_schedule(1.0)
        )
    
    def forward(self, obs: dgl.DGLHeteroGraph):
        """
        Forward pass
        
        Args:
            obs: DGL hetero graph
            
        Returns:
            (actions, values, log_probs)
        """
        # Extract features via GNN
        features = self.features_extractor(obs)  # (batch, latent_dim)
        
        # Actor: get action distribution
        action_logits = self.action_net(features)  # (batch, num_actions)
        distribution = self.action_dist.proba_distribution(action_logits)
        
        # Sample action
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        
        # Critic: get state value
        values = self.value_net(features)  # (batch, 1)
        
        return actions, values, log_probs
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions (used in train())
        
        Returns:
            (values, log_probs, entropy)
        """
        features = self.features_extractor(obs)
        
        # Actor
        action_logits = self.action_net(features)
        distribution = self.action_dist.proba_distribution(action_logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Critic
        values = self.value_net(features)
        
        return values, log_probs, entropy
    
    def predict_values(self, obs):
        """Get state value (used for bootstrapping)"""
        features = self.features_extractor(obs)
        return self.value_net(features)
```

#### GraphExtractor (GNN)

```python
class GraphExtractor(nn.Module):
    """
    Graph Neural Network feature extractor
    
    Process heterogeneous graph:
        - Robot node
        - Human nodes
        - Obstacle nodes
        - Wall nodes
    """
    
    def __init__(
        self,
        node_embedding_size=32,
        num_gnn_layers=2,
        output_size=128,
    ):
        super().__init__()
        
        # Node embedding layers (separate for each node type)
        self.robot_embedding = nn.Linear(7, node_embedding_size)
        # Robot features: [px, py, vx, vy, radius, goal_x, goal_y]
        
        self.human_embedding = nn.Linear(5, node_embedding_size)
        # Human features: [px, py, vx, vy, radius]
        
        self.obstacle_embedding = nn.Linear(4, node_embedding_size)
        # Obstacle features: [px, py, radius, type]
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(node_embedding_size)
            for _ in range(num_gnn_layers)
        ])
        
        # Readout/aggregation
        self.readout = nn.Linear(node_embedding_size, output_size)
        
        self.features_dim = output_size
    
    def forward(self, graph: dgl.DGLHeteroGraph):
        """
        Args:
            graph: DGL hetero graph with node types:
                   'robot', 'human', 'obstacle', 'wall'
        
        Returns:
            features: (batch, features_dim)
        """
        # Node embedding
        robot_feat = self.robot_embedding(graph.nodes['robot'].data['feat'])
        human_feat = self.human_embedding(graph.nodes['human'].data['feat'])
        obstacle_feat = self.obstacle_embedding(graph.nodes['obstacle'].data['feat'])
        
        # Update graph
        graph.nodes['robot'].data['h'] = robot_feat
        graph.nodes['human'].data['h'] = human_feat
        graph.nodes['obstacle'].data['h'] = obstacle_feat
        
        # Graph convolution (message passing)
        for gnn_layer in self.gnn_layers:
            graph = gnn_layer(graph)
        
        # Readout: aggregate to graph-level
        # Use robot node features as graph representation
        robot_h = graph.nodes['robot'].data['h']  # (1, node_embedding_size)
        
        # Or mean pooling over all nodes
        # all_h = th.cat([
        #     graph.nodes['robot'].data['h'],
        #     th.mean(graph.nodes['human'].data['h'], dim=0, keepdim=True),
        #     th.mean(graph.nodes['obstacle'].data['h'], dim=0, keepdim=True),
        # ], dim=0)
        # graph_feat = th.mean(all_h, dim=0, keepdim=True)
        
        graph_feat = self.readout(robot_h)
        
        return graph_feat
```

---

### 4.5 Module: crowd_sim/envs/crowd_sim.py

**Purpose:** Gym environment for crowd navigation

#### Key Methods

```python
class CrowdSim(gym.Env):
    
    def configure(self, config):
        """Configure environment from config object"""
        # Time settings
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.robot_sensor_range = config.env.robot_sensor_range
        
        # Reward function
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.goal_factor = config.reward.goal_factor
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.discomfort_dist = config.reward.discomfort_dist
        self.re_rvo = config.reward.re_rvo
        self.re_theta = config.reward.re_theta
        
        # Scenario settings
        self.human_num = config.sim.human_num
        self.obstacle_num = config.sim.obstacle_num
        self.wall_num = config.sim.wall_num
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
    
    def reset(self, phase='train'):
        """
        Reset environment for new episode
        
        Returns:
            observations: (List[ObservableState], List[ObstacleState], List[WallState])
        """
        self.global_time = 0
        
        # Generate scenario
        if phase == 'train':
            scenario = self.train_val_scenario
        else:
            scenario = self.test_scenario
        
        # Reset robot
        self.robot.set_position(self._sample_position())
        self.robot.set_goal_position(self._sample_goal())
        self.robot.set_velocity((0, 0))
        
        # Reset humans
        self.humans = []
        for i in range(self.human_num):
            human = Human()
            human.set_position(self._sample_position())
            human.set_goal_position(self._sample_goal())
            human.set_velocity(self._sample_velocity())
            human.set_policy(policy_factory['orca']())
            self.humans.append(human)
        
        # Generate obstacles
        self.obstacles = self._generate_obstacles(self.obstacle_num)
        
        # Generate walls
        self.walls = self._generate_walls(self.wall_num)
        
        # Get initial observation
        ob = self._get_observation()
        
        return ob
    
    def step(self, action: ActionDiff):
        """
        Execute action and return next state
        
        Args:
            action: ActionDiff(left_acc, right_acc)
        
        Returns:
            (observation, reward, done, info)
        """
        # Update robot
        self.robot.step(action)
        
        # Update humans (using their policies, e.g., ORCA)
        for human in self.humans:
            # ORCA policy needs visible neighbors
            neighbors = self._get_visible_neighbors(human)
            human_action = human.policy.predict(neighbors)
            human.step(human_action)
        
        # Check termination conditions
        reaching_goal = self._check_reaching_goal()
        collision = self._check_collision()
        timeout = (self.global_time >= self.time_limit)
        
        # Compute reward
        if reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif timeout:
            reward = 0
            done = True
            info = Timeout()
        else:
            # Step reward
            reward = self._compute_step_reward()
            done = False
            info = Nothing()
        
        # Get observation
        ob = self._get_observation()
        
        # Update time
        self.global_time += self.time_step
        
        return ob, reward, done, info
    
    def _compute_step_reward(self):
        """
        Compute reward for non-terminal step
        
        Components:
        1. Goal progress: -goal_factor * distance_to_goal
        2. Discomfort penalty: for being too close to humans
        3. RVO reward: for following reciprocal velocity obstacle
        4. Theta reward: for heading alignment with goal
        """
        robot_pos = self.robot.get_position()
        robot_goal = self.robot.get_goal_position()
        
        # 1. Goal progress
        distance_to_goal = norm(np.array(robot_goal) - np.array(robot_pos))
        goal_reward = -self.goal_factor * distance_to_goal
        
        # 2. Discomfort penalty
        discomfort_penalty = 0
        for human in self.humans:
            human_pos = human.get_position()
            distance = norm(np.array(human_pos) - np.array(robot_pos))
            
            if distance < self.discomfort_dist:
                discomfort_penalty += (self.discomfort_dist - distance) * \
                                     self.discomfort_penalty_factor
        
        # 3. RVO reward (encourage following RVO)
        rvo_reward = self._compute_rvo_reward()
        
        # 4. Theta reward (heading alignment)
        theta_reward = self._compute_theta_reward()
        
        # Total reward
        reward = (goal_reward 
                 - discomfort_penalty 
                 + self.re_rvo * rvo_reward 
                 + self.re_theta * theta_reward)
        
        return reward
    
    def _get_observation(self):
        """
        Get observation for robot
        
        Returns:
            (human_states, obstacle_states, wall_states)
        """
        robot_pos = self.robot.get_position()
        
        # Get visible humans
        visible_humans = []
        for human in self.humans:
            distance = norm(np.array(human.get_position()) - np.array(robot_pos))
            if distance < self.robot_sensor_range:
                visible_humans.append(human.get_observable_state())
        
        # Get visible obstacles
        visible_obstacles = []
        for obstacle in self.obstacles:
            distance = norm(np.array(obstacle.position) - np.array(robot_pos))
            if distance < self.robot_sensor_range:
                visible_obstacles.append(obstacle.get_observable_state())
        
        # Get visible walls
        visible_walls = []
        for wall in self.walls:
            # Check if wall is in sensor range
            if self._wall_in_range(wall, robot_pos, self.robot_sensor_range):
                visible_walls.append(wall.get_observable_state())
        
        return (visible_humans, visible_obstacles, visible_walls)
```

---

## 5. Flow Training

### 5.1 Training Loop Chi tiết

```
┌──────────────────────────────────────────────────────┐
│         TRAINING LOOP (5M timesteps)                  │
└──────────────────────────────────────────────────────┘

Initialization:
    - Load config
    - Create environment
    - Initialize policy network
    - Initialize rollout buffer (size 2048)

For each iteration:
    
    ┌────────────── PHASE 1: COLLECT ROLLOUTS ──────────────┐
    │                                                        │
    │  For step in range(2048):                            │
    │                                                        │
    │    1. Generate action mask                           │
    │       ├─ For each goal in grid (81 goals):          │
    │       │    ├─ Test MPC feasibility                   │
    │       │    └─ mask[i] = 1 if feasible else 0        │
    │       └─ action_masks: (81,)                         │
    │                                                        │
    │    2. Get action from policy                         │
    │       policy(state, action_masks)                    │
    │       ├─ GNN feature extraction                      │
    │       ├─ Actor network → logits                      │
    │       ├─ Apply mask (set invalid to -inf)           │
    │       ├─ Sample action                               │
    │       └─ Returns: (action, value, log_prob)          │
    │                                                        │
    │    3. Execute action in environment                  │
    │       ├─ Convert action index to goal (x, y)         │
    │       ├─ Call MPC solver                             │
    │       ├─ MPC returns (left_acc, right_acc)           │
    │       ├─ env.step((left_acc, right_acc))             │
    │       └─ Returns: (obs, reward, done, info)          │
    │                                                        │
    │    4. Store transition in buffer                     │
    │       buffer.add(state, action, reward, value,       │
    │                 log_prob, done, action_masks)        │
    │                                                        │
    │    5. If done: reset environment                     │
    │                                                        │
    │  End for                                             │
    │                                                        │
    │  6. Compute GAE advantages                           │
    │     For t in reversed(range(2048)):                  │
    │         delta = reward[t] + gamma*value[t+1] - value[t]│
    │         advantage[t] = delta + gamma*lambda*advantage[t+1]│
    │                                                        │
    │  7. Compute returns                                  │
    │     returns[t] = advantages[t] + values[t]           │
    │                                                        │
    └────────────────────────────────────────────────────────┘
    
    ┌────────────── PHASE 2: UPDATE POLICY ─────────────────┐
    │                                                        │
    │  For epoch in range(10):                             │
    │                                                        │
    │    Shuffle buffer indices                            │
    │                                                        │
    │    For mini_batch in buffer.get(batch_size=64):     │
    │                                                        │
    │      1. Forward pass                                 │
    │         values, log_probs, entropy = policy.evaluate(│
    │             mini_batch.obs,                          │
    │             mini_batch.actions,                      │
    │             mini_batch.action_masks                  │
    │         )                                            │
    │                                                        │
    │      2. Normalize advantages                         │
    │         adv = (adv - adv.mean()) / adv.std()         │
    │                                                        │
    │      3. Compute policy loss (PPO clipped)           │
    │         ratio = exp(log_prob - old_log_prob)         │
    │         surr1 = ratio * advantages                   │
    │         surr2 = clip(ratio, 0.8, 1.2) * advantages   │
    │         policy_loss = -min(surr1, surr2).mean()      │
    │                                                        │
    │      4. Compute value loss                           │
    │         value_loss = MSE(values, returns)            │
    │                                                        │
    │      5. Compute entropy loss                         │
    │         entropy_loss = -entropy.mean()               │
    │                                                        │
    │      6. Total loss                                   │
    │         loss = policy_loss                           │
    │              + 0.5 * value_loss                      │
    │              + 0.001 * entropy_loss                  │
    │                                                        │
    │      7. Backpropagation                              │
    │         loss.backward()                              │
    │         clip_grad_norm(parameters, 0.5)              │
    │         optimizer.step()                             │
    │                                                        │
    │    End for mini_batch                                │
    │                                                        │
    │  End for epoch                                       │
    │                                                        │
    └────────────────────────────────────────────────────────┘
    
    ┌─────────── PHASE 3: EVALUATION (every 500 steps) ─────┐
    │                                                        │
    │  For episode in range(100):                          │
    │    Run episode with deterministic policy             │
    │    Record: success rate, collision rate, avg time    │
    │                                                        │
    │  If success_rate > best_success_rate:                │
    │    Save model as "best_model.pth"                    │
    │                                                        │
    └────────────────────────────────────────────────────────┘
    
    ┌────────── PHASE 4: CURRICULUM (if enabled) ───────────┐
    │                                                        │
    │  Gradually increase difficulty:                      │
    │    - Increase number of humans                       │
    │    - Increase human speed                            │
    │    - Decrease robot sensor range                     │
    │                                                        │
    └────────────────────────────────────────────────────────┘

End for iteration
```

### 5.2 Hyperparameters Summary

```python
# PPO hyperparameters
learning_rate = 2.5e-4 → 1.0e-4  # Linear decay
n_steps = 2048                    # Rollout length
batch_size = 64                   # Mini-batch size
n_epochs = 10                     # Update epochs
gamma = 0.99                      # Discount factor
gae_lambda = 0.95                 # GAE lambda
clip_range = 0.2                  # PPO clip epsilon
ent_coef = 0.001                  # Entropy coefficient
vf_coef = 0.5                     # Value loss coefficient
max_grad_norm = 0.5               # Gradient clipping

# Action space
action_dim = 9                    # 9x9 grid = 81 actions
action_range = 2.25               # [-2.25, 2.25] meters

# Environment
time_step = 0.25                  # 250ms per step
robot_sensor_range = 4.0          # 4 meters
human_num = 5                     # 5 humans
obstacle_num = 3                  # 3 obstacles
wall_num = 4                      # 4 walls

# Reward weights
success_reward = 0.25
collision_penalty = -0.25
goal_factor = 0.1
discomfort_penalty_factor = 0.5
re_rvo = 0.01
re_theta = 0.01

# Training
total_timesteps = 5e6             # 5 million steps
eval_freq = 500                   # Evaluate every 500 steps
n_eval_episodes = 100             # 100 episodes per evaluation
```

---

## 6. Các Kỹ thuật Chính

### 6.1 Action Masking (AM)

**Vấn đề:** Không phải tất cả local goals đều feasible với MPC solver

**Giải pháp:** Test MPC trước, mask out invalid actions

```python
# Pseudo-code
action_masks = []
for goal in candidate_goals:
    if mpc_can_solve(goal):
        action_masks.append(1)  # Valid
    else:
        action_masks.append(0)  # Invalid

# Apply mask in policy
logits[action_masks == 0] = -inf
distribution = Categorical(logits=logits)
action = distribution.sample()
```

**Lợi ích:**
- Tránh lãng phí exploration vào invalid actions
- Training ổn định hơn
- Convergence nhanh hơn

### 6.2 Privileged Learning (PL)

**Ý tưởng:** Sử dụng thông tin từ MPC expert trajectory để guide training

**Cách implement:**

```python
# During rollout collection
if use_PL:
    # Get expert trajectory from MPC
    expert_traj = mpc_solver.get_full_trajectory(
        horizon=PL_traj_length  # e.g., 4 steps
    )
    
    # Option 1: Reward shaping
    # Reward += alpha * reward_for_following_expert(robot_traj, expert_traj)
    
    # Option 2: Auxiliary loss
    # Add loss term: MSE(robot_traj, expert_traj)
    
    # Option 3: Store in buffer for imitation
    buffer.add(..., expert_traj=expert_traj)
```

**Lợi ích:**
- Học nhanh hơn từ expert
- Smoothness cao hơn
- Generalization tốt hơn

### 6.3 Spatio-Temporal Path Search

**Vấn đề:** A* thông thường không consider động học và moving obstacles

**Giải pháp:** A* với time dimension

```python
# Traditional A*: search in (x, y) space
# Spatio-temporal A*: search in (x, y, t) space

def st_astar(start, goal, dynamic_obstacles):
    # State: (x, y, t)
    # Edge: respect kinematic constraints (max velocity, acceleration)
    # Cost: time + distance
    # Avoid: predicted positions of obstacles at time t
    
    open_set = [(start, 0)]
    
    while open_set:
        current, t = pop_minimum(open_set)
        
        if current == goal:
            return reconstruct_path()
        
        for neighbor in get_neighbors(current):
            # Check kinematic feasibility
            if not is_kinematically_feasible(current, neighbor, dt):
                continue
            
            # Check collision at time t+dt
            t_next = t + dt
            if collides_with_obstacles(neighbor, t_next, dynamic_obstacles):
                continue
            
            # Add to open set
            g_score = g(current) + distance(current, neighbor)
            f_score = g_score + heuristic(neighbor, goal)
            open_set.add((neighbor, t_next), f_score)
    
    return None  # No path found
```

**Lợi ích:**
- Better initial guess cho MPC
- Tránh local minima
- Faster convergence

### 6.4 Graph Neural Network

**Tại sao dùng GNN?**
- Số humans/obstacles thay đổi mỗi episode
- Spatial relationships quan trọng
- Permutation invariant

**Graph Structure:**

```python
# Heterogeneous graph
Graph:
    Nodes:
        - robot: 1 node, features [px, py, vx, vy, r, gx, gy]
        - human: N nodes, features [px, py, vx, vy, r]
        - obstacle: M nodes, features [px, py, r, type]
        - wall: K nodes, features [start_x, start_y, end_x, end_y]
    
    Edges:
        - robot → human: spatial proximity
        - robot → obstacle: spatial proximity
        - robot → wall: spatial proximity
        - human → human: social force (optional)
```

**Message Passing:**

```python
# For each layer:
for node_type in ['robot', 'human', 'obstacle']:
    # Gather messages from neighbors
    messages = []
    for neighbor in graph.neighbors(node):
        edge_feat = edge_features(node, neighbor)
        message = MLP([neighbor.feat, edge_feat])
        messages.append(message)
    
    # Aggregate messages
    aggregated = sum(messages) / len(messages)
    
    # Update node features
    node.feat = MLP([node.feat, aggregated])
```

---

## 7. Tips Debugging

### 7.1 Kiểm tra Training đang chạy

```bash
# Check tensorboard
tensorboard --logdir=train_data/your_experiment/

# Monitor:
# - rollout/ep_ret_mean: Average return (should increase)
# - rollout/ep_len_mean: Episode length
# - train/policy_gradient_loss: Policy loss
# - train/value_loss: Value loss
# - train/entropy_loss: Entropy (should decrease gradually)
```

### 7.2 Common Issues

#### Issue 1: Training không converge

**Possible causes:**
- Learning rate quá cao/thấp
- Reward scale không phù hợp
- Action mask quá restrictive

**Solutions:**
```python
# Adjust learning rate
learning_rate = get_linear_fn(1.5e-4, 5.0e-5, 0.5)

# Tune reward weights
goal_factor = 0.1        # Try 0.05, 0.2
safe_weight = 0.5        # Try 0.3, 0.7
collision_penalty = -0.25  # Try -0.1, -0.5
```

#### Issue 2: Policy collapse (chỉ chọn 1 action)

**Possible causes:**
- Entropy coefficient quá thấp
- Early termination condition wrong

**Solutions:**
```python
# Increase entropy coefficient
ent_coef = 0.01  # instead of 0.001

# Check action distribution
print(f"Action distribution: {th.softmax(action_logits, dim=-1)}")
```

#### Issue 3: MPC solver quá chậm

**Solutions:**
```python
# Use FORCES PRO instead of IPOPT
ocp_planner.set_solver("forces")

# Reduce MPC horizon
ocp_planner.set_horizon(15)  # instead of 20

# Increase time step
ocp_planner.set_dt(0.25)  # instead of 0.1

# Enable warm start
ocp_planner.enable_warm_start(True)
```

#### Issue 4: Out of memory

**Solutions:**
```python
# Reduce batch size
batch_size = 32  # instead of 64

# Reduce rollout buffer size
n_steps = 1024  # instead of 2048

# Use gradient accumulation
for _ in range(2):
    loss = compute_loss(mini_batch)
    (loss / 2).backward()
optimizer.step()
```

### 7.3 Debugging Tools

```python
# Enable debug mode
python train_ppo.py --debug

# Visualize episodes
python eval_ppo.py --visualize --model_path=best_model.pth

# Check action masks
env.action_mask_vis = True
env.render(mode='human')

# Profile code
import cProfile
cProfile.run('model.learn(1000)', 'profile_stats')

import pstats
p = pstats.Stats('profile_stats')
p.sort_stats('cumtime').print_stats(20)
```

### 7.4 Validation Checklist

- [ ] Environment reset correctly?
- [ ] Observation space correct?
- [ ] Action space correct?
- [ ] Reward computation correct?
- [ ] Termination conditions correct?
- [ ] MPC solver returns valid solutions?
- [ ] Action masks generated correctly?
- [ ] GNN receives correct graph structure?
- [ ] Policy outputs valid distribution?
- [ ] Gradients flow correctly?

---

## 📚 Tài liệu tham khảo

1. **Paper:** [Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation](https://arxiv.org/abs/2506.09859)
2. **PPO Algorithm:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
3. **Graph Neural Networks:** [Graph Neural Networks: A Review](https://arxiv.org/abs/1812.08434)
4. **ORCA:** [Reciprocal n-Body Collision Avoidance](https://gamma.cs.unc.edu/ORCA/)
5. **MPC:** [Model Predictive Control: Theory and Design](https://www.cambridge.org/core/books/model-predictive-control/4C4C7A7F5C1E5B5B7E6F9E1F5D5A5C5B)

---

## ❓ FAQ

**Q: Tôi nên bắt đầu từ đâu?**  
A: Bắt đầu từ [train_ppo.py](drl_moudle/train_ppo.py), đọc kỹ hàm `main()`. Sau đó đọc [algorithms/graph_ppo.py](drl_moudle/algorithms/graph_ppo.py) để hiểu PPO algorithm. Cuối cùng đọc [crowd_sim.py](drl_moudle/crowd_sim/envs/crowd_sim.py) để hiểu environment.

**Q: Làm sao để train model mới?**  
A: Xem [assets/Run.md](assets/Run.md) hoặc chạy:
```bash
cd drl_moudle
python train_ppo.py --output_dir train_data/my_exp --action_dim 9
```

**Q: Làm sao để test model đã train?**  
A: Chạy:
```bash
python eval_ppo.py --model_path data/model1/best_model.pth --visualize
```

**Q: Action Masking hoạt động như thế nào?**  
A: Với mỗi candidate goal, gọi MPC solver để kiểm tra feasibility. Nếu MPC fail → mask = 0. Policy chỉ sample từ valid actions (mask = 1).

**Q: Privileged Learning là gì?**  
A: Sử dụng trajectory từ MPC expert (có access đến ground-truth info) để guide RL training. Có thể dùng cho reward shaping hoặc auxiliary loss.

**Q: Tại sao dùng Graph NN?**  
A: Vì số humans/obstacles thay đổi mỗi episode. GNN có thể handle variable-size input và học spatial relationships.

**Q: MPC solver nào nhanh nhất?**  
A: FORCES PRO > IPOPT. FORCES PRO là commercial solver, có thể code generation. IPOPT là open-source, chậm hơn nhưng đủ dùng.

**Q: Training mất bao lâu?**  
A: Khoảng 24-48 giờ cho 5M timesteps trên 1 GPU (RTX 3090). Có thể giảm xuống bằng cách giảm action_dim hoặc tăng n_steps.

---

## 🎓 Kết luận

Đây là một dự án phức tạp kết hợp nhiều kỹ thuật:
- **Deep RL (PPO)** cho high-level decision making
- **Graph Neural Network** cho spatial reasoning
- **Model Predictive Control** cho low-level control
- **Action Masking** cho efficiency
- **Privileged Learning** cho sample efficiency

Đọc code theo thứ tự đề xuất trong guide này, bạn sẽ hiểu được toàn bộ hệ thống. Chúc bạn học tốt! 🚀

---

**Cập nhật lần cuối:** 5 Tháng 3, 2026  
**Author:** GitHub Copilot  
**Version:** 1.0
