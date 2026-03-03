# HALO - Hướng dẫn Kiến trúc Chi tiết
# Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation

---

## 📋 Mục lục
1. [Tổng quan Hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc Chi tiết](#2-kiến-trúc-chi-tiết)
3. [Module Training (drl_module)](#3-module-training-drl_module)
4. [Module MPC Solver (src/ocp_planner)](#4-module-mpc-solver-srcocp_planner)
5. [Quy trình Training](#5-quy-trình-training)
6. [Quy trình Evaluation & Testing](#6-quy-trình-evaluation--testing)
7. [Cách sử dụng](#7-cách-sử-dụng)

---

## 1. Tổng quan Hệ thống

### 1.1 Mô tả Tổng quan
HALO là một hệ thống điều hướng robot trong môi trường động với đám đông người. Hệ thống kết hợp:

- **Graph Neural Network (GNN)** được huấn luyện bằng Deep Reinforcement Learning (PPO)
- **Model Predictive Control (MPC)** để xử lý các ràng buộc động học
- **Spatio-temporal Path Search** để tạo궤ại tham chiếu
- **Action Masking** và **Privileged Learning** để cải thiện hiệu suất

### 1.2 Luồng xử lý chính

```
┌─────────────────────────────────────────────────────────────────┐
│                     HALO SYSTEM ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────┘

    [Observation: Robot + Humans + Obstacles]
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │   Graph Neural Network (GNN Policy)      │
    │   - Encode spatial relationships         │
    │   - Generate cost-to-go estimation       │
    │   - Output: Local goal recommendation    │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │   Action Masking (AM)                    │
    │   - Filter infeasible actions            │
    │   - Based on MPC solver feedback         │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │   Selected Local Goal                    │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │   Spatio-Temporal Path Search            │
    │   - A* for initial path                  │
    │   - Consider kinematic constraints       │
    │   - Generate reference trajectory        │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │   MPC Solver (IPOPT/FORCES PRO)         │
    │   - Input: Reference trajectory          │
    │   - Solve non-convex optimization        │
    │   - Enforce safety constraints           │
    │   - Output: Left/Right wheel acc         │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │   Robot Control (Differential Drive)     │
    │   - Execute motion commands              │
    └─────────────────────────────────────────┘
```

---

## 2. Kiến trúc Chi tiết

### 2.1 Cấu trúc Thư mục

```
HALO/
├── drl_module/                    # Module Deep RL Training
│   ├── train_ppo.py              # Script training chính
│   ├── eval_ppo.py               # Script evaluation & testing
│   ├── algorithms/               # Implementation algorithms
│   │   ├── graph_ppo.py         # Base Graph PPO
│   │   └── mpc_ppo.py           # MPC-integrated PPO
│   ├── configs/                  # Configuration files
│   │   ├── config.py            # Base config
│   │   └── mpc_rl.py            # MPC-RL specific config
│   ├── crowd_sim/               # Simulation environment
│   │   └── envs/
│   │       ├── crowd_sim.py     # Main environment
│   │       ├── policy/          # Human policies (ORCA, etc)
│   │       └── utils/           # Utilities (Robot, Human, State, Action)
│   ├── modules/                 # Core RL modules
│   │   ├── graph_ppo_core.py   # Graph-based Actor-Critic
│   │   ├── mpc_ppo_core.py     # Masked policy for MPC
│   │   ├── gnn_models.py       # GNN architectures
│   │   ├── callbacks.py        # Training callbacks
│   │   └── evaluation.py       # Evaluation utilities
│   └── simple_env/              # Simplified environment for testing
│
└── src/ocp_planner/             # MPC Solver Package (C++)
    ├── src/
    │   ├── planner.h/cc         # Main planner interface
    │   ├── mpc.h/cc             # MPC solver implementation
    │   ├── astar.h/cc           # A* path planning
    │   ├── lookahead.h/cc       # Velocity planning
    │   ├── smooth.h/cc          # Path smoothing
    │   ├── types.h              # Data structures
    │   └── ocp_bind.cc          # Python bindings (pybind11)
    └── srv/
        └── OcpLocalPlann.srv    # ROS service definition
```

---

## 3. Module Training (drl_module)

### 3.1 Script Training (`train_ppo.py`)

#### 3.1.1 Chức năng chính
Training một Graph Neural Network policy sử dụng PPO algorithm với MPC integration.

#### 3.1.2 Các bước thực hiện

**Bước 1: Setup và Configuration**
```python
# Load config file
config = importlib.util.module_from_spec(spec)

# Khởi tạo environment configuration
env_config = config.EnvConfig(args.debug)

# Thiết lập các hyperparameters:
- success_reward: Phần thưởng khi đến đích
- collision_penalty: Phạt khi va chạm
- goal_factor: Trọng số hướng tới mục tiêu
- safe_weight: Trọng số an toàn (tránh va chạm)
- re_rvo: Reward cho RVO (Reciprocal Velocity Obstacle)
- re_theta: Reward cho góc hướng
```

**Bước 2: Khởi tạo Environment**
```python
env = gym.make("CrowdSim-v0")
env.configure(env_config)
env.set_phase(0)  # Curriculum learning phase

# Setup robot
robot = Robot(env_config, "robot")
robot.set_policy(ExternalPolicy())  # Policy từ RL
env.set_robot(robot)
```

**Bước 3: Thiết lập không gian hành động (Action Space)**
```python
# Discrete action space: action_dim x action_dim
# Mỗi action tuple (i, j) tương ứng với 1 local goal
action_dim = 9  # Tạo lưới 9x9 = 81 actions
goal_range = (-2.25, 2.25)  # Phạm vi local goal (m)
env.action_space = gym.spaces.Discrete(81)

# Các tính năng:
env.use_AM = True   # Action Masking
env.use_PL = True   # Privileged Learning
env.PL_traj_length = 4      # Số bước trajectory cho PL
env.PL_traj_gamma = 0.9     # Discount factor cho PL
```

**Bước 4: Khởi tạo Model**
```python
model = MpcPPO(
    "GraphPolicy",
    env,
    learning_rate=lr_schedule,    # Linear decay từ 2.5e-4 đến 1.0e-4
    n_steps=2048,                 # Steps per rollout
    batch_size=64,
    ent_coef=0.001,              # Entropy coefficient
    tensorboard_log=output_dir,
    action_dim=action_dim,
    goal_coord_range=goal_range,
)
```

**Bước 5: Training Loop**
```python
# Callbacks
curriculum_callback = CurriculumCallback()  # Tăng độ khó dần
eval_callback = EvalCallback(
    eval_env=env,
    n_eval_episodes=100,
    eval_freq=500,              # Eval mỗi 500 iterations
)

# Train 5 million steps
model.learn(int(5e6), callback=[eval_callback, curriculum_callback])
```

### 3.2 Algorithm Implementation (`algorithms/mpc_ppo.py`)

#### 3.2.1 Class MpcPPO

Kế thừa từ `GraphPPO` và tích hợp MPC solver.

**Key Components:**

1. **Rollout Collection với Action Masking**
```python
def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
    # Lấy observation từ environment
    obs = env.reset()
    
    for step in range(n_rollout_steps):
        # 1. Predict action từ policy với action mask
        action, value, log_prob = self.policy.forward(
            obs, 
            action_mask=env.action_mask
        )
        
        # 2. Convert action index thành local goal coordinates
        local_goal = self.map_action_to_goal(action)
        
        # 3. Gọi MPC solver để tính control
        left_acc, right_acc, mpc_success = self.call_mpc_solver(
            robot_state=env.robot.get_full_state(),
            humans=obs.humans,
            obstacles=obs.obstacles,
            local_goal=local_goal
        )
        
        # 4. Thực thi action trong environment
        next_obs, reward, done, info = env.step(
            ActionDiff(left_acc, right_acc)
        )
        
        # 5. Update action mask dựa trên MPC feedback
        env.update_action_mask(mpc_success)
        
        # 6. Lưu vào buffer
        rollout_buffer.add(
            obs, action, reward, done, value, log_prob,
            mask=env.action_mask
        )
        
        obs = next_obs
```

2. **MPC Solver Integration**
```python
def call_mpc_solver(self, robot_state, humans, obstacles, local_goal):
    if self.use_ros:
        # Gọi ROS service
        return self._call_ros_service(...)
    else:
        # Gọi Python binding trực tiếp
        return self._call_python_binding(...)

def _call_python_binding(self, ...):
    # Chuẩn bị input cho C++ solver
    mpc_input = ocp_planner_py.MPCInputForPython()
    mpc_input.ob = joint_state
    mpc_input.sub_goal = local_goal
    
    # Gọi solver
    mpc_output = self.ocp_planner.run_mpc_solver(mpc_input)
    
    return mpc_output.al, mpc_output.ar, mpc_output.success
```

3. **Privileged Learning**
```python
def compute_privileged_value(self, obs, traj_length=4):
    """
    Học value function với thông tin privileged:
    - Biết trước trajectory của người trong tương lai
    - Giúp policy học tốt hơn mặc dù không có info này lúc inference
    """
    privileged_obs = self.augment_with_future_trajectory(
        obs, 
        traj_length=traj_length
    )
    privileged_value = self.value_network(privileged_obs)
    return privileged_value
```

### 3.3 Policy Network (`modules/mpc_ppo_core.py`)

#### 3.3.1 MaskedActorCriticPolicy

**Graph-based Policy với Action Masking:**

```python
class MaskedActorCriticPolicy(GraphActorCriticPolicy):
    def forward(self, state, action_mask=None):
        # 1. Encode state thành graph
        graph = joint_state_as_graph(state)
        
        # 2. GNN feature extraction
        features = self.graph_extractor(graph)
        
        # 3. Actor network
        logits = self.action_net(features)
        
        # 4. Apply action mask
        if action_mask is not None:
            masked_logits = logits.masked_fill(
                ~action_mask.bool(), 
                float('-inf')
            )
        else:
            masked_logits = logits
        
        # 5. Sample action
        action_distribution = Categorical(logits=masked_logits)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        
        # 6. Critic network
        value = self.value_net(features)
        
        return action, value, log_prob
```

### 3.4 Environment (`crowd_sim/envs/crowd_sim.py`)

#### 3.4.1 CrowdSim Environment

**Các thành phần chính:**

1. **State Representation**
```python
class JointState:
    robot_state: FullState      # [px, py, vx, vy, radius, gx, gy, v_pref, theta]
    human_states: List[ObservableState]  # List of humans
    obstacle_states: List[ObstacleState]
    wall_states: List[WallState]
```

2. **Reset Function** - Tạo scenario mới
```python
def reset(self, phase='train', for_debug=False, seed=None):
    # 1. Random robot start & goal
    self.robot.set(px, py, gx, gy, vx, vy, theta)
    
    # 2. Generate humans với ORCA policy
    for i in range(self.human_num):
        human = Human()
        human.set_policy(ORCA())
        self.humans.append(human)
    
    # 3. Generate static obstacles
    for i in range(self.static_obstacle_num):
        obstacle = Obstacle(px, py, radius)
        self.obstacles.append(obstacle)
    
    # 4. Generate walls
    self.walls = [left_wall, right_wall, top_wall, bottom_wall]
    
    # 5. Initialize action mask (all actions valid initially)
    self.action_mask = torch.ones(self.action_space.n)
    
    return self.get_observation()
```

3. **Step Function** - Tiến hành 1 bước simulation
```python
def step(self, action: ActionDiff):
    # 1. Update robot state
    self.robot.step(action)
    
    # 2. Update humans (ORCA policy)
    for human in self.humans:
        human_action = human.act(self.get_observable_state())
        human.step(human_action)
    
    # 3. Check collision
    collision = self.check_collision()
    
    # 4. Check reaching goal
    reaching_goal = self.robot.reached_goal()
    
    # 5. Compute reward
    reward = self.compute_reward(collision, reaching_goal)
    
    # 6. Update global time
    self.global_time += self.time_step
    
    done = collision or reaching_goal or (self.global_time >= self.time_limit)
    
    return self.get_observation(), reward, done, info
```

4. **Reward Function**
```python
def compute_reward(self, collision, reaching_goal):
    if collision:
        return self.collision_penalty  # -0.25
    elif reaching_goal:
        return self.success_reward     # +0.25
    else:
        # Reward shaping
        reward = 0.0
        
        # 1. Goal progress
        reward += self.goal_factor * delta_distance_to_goal
        
        # 2. Safety (distance to humans and obstacles)
        min_dist = min(distances_to_humans_and_obstacles)
        if min_dist < self.discomfort_dist:
            reward += -self.discomfort_penalty_factor * (
                self.discomfort_dist - min_dist
            )
        
        # 3. RVO compliance (smooth motion)
        reward += self.re_rvo * rvo_score
        
        # 4. Heading alignment
        reward += self.re_theta * heading_score
        
        return reward
```

5. **Action Masking Update**
```python
def update_action_mask(self, mpc_feedbacks):
    """
    Update action mask dựa trên feedback từ MPC solver:
    - Nếu MPC fail to solve với 1 local goal -> mask action đó
    - Nếu trajectory không an toàn -> mask action đó
    """
    for action_idx, feedback in enumerate(mpc_feedbacks):
        if not feedback.success or not feedback.safe:
            self.action_mask[action_idx] = 0  # Mask action này
```

### 3.5 GNN Architecture (`modules/gnn_models.py`)

**Graph Construction:**
- **Nodes**: Robot, Humans, Obstacles, Walls
- **Edges**: 
  - Robot to all entities
  - Human to human (within sensor range)
  - Spatial relationships

**GNN Layers:**
1. Node feature encoding
2. Graph attention layers
3. Message passing
4. Readout for policy & value

---

## 4. Module MPC Solver (src/ocp_planner)

### 4.1 Planner Class (`planner.h/cc`)

#### 4.1.1 Cấu trúc chính

```cpp
class Planner {
public:
    // Components
    std::unique_ptr<AStar> astar_planner_;
    std::unique_ptr<SmoothCorner> path_smoother_;
    std::unique_ptr<LookAhead> vel_planner_;
    std::unique_ptr<Mpc> ocp_planner_;
    
    // Main interface
    MPCOutputForPython RunSlover(MPCInputForPython& input);
};
```

#### 4.1.2 RunSolver Flow

```cpp
MPCOutputForPython Planner::RunSlover(MPCInputForPython& input) {
    // 1. Extract info từ input
    RobotState robot = input.ob.robot;
    Vector2d local_goal = input.sub_goal;
    
    // 2. A* path planning
    // Tìm đường đi từ robot đến local_goal tránh obstacles
    std::vector<Point> astar_path;
    bool astar_success = astar_planner_->SearchPath(
        robot.position, 
        local_goal, 
        astar_path
    );
    
    if (!astar_success) {
        output.success = false;
        return output;
    }
    
    // 3. Path smoothing
    // Làm mượt đường đi A*
    std::vector<Point> smooth_path;
    path_smoother_->Smooth(astar_path, smooth_path);
    
    // 4. Spatio-temporal search
    // Gán velocity profile cho từng điểm trên path
    // Xét đến kinematic constraints (max_vel, max_acc)
    std::vector<Point> st_path;
    vel_planner_->PlanVelocity(
        smooth_path, 
        robot.v, 
        robot.yaw,
        st_path  // path with (x, y, v, t)
    );
    
    // 5. MPC solver
    // Solve nonlinear optimization để track st_path
    MpcReturn mpc_result = ocp_planner_->RunMpc(
        input.ob,    // Joint state
        st_path      // Reference trajectory
    );
    
    // 6. Package output
    output.success = mpc_result.success;
    output.al = mpc_result.left_acc;
    output.ar = mpc_result.right_acc;
    output.astar_path = astar_path;
    output.st_path = st_path;
    
    return output;
}
```

### 4.2 MPC Solver (`mpc.h/cc`)

#### 4.2.1 Problem Formulation

**Decision Variables:**
```cpp
// Horizon: NP = 10 steps
// State: [px, py, theta, v, yaw_rate]
// Control: [left_acc, right_acc]

X = [x_0, u_0, x_1, u_1, ..., x_{NP-1}, u_{NP-1}, x_NP]
```

**Objective Function:**
```cpp
min J = Σ_{k=0}^{NP} [
    // 1. Tracking error
    w_pos * ||p_k - p_ref_k||^2 +
    w_vel * ||v_k - v_ref_k||^2 +
    w_yaw * ||theta_k - theta_ref_k||^2 +
    
    // 2. Control effort
    w_u * ||u_k||^2 +
    
    // 3. Control smoothness
    w_du * ||u_k - u_{k-1}||^2
]
```

**Constraints:**
```cpp
// 1. Dynamics constraints (Differential drive kinematics)
x_{k+1} = f(x_k, u_k)

// 2. Box constraints
v_min <= v_k <= v_max
|yaw_rate_k| <= yaw_rate_max
|left_acc_k| <= acc_max
|right_acc_k| <= acc_max

// 3. Obstacle avoidance (convex approximation)
For each obstacle i:
    A_i * p_k >= b_i + safety_margin

// 4. Human avoidance (predicted trajectory)
For each human j at step k:
    ||p_k - p_human_j(k)|| >= r_robot + r_human + safety_margin
```

#### 4.2.2 Implementation với CasADi

```cpp
int Mpc::SolveMpc(...) {
    // 1. Setup CasADi optimization problem
    casadi::Opti opti;
    
    // 2. Decision variables
    auto X = opti.variable(state_dim, NP+1);
    auto U = opti.variable(control_dim, NP);
    
    // 3. Objective
    casadi::MX cost = 0;
    for (int k = 0; k < NP; k++) {
        // Tracking cost
        cost += casadi::MX::mtimes({
            (X(casadi::Slice(0,2), k) - ref_path[k]).T(),
            Q_pos,
            X(casadi::Slice(0,2), k) - ref_path[k]
        });
        
        // Control cost
        cost += casadi::MX::mtimes({U(casadi::Slice(), k).T(), R, U(casadi::Slice(), k)});
    }
    opti.minimize(cost);
    
    // 4. Dynamics constraints
    auto model = RobotModelDifferential();
    for (int k = 0; k < NP; k++) {
        auto x_next = DynamicRK4(model, dt)(X(casadi::Slice(), k), U(casadi::Slice(), k));
        opti.subject_to(X(casadi::Slice(), k+1) == x_next);
    }
    
    // 5. Path constraints
    for (int k = 0; k < NP; k++) {
        // Velocity limits
        opti.subject_to(X(3, k) >= 0);
        opti.subject_to(X(3, k) <= max_v);
        
        // Control limits
        opti.subject_to(opti.bounded(-max_acc, U(0, k), max_acc));
        opti.subject_to(opti.bounded(-max_acc, U(1, k), max_acc));
        
        // Obstacle avoidance
        for (auto& obst : obstacles) {
            opti.subject_to(
                casadi::MX::norm_2(X(casadi::Slice(0,2), k) - obst.pos) 
                >= robot_radius + obst.radius + safety_margin
            );
        }
    }
    
    // 6. Initial condition
    opti.subject_to(X(casadi::Slice(), 0) == x0);
    
    // 7. Warm start with initial guess
    opti.set_initial(X, init_guess.states);
    opti.set_initial(U, init_guess.controls);
    
    // 8. Solver options (IPOPT)
    opti.solver("ipopt", {
        {"print_level", verbose_ ? 5 : 0},
        {"max_iter", 100},
        {"tol", 1e-4},
        {"linear_solver", "ma57"}  // Requires academic license
    });
    
    // 9. Solve
    try {
        auto sol = opti.solve();
        
        // Extract solution
        auto X_opt = sol.value(X);
        auto U_opt = sol.value(U);
        
        output.success = true;
        output.left_acc = U_opt(0, 0);
        output.right_acc = U_opt(1, 0);
        
        return 0;
    } catch (std::exception& e) {
        output.success = false;
        return -1;
    }
}
```

### 4.3 Python Binding (`ocp_bind.cc`)

```cpp
PYBIND11_MODULE(ocp_planner_py, m) {
    // Bind các data structures
    py::class_<RobotState>(m, "RobotState")
        .def(py::init<>())
        .def_readwrite("px", &RobotState::px)
        .def_readwrite("py", &RobotState::py)
        ...;
    
    // Bind Planner class
    py::class_<Planner>(m, "OcpPlanner")
        .def(py::init<>())
        .def("run_mpc_solver", &Planner::RunSlover);
}
```

**Sử dụng trong Python:**
```python
import ocp_planner_py

# Khởi tạo planner
planner = ocp_planner_py.OcpPlanner()

# Chuẩn bị input
mpc_input = ocp_planner_py.MPCInputForPython()
mpc_input.ob = joint_state
mpc_input.sub_goal = local_goal

# Gọi solver
mpc_output = planner.run_mpc_solver(mpc_input)

# Lấy kết quả
if mpc_output.success:
    left_acc = mpc_output.al
    right_acc = mpc_output.ar
```

---

## 5. Quy trình Training

### 5.1 Chuẩn bị

```bash
# 1. Setup environment
cd ~/HALO
conda activate mpc_rl
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 2. Build C++ solver
source ./devel/setup.bash
./build_cp.sh
```

### 5.2 Curriculum Learning

Training chia làm 4 phases với độ khó tăng dần:

**Phase 0**: 1 human, 1 obstacle
**Phase 1**: 1 human, 3 obstacles  
**Phase 2**: 3 humans, 3 obstacles
**Phase 3+**: 5 humans, 3 obstacles

```python
class CurriculumCallback(BaseCallback):
    def _on_step(self):
        # Tự động tăng phase khi success rate > threshold
        if self.success_rate > 0.85:
            self.env.set_phase(self.current_phase + 1)
```

### 5.3 Training Command

```bash
cd drl_module
python train_ppo.py \
    --config configs/mpc_rl.py \
    --output_dir train_data/experiment_1 \
    --action_dim 9 \
    --action_range 2.25 \
    --use_AM True \
    --use_PL True \
    --PL_traj_length 4 \
    --randomseed 3
```

### 5.4 Monitoring Training

```bash
# Tensorboard
tensorboard --logdir train_data/experiment_1

# Metrics tracked:
- rollout/ep_rew_mean: Average episode reward
- train/policy_loss: Policy loss
- train/value_loss: Value loss
- rollout/success_rate: Success rate
- rollout/collision_rate: Collision rate
```

### 5.5 Trained Model Output

```
train_data/experiment_1/
├── config.py                    # Configuration backup
├── output.log                   # Training log
├── best_model.zip              # Best model checkpoint
├── PPO_1/                      # Tensorboard logs
│   └── events.out.tfevents...
└── evaluations.npz             # Evaluation results
```

---

## 6. Quy trình Evaluation & Testing

### 6.1 Evaluation Script (`eval_ppo.py`)

#### 6.1.1 Load trained model

```python
model = MpcPPO(
    "GraphPolicy",
    env,
    device=device,
    action_dim=9,
    goal_coord_range=(-2.0, 2.0),
)

# Load trained weights
model.set_parameters("data/model1/best_model", device=device)
```

#### 6.1.2 Batch Evaluation

```python
# Test trên 500 episodes
n_eval_episodes = 500

success_rate, collision_rate, avg_time, avg_return = evaluate_policy(
    model,
    env,
    n_eval_episodes=n_eval_episodes,
    render=False,
    phase='test',
)

print(f"Success rate: {100 * success_rate:.2f}%")
print(f"Collision rate: {100 * collision_rate:.2f}%")
print(f"Average time: {avg_time:.2f}s")
```

#### 6.1.3 Single Episode với Visualization

```python
n_eval_episodes = 1

ob = env.reset(phase='test', seed=1421)
robot_state = env.robot.get_full_state()
joint_state = JointState(robot_state, ob)

done = False
step_count = 0

while not done:
    # Predict action
    action = model.predict(
        joint_state, 
        action_mask=env.action_mask,
        deterministic=True
    )
    
    left_acc, right_acc = action[0]
    
    # Visualize
    env.render(mode="debug", output_file="images/")
    
    # Step
    ob, reward, done, info = env.step(ActionDiff(left_acc, right_acc))
    
    new_robot_state = env.robot.get_full_state()
    joint_state = JointState(new_robot_state, ob)
    
    step_count += 1

print(f"Episode finished in {step_count} steps")
print(f"Result: {info['event']}")  # ReachGoal, Collision, Timeout
```

### 6.2 Evaluation Command

```bash
cd drl_module

# Batch evaluation (500 episodes)
python eval_ppo.py \
    --model_dir data/model1 \
    --n_eval_episodes 500 \
    --use_action_mask True \
    --randomseed 1421

# Single episode với visualization
python eval_ppo.py \
    --model_dir data/model1 \
    --n_eval_episodes 1 \
    --visualize \
    --output_image_dir images/test_1421 \
    --randomseed 1421
```

### 6.3 Testing Scenarios

**Test environments** (phase=10):
- 5 humans với ORCA policy
- 3 static obstacles  
- 4 walls (boundaries)
- Random start & goal positions
- Test cases: seed từ 1100-1599 (500 scenarios)

### 6.4 Evaluation Metrics

```python
class EvaluationMetrics:
    success_rate: float          # % episodes đạt goal
    collision_rate: float        # % episodes va chạm
    timeout_rate: float          # % episodes timeout
    avg_time_to_goal: float     # Thời gian trung bình đến goal
    avg_path_length: float      # Độ dài quãng đường
    avg_reward: float           # Reward trung bình
    min_distance_to_human: float # Khoảng cách an toàn
```

---

## 7. Cách sử dụng

### 7.1 Installation

#### 7.1.1 Prerequisites

```bash
# Ubuntu 20.04
# ROS Noetic
# Python 3.8+
# CUDA 11.3+ (nếu dùng GPU)
```

#### 7.1.2 Dependencies

```bash
# System dependencies
sudo apt-get install -y \
    libeigen3-dev \
    libopencv-dev

# Install Casadi
# https://web.casadi.org/get/

# Install IPOPT với MA57 solver (academic license)
# https://coin-or.github.io/Ipopt/
```

#### 7.1.3 Python Environment

```bash
conda env create -f drl_module/environment.yml
conda activate mpc_rl
```

#### 7.1.4 Build C++ Module

```bash
cd HALO

# Build với pybind11
./build_cp.sh

# Hoặc build với ROS (nếu cần ROS service)
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

### 7.2 Quick Start - Training

```bash
# 1. Activate environment
conda activate mpc_rl
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 2. Navigate to training directory
cd drl_module

# 3. Start training
python train_ppo.py \
    --output_dir train_data/my_experiment \
    --action_dim 9 \
    --action_range 2.25 \
    --use_AM True \
    --use_PL True

# Training sẽ chạy 5M steps (~10-20 giờ trên GPU)
```

### 7.3 Quick Start - Evaluation

```bash
# 1. Activate environment
conda activate mpc_rl
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 2. Batch evaluation
cd drl_module
python eval_ppo.py \
    --model_dir data/model1 \
    --n_eval_episodes 500

# 3. Single episode với visualization
python eval_ppo.py \
    --model_dir data/model1 \
    --n_eval_episodes 1 \
    --visualize \
    --output_image_dir images/demo
```

### 7.4 Deployment (ROS)

```bash
# 1. Build với ROS service
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

# 2. Launch ROS service
roslaunch ocp_planner ocp_planner_node.launch

# 3. Run policy node (Python)
cd drl_module
python ros_policy_node.py --model_dir data/model1
```

**Kiến trúc ROS:**
```
[Robot State] -> [Policy Node] -> [Local Goal] 
                                        ↓
                                  [MPC Service]
                                        ↓
                                  [Control Cmd] -> [Robot Hardware]
```

### 7.5 Hyperparameter Tuning

**Training hyperparameters** trong [train_ppo.py](drl_moudle/train_ppo.py#L89):

```python
# Learning rate schedule
lr_schedule = get_linear_fn(
    start=2.5e-4,  # Initial learning rate
    end=1.0e-4,    # Final learning rate
    fraction=0.5   # Fraction of training to decay
)

# PPO parameters
model = MpcPPO(
    learning_rate=lr_schedule,
    n_steps=2048,           # Rollout buffer size
    batch_size=64,          # Minibatch size
    n_epochs=10,            # Epochs per update
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE lambda
    clip_range=0.2,         # PPO clip range
    ent_coef=0.001,         # Entropy coefficient
    vf_coef=0.5,            # Value function coefficient
    max_grad_norm=0.5,      # Gradient clipping
)

# Reward weights
env_config.reward.success_reward = 0.25
env_config.reward.collision_penalty = -0.25
env_config.reward.goal_factor = 0.1
env_config.reward.discomfort_penalty_factor = 0.5
env_config.reward.re_rvo = 0.01
env_config.reward.re_theta = 0.01

# Action space
action_dim = 9              # Grid size: 9x9 = 81 actions
action_range = 2.25         # Local goal range: ±2.25m

# Privileged learning
use_PL = True
PL_traj_length = 4          # Future trajectory length
PL_traj_gamma = 0.9         # Discount for future info
```

**MPC parameters** trong [src/ocp_planner/src/types.h](src/ocp_planner/src/types.h):

```cpp
// Time parameters
constexpr double kDT = 0.25;          // Timestep (s)
constexpr int kNP = 10;               // Prediction horizon

// Kinematics limits
constexpr double kMaxLinearVel = 1.0;    // m/s
constexpr double kMaxLinearAcc = 1.0;    // m/s²
constexpr double kMaxAngularVel = 3.0;   // rad/s
constexpr double kMaxAngularAcc = 3.0;   // rad/s²

// Safety parameters
constexpr double kSafetyMargin = 0.2;    // m
constexpr double kRobotRadius = 0.3;     // m

// Cost weights
constexpr double kWeightPos = 10.0;      // Position tracking
constexpr double kWeightVel = 1.0;       // Velocity tracking
constexpr double kWeightYaw = 1.0;       // Heading tracking
constexpr double kWeightControl = 0.1;   // Control effort
constexpr double kWeightSmooth = 0.5;    // Control smoothness
```

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue 1: MPC solver fails frequently**
- Kiểm tra IPOPT installation và MA57 solver
- Tăng `max_iter` trong MPC solver options
- Giảm prediction horizon `NP`
- Kiểm tra constraints có khả thi không

**Issue 2: Training không converge**
- Giảm learning rate
- Tăng entropy coefficient (khuyến khích exploration)
- Kiểm tra reward function scaling
- Sử dụng curriculum learning

**Issue 3: Collision rate cao**
- Tăng `discomfort_penalty_factor`
- Tăng `safety_margin` trong MPC
- Kiểm tra obstacle avoidance constraints
- Review action masking logic

**Issue 4: Import ocp_planner_py failed**
- Kiểm tra build C++ module thành công
- Set `LD_LIBRARY_PATH` đúng
- Kiểm tra pybind11 compatibility với Python version

### 8.2 Debugging Tips

```bash
# Enable verbose logging
python train_ppo.py --debug

# Visualize single episode
python eval_ppo.py --n_eval_episodes 1 --visualize

# Check MPC solver output
# Set verbose=1 trong Planner constructor

# Profile performance
python -m cProfile -o profile.stats train_ppo.py
python -m pstats profile.stats
```

---

## 9. Performance Benchmarks

### 9.1 Simulation Results

**Test Environment**: 5 humans, 3 obstacles, phase=10

| Method | Success Rate | Collision Rate | Avg Time (s) |
|--------|-------------|----------------|--------------|
| A* + MPC | 72.3% | 15.4% | 18.5 |
| A* + MPC + AM | 84.1% | 8.2% | 16.2 |
| **HALO (Ours)** | **91.8%** | **3.5%** | **14.7** |

### 9.2 Computational Performance

- **Policy Inference**: ~5ms (GPU), ~15ms (CPU)
- **MPC Solve**: ~50-100ms (depends on complexity)
- **Total Control Frequency**: ~10Hz

### 9.3 Training Statistics

- **Total Training Steps**: 5M
- **Training Time**: ~12 hours (RTX 3090)
- **Model Size**: ~5MB (GNN parameters)
- **Buffer Memory**: ~2GB (n_steps=2048)

---

## 10. Tài liệu Tham khảo

### 10.1 Paper

```bibtex
@misc{liu2025hierarchicallearningenhancedmpcsafe,
      title={Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation with Heterogeneous Constraints}, 
      author={Huajian Liu and Yixuan Feng and Wei Dong and Kunpeng Fan and Chao Wang and Yongzhuo Gao},
      year={2025},
      eprint={2506.09859},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.09859}, 
}
```

### 10.2 Key Algorithms

- **PPO**: Proximal Policy Optimization (Schulman et al., 2017)
- **GNN**: Graph Neural Networks for multi-agent scenarios
- **MPC**: Model Predictive Control với nonlinear optimization
- **ORCA**: Optimal Reciprocal Collision Avoidance (van den Berg et al., 2011)
- **GAE**: Generalized Advantage Estimation (Schulman et al., 2015)

### 10.3 Software Libraries

- **PyTorch**: Deep learning framework
- **DGL**: Deep Graph Library
- **Gym**: Reinforcement learning environment
- **CasADi**: Optimization framework
- **IPOPT**: Interior Point Optimizer
- **pybind11**: Python-C++ binding

---

## 11. Future Work & Extensions

### 11.1 Potential Improvements

1. **Multi-robot coordination**: Extend to multiple robots
2. **Recurrent policy**: LSTM/GRU for partial observability
3. **End-to-end learning**: Learn MPC parameters
4. **Real-world deployment**: Transfer to real robots
5. **Dynamic environments**: Handle non-stationary obstacles

### 11.2 Research Directions

- Transfer learning từ simulation sang real-world
- Meta-learning cho adaptive navigation
- Investigate social navigation behaviors
- Integration với semantic maps

---

## 📧 Contact & Support

For questions and issues:
- **GitHub Issues**: [HALO Issues](https://github.com/TIB-K330/HALO/issues)
- **Email**: Xem trong paper
- **Video Demos**: 
  - [Bilibili](https://www.bilibili.com/video/BV166MizgEht/)
  - [YouTube](https://www.youtube.com/watch?v=oymtbh-l1eM)

---

**Chúc bạn thành công với HALO Project! 🚀**
