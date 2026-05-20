# Mapping cong thuc ly thuyet sang code trong HALO_1

Tai lieu nay map cac cong thuc trong phan ly thuyet RL/GNN/PPO/kinematic sang cac doan code hien co trong repo `HALO_1`. Mot so cong thuc la nen tang ly thuyet nen khong co mot ham code rieng, ma duoc hien thuc gian tiep trong rollout, buffer, policy/value network, reward va planner.

## 1. MDP va vong lap Agent - Environment

### Cong thuc/khai niem

- Trang thai: `S_t`
- Hanh dong: `A_t`
- Thuong: `R_{t+1}`
- Trang thai tiep theo: `S_{t+1}`
- Tuong tac: agent chon action, environment tra ve `(observation, reward, done, info)`

### Code tuong ung

- Vong lap rollout PPO/MPC:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:406-425`
  - `actions, values, log_probs = self.policy(...)`
  - `wheel_acc = self.run_solver(...)`
  - `new_ob, reward, done, info = env.step(ActionDiff(...))`

- Chuyen observation thanh graph state:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:476-479`
  - `self._raw_last_state = JointState(new_robot_state, new_ob)`
  - `self._last_state = joint_state_as_graph(...)`

- Ham tao graph tu `JointState`:
  - `HALO_1/drl_moudle/modules/utils.py:67-71`

- Moi truong cap nhat state va tra ve reward:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1331-1439`

### Ghi chu

Cong thuc dong hoc MDP:

```tex
p(s', r | s, a) = P(S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a)
```

khong duoc code thanh bang xac suat ro rang. Repo nay la model-free/on-policy PPO: du lieu mau `(s, a, r, done, value, log_prob)` duoc thu truc tiep tu simulation thay vi hoc hoac luu `p(s', r | s, a)`.

## 2. Return chiet khau `G_t`

### Cong thuc

```tex
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
```

### Code tuong ung

- Tinh return de log episode:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:435-442`
  - `step_return = sum([pow(self.gamma, t) * reward for t, reward in enumerate(self.rewards[step:])])`

- Gia tri `gamma`:
  - Mac dinh trong `MpcPPO.__init__`: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:73`
  - Gan vao buffer: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:137-143`

### Ghi chu

Return dung de train PPO khong chi la Monte Carlo return thuan, ma la target tu GAE:

```python
self.returns = self.advantages + self.values
```

o `HALO_1/drl_moudle/modules/mpc_ppo_core.py:117-122`.

## 3. Reward function trong environment

### Cong thuc ly thuyet

```tex
R_{t+1} \in R
```

Trong code, reward gom nhieu thanh phan: tien ve goal, den dich, va cham, safety penalty, huong robot, RVO penalty, trajectory reward tu privileged learning.

### Code tuong ung

- Cau hinh trong training:
  - `HALO_1/drl_moudle/train_ppo.py:55-61`

- Luu config vao environment:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:147-153`

- Tinh reward chinh:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:923-925`
  - `reward = reward_arrival + weight_goal * reward_goal + re_theta * reward_theta + reward_time`
  - `constraint = reward_col + weight_safe * safety_penalty`

- Reward sau khi step, them RVO va scale:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1431-1435`
  - `reward = 100 * (reward + constraint + self.re_rvo * rvo_reward)`
  - `reward = reward + trajectory_reward`

- Reward du doan trajectory cho privileged learning:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1360-1367`

## 4. Policy `pi(a|s)`

### Cong thuc

```tex
\pi(a|s) = P(A_t=a | S_t=s)
```

### Code tuong ung

- Actor tao logits action:
  - `HALO_1/drl_moudle/modules/graph_ppo_core.py:417-430`
  - `mean_actions = self.action_net(latent_pi)`
  - voi action roi rac, `mean_actions` la logits cho categorical distribution.

- Distribution roi rac:
  - `HALO_1/drl_moudle/modules/distributions.py:244-287`
  - `CategoricalMasked(logits=action_logits, masks=action_masks)`
  - `sample()` lay action ngau nhien theo policy.
  - `mode()` lay action co xac suat lon nhat.

- Masked policy trong MpcPPO:
  - `HALO_1/drl_moudle/modules/mpc_ppo_core.py:198-220`
  - `distribution = self._get_action_dist_from_latent(latent_pi, action_masks=action_mask)`
  - `actions = distribution.get_actions(...)`

- Action space 9x9 = 81 local goals:
  - `HALO_1/drl_moudle/train_ppo.py:83-90`
  - `env.action_space = gym.spaces.Discrete(action_dim * action_dim)`

- Map action index sang local goal:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:676-685`

## 5. Value function `V(s)` va Critic

### Cong thuc

```tex
v_\pi(s) = E_\pi[G_t | S_t=s]
```

### Code tuong ung

- Critic network output mot scalar value:
  - `HALO_1/drl_moudle/modules/graph_ppo_core.py:323`
  - `self.value_net = nn.Linear(..., 1)`

- Forward tinh `values`:
  - `HALO_1/drl_moudle/modules/mpc_ppo_core.py:213-220`
  - `values = self.value_net(latent_vf)`

- Du doan value cho terminal/bootstrap:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:454-460`
  - `terminal_value = self.policy.predict_values(terminal_state)`

- Du doan value buoc cuoi rollout:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:481-485`

### Ghi chu

Repo khong dung `q_\pi(s,a)` trong PPO. PPO nay la Actor-Critic voi `V(s)` va advantage, khong phai Q-learning.

## 6. Advantage function va GAE

### Cong thuc

```tex
A(s,a) = Q(s,a) - V(s)
```

Trong PPO thuc te, advantage duoc uoc luong bang GAE:

```tex
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

```tex
\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
```

### Code tuong ung

- GAE trong `MaskedRolloutBuffer`:
  - `HALO_1/drl_moudle/modules/mpc_ppo_core.py:87-122`
  - `delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]`
  - `last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam`
  - `self.advantages[step] = last_gae_lam`
  - `self.returns = self.advantages + self.values`

- Goi tinh returns/advantage sau khi rollout:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:481-485`

- Normalize advantage truoc khi update:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:522-527`

## 7. PPO clipped surrogate objective

### Cong thuc

```tex
r_t(\theta) =
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}
```

```tex
L^{CLIP} =
E_t[
\min(r_t \hat{A}_t,
clip(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)
]
```

### Code tuong ung

- Tinh log prob moi:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:516-520`

- Tinh ratio:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:528-529`
  - `ratio = th.exp(log_prob - rollout_data.old_log_prob)`

- Clipped surrogate:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:531-534`
  - `policy_loss_1 = advantages * ratio`
  - `policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)`
  - `policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()`

- `epsilon = clip_range`:
  - Mac dinh `0.2`: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:75`
  - schedule: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:495-496`

## 8. PPO total loss, value loss, entropy

### Cong thuc ly thuyet

```tex
L^{PPO}_{total}
= E[L^{CLIP} - c_1 L^{VF} + c_2 S[\pi_\theta]]
```

### Code tuong ung

Trong code dang minimize loss, nen dau duoc viet theo convention PyTorch:

- Value loss:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:541-550`
  - `value_loss = F.mse_loss(rollout_data.returns, values_pred)`

- Entropy loss:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:553-560`
  - `entropy_loss = -th.mean(entropy)`

- Total loss:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:562`
  - `loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss`

- He so:
  - `ent_coef`, `vf_coef`: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:78-79`
  - Khi train: `ent_coef=0.001` o `HALO_1/drl_moudle/train_ppo.py:98-105`

- Backprop va update:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:579-584`

## 9. Bellman expectation/optimality equations

### Cong thuc

```tex
v_\pi(s) =
\sum_a \pi(a|s)
\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]
```

```tex
v^*(s) = \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v^*(s')]
```

### Code tuong ung

Khong co value iteration/policy iteration de giai Bellman truc tiep. Trong repo:

- Bellman backup xuat hien gian tiep trong TD error cua GAE:
  - `HALO_1/drl_moudle/modules/mpc_ppo_core.py:117`
  - `delta = reward + gamma * next_value - current_value`

- `max_a q*(s,a)` khong duoc code trong PPO training. Khi inference deterministic, action duoc lay theo mode cua distribution:
  - `HALO_1/drl_moudle/modules/distributions.py:286-287`
  - `return th.argmax(self.distribution.probs, dim=1)`

Day la greedy theo xac suat policy da hoc, khong phai greedy tren `q^*`.

## 10. GNN: bieu dien do thi

### Cong thuc/khai niem

```tex
G = (V, E)
```

Node: robot, human, obstacle, wall. Edge: tuong tac giua cac thuc the.

### Code tuong ung

- Tao graph tu `JointState`:
  - `HALO_1/drl_moudle/modules/utils.py:67-71`

- Khai bao loai node/edge:
  - `HALO_1/drl_moudle/modules/crowd_graph.py:10-21`
  - `rels = ['h2r', 'o2r', 'w2r', 'o2h', 'w2h', 'h2h']`

- Bien doi state sang RVO/robot-centric features:
  - `HALO_1/drl_moudle/modules/crowd_graph.py:37-81`

- Feature cua robot:
  - `HALO_1/drl_moudle/modules/crowd_graph.py:85-97`
  - gom van toc banh trai/phai, khoang cach den dich, v_pref, heading tuong doi.

- Feature cua human/obstacle/wall voi VO/RVO:
  - `HALO_1/drl_moudle/modules/crowd_graph.py:132-175`

- Gan node features:
  - `HALO_1/drl_moudle/modules/crowd_graph.py:192-224`

- Tao edge:
  - obstacle -> robot: `HALO_1/drl_moudle/modules/crowd_graph.py:232-243`
  - human -> robot: `HALO_1/drl_moudle/modules/crowd_graph.py:244-253`
  - wall -> robot: `HALO_1/drl_moudle/modules/crowd_graph.py:255-265`
  - obstacle/wall -> human: `HALO_1/drl_moudle/modules/crowd_graph.py:267-289`
  - human <-> human: `HALO_1/drl_moudle/modules/crowd_graph.py:291-309`

- Tao DGL graph:
  - `HALO_1/drl_moudle/modules/crowd_graph.py:314-316`

## 11. GNN message passing

### Cong thuc

```tex
h_i^{(l+1)}
= UPDATE(h_i^{(l)}, AGGREGATE({h_j^{(l)}, j in N(i)}))
```

### Code tuong ung

Repo hien dang dung `GATConv`, khong dung class `GNNFiLMLayer` trong forward mac dinh.

- Feature extractor:
  - `HALO_1/drl_moudle/modules/torch_layers.py:107-152`
  - `z1 = self.encoder(node_features)`
  - `sub_graph = dgl.add_self_loop(sub_graph)`
  - `h1 = self.gnn(sub_graph, z1)`
  - `return h1 + z1`

- GAT layer:
  - `HALO_1/drl_moudle/modules/torch_layers.py:130`
  - `self.gnn = GATConv(...)`

- Ban GNN-FiLM co message passing ro rang nhung dang bi comment trong `GraphExtractor`:
  - `HALO_1/drl_moudle/modules/gnn_models.py:59-76`
  - `graph.update_all(self.message_func, fn.sum('m', 'h'))`

## 12. Noi graph embedding voi robot state de tao observation co kich thuoc co dinh

### Cong thuc

```tex
H = Concat(h_0, h_0^{GNN})
```

### Code tuong ung

- Lay robot raw state va GNN embedding, roi concat:
  - `HALO_1/drl_moudle/modules/graph_ppo_core.py:376-415`
  - single graph:
    - `robot_state = cur_state.ndata['h'][0, 4:9].clone()`
    - `pi_features = self.pi_features_extractor(cur_state)[0, :]`
    - `pi_features = th.cat((robot_state, pi_features), dim=0)`
  - batch:
    - lay `robot_ids`
    - `robot_states = robot_features[:, 4:9].clone()`
    - `th.cat((robot_states, pi_features), dim=1)`

- `features_dim + 5`:
  - `HALO_1/drl_moudle/modules/graph_ppo_core.py:250-251`
  - `self.features_dim = self.features_extractor.features_dim + 5`

## 13. Actor-Critic architecture

### Cong thuc/khai niem

- Actor sinh `pi_theta(a|s)`
- Critic sinh `V_phi(s)`

### Code tuong ung

- Default net architecture:
  - `HALO_1/drl_moudle/modules/graph_ppo_core.py:235-237`
  - `net_arch = dict(pi=[256, 256], vf=[256, 256])`

- Tach actor/critic MLP:
  - `HALO_1/drl_moudle/modules/torch_layers.py:52-104`
  - `forward_actor()`
  - `forward_critic()`

- Actor head:
  - `HALO_1/drl_moudle/modules/graph_ppo_core.py:316-320`
  - `self.action_net = ...`

- Critic head:
  - `HALO_1/drl_moudle/modules/graph_ppo_core.py:323`
  - `self.value_net = nn.Linear(..., 1)`

## 14. Local goal action va MPPI/MPC backend

### Khai niem trong doan ly thuyet

Actor chon mot local goal roi planner/MPC sinh dieu khien thap hon.

### Code tuong ung

- Actor chon action index:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:409-415`

- Chuyen action index thanh local goal:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:676-685`

- Chuyen local goal tu robot frame sang map frame:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:693-700`

- Goi OCP/MPC solver:
  - Python pybind: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:260-266`
  - ROS service: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:156-258`

- Ket qua solver la gia toc banh trai/phai:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:276-284`
  - return `np.array((ans.al, ans.ar))`

## 15. Bo sung AI: khong gian hanh dong local-goal roi rac

### Ly thuyet can bo sung

Trong code nay, Actor khong hoc truc tiep lenh dieu khien thap nhu van toc, gia toc banh, hay PWM. Actor hoc mot chinh sach roi rac tren tap cac muc tieu cuc bo:

```tex
\mathcal{A} = \{g_{ij} = (x_i, y_j)\mid i,j=1,\dots,N\}
```

voi `N = action_dim`. Neu `action_dim = 9`, tong so hanh dong la:

```tex
|\mathcal{A}| = N^2 = 81
```

Chinh sach Actor sinh phan phoi:

```tex
\pi_\theta(g_{ij}|s)
```

Sau khi lay mau hoac chon tham lam mot `action_index`, he thong anh xa index do thanh toa do local goal trong he quy chieu robot. Local goal nay tiep tuc duoc dung lam dau vao cho bo lap ke hoach cuc bo.

### Code tuong ung

- Khai bao action dimension:
  - `HALO_1/drl_moudle/train_ppo.py:83-90`
  - `action_dim = args.action_dim`
  - `env.action_space = gym.spaces.Discrete(action_dim * action_dim)`

- Gia tri mac dinh `action_dim = 9`, nen co 81 hanh dong:
  - `HALO_1/drl_moudle/train_ppo.py:200`

- Actor lay action index:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:409-415`
  - `actions, values, log_probs = self.policy(...)`
  - `env.set_action_index(actions[0])`

- Map action index sang local goal:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:676-685`
  - `action_values = np.linspace(min_goal_coord, max_goal_coord, self.num_actions_per_dim)`
  - `x_index = action_index // self.num_actions_per_dim`
  - `y_index = action_index % self.num_actions_per_dim`

### Nen them vao ly thuyet

Nen them mot tieu muc sau PPO hoac trong phan kien truc Actor-Critic:

```tex
\pi_\theta(a|s),\quad a \in \{0,1,\dots,N^2-1\}
```

```tex
a \mapsto g(a) = (x_i, y_j)
```

Trong do `g(a)` la local goal ung voi hanh dong roi rac `a`.

## 16. Bo sung AI: Action Masking cho PPO roi rac

### Ly thuyet can bo sung

Code co co che Action Mask de loai bo cac local goal khong hop le truoc khi lay mau tu policy. Ve mat xac suat, neu `m(s,a)` la mask cua action `a` tai state `s`, thi:

```tex
m(s,a) =
\begin{cases}
1, & a \text{ hop le}\\
0, & a \text{ khong hop le}
\end{cases}
```

Chinh sach sau mask co the mo ta:

```tex
\pi_\theta^{mask}(a|s) =
\frac{m(s,a)\pi_\theta(a|s)}
{\sum_{a'}m(s,a')\pi_\theta(a'|s)}
```

Trong code, mask co them gia tri dac biet `2.0` de uu tien action gan goal khi robot da o gan dich. Logit cua action mask `0.0` bi day ve gia tri rat am, con mask `2.0` bi day ve gia tri rat duong.

### Code tuong ung

- Tinh action mask trong environment:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1167-1240`

- Cac dieu kien mask:
  - Vuot tam local MPC: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1192-1195`
  - Qua gan obstacle tron: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1197-1202`
  - Qua gan wall: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1205-1210`
  - Nam trong polygon obstacle: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1213-1217`
  - Vuot boundary: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1220-1229`
  - Uu tien action gan goal: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1232-1240`

- Dua mask vao policy khi rollout:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:407-410`
  - `action_mask = env.action_mask`
  - `self.policy(self._last_state, action_mask=action_mask)`

- Luu mask vao rollout buffer:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:467-475`

- Dung mask lai khi tinh log-prob trong PPO update:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:516-520`

- Mask logits trong categorical distribution:
  - `HALO_1/drl_moudle/modules/distributions.py:18-35`
  - `logits = th.where(self.masks == 0.0, -1e+8, ...)`
  - `logits = th.where(self.masks == 2.0, 1e+8, ...)`

### Nen them vao ly thuyet

Nen them mot tieu muc "Action Masking" sau phan PPO, vi day la mot cai tien quan trong cho navigation an toan. Action Mask giup tac nhan khong lang phi mau vao cac local goal ro rang khong kha thi, dong thoi giam nguy co va cham trong qua trinh exploration.

## 17. Bo sung AI: Reward shaping thuc te trong code

### Ly thuyet can bo sung

Phan reward trong ly thuyet hien moi dung reward hypothesis tong quat. Trong code, reward cua bai toan dieu huong duoc shaping thanh nhieu thanh phan:

```tex
r_t =
100\left(
r_{arrival}
+ w_g r_{goal}
+ w_\theta r_\theta
+ r_{collision}
+ w_s r_{safe}
+ w_{rvo}r_{rvo}
\right)
+ r_{traj}
```

Trong do:

- `r_arrival`: thuong khi den dich.
- `r_goal`: tien bo ve gan goal.
- `r_theta`: phat/thuong theo huong robot so voi huong den goal.
- `r_collision`: phat khi va cham.
- `r_safe`: phat khi vi pham khoang cach an toan.
- `r_rvo`: phat theo nguy co va cham tu Velocity Obstacle/RVO.
- `r_traj`: reward phu dua tren danh gia trajectory tuong lai khi bat privileged learning.

### Code tuong ung

- Hyperparameters reward tu command line:
  - `HALO_1/drl_moudle/train_ppo.py:194-199`
  - `safe_weight`, `goal_weight`, `re_collision`, `re_arrival`, `re_rvo`, `re_theta`

- Gan vao config environment:
  - `HALO_1/drl_moudle/train_ppo.py:55-61`

- Thanh phan reward chinh:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:904-925`
  - `reward = reward_arrival + weight_goal * reward_goal + re_theta * reward_theta + reward_time`
  - `constraint = reward_col + weight_safe * safety_penalty`

- RVO reward:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1136-1165`
  - Duoc cong vao reward tai `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1431-1433`

- Scale reward:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1433`
  - `reward = 100 * (reward + constraint + self.re_rvo * rvo_reward)`

### Nen them vao ly thuyet

Nen them mot tieu muc "Ham phan thuong cho bai toan dieu huong" thay vi chi noi reward tong quat. Dieu nay giup nguoi doc hieu tac nhan PPO hoc tu tin hieu nao trong code.

## 18. Bo sung AI: Privileged Learning va trajectory reward

### Ly thuyet can bo sung

Code co co che privileged learning (`use_PL`) de danh gia them cac trang thai du doan tren trajectory cua robot. Neu bo lap ke hoach sinh mot chuoi trang thai tuong lai, environment cong them reward:

```tex
r_{traj} =
\sum_{i=1}^{K-1}\gamma_{PL}^{i} r(s_{t+i})
```

Trong do:

- `K = PL_traj_length`
- `gamma_PL = PL_traj_gamma`
- `r(s_{t+i})` duoc tinh bang ham `eval_state(i)`, gom heading reward, collision/safety va RVO risk tren trang thai du doan.

### Code tuong ung

- Bat/tat privileged learning trong training:
  - `HALO_1/drl_moudle/train_ppo.py:91-94`
  - `env.use_PL = args.use_PL`
  - `env.PL_traj_length = args.PL_traj_length`
  - `env.PL_traj_gamma = args.PL_traj_gamma`

- Tham so mac dinh:
  - `HALO_1/drl_moudle/train_ppo.py:204-206`
  - `use_PL=True`, `PL_traj_length=4`, `PL_traj_gamma=0.9`

- Cong trajectory reward:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1360-1367`
  - `trajectory_reward += state_reward * pow(r_gamma, i)`

- Ham danh gia state du doan:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:943-1073`

- Cong vao reward cuoi:
  - `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1433-1435`

### Nen them vao ly thuyet

Nen them vao phan reward shaping hoac phan "PPO ket hop planner". Diem nay quan trong vi tac nhan khong chi hoc tu hau qua mot buoc, ma con nhan tin hieu tu chat luong trajectory ngan han do planner sinh ra.

## 19. Bo sung AI: GAT thay cho message passing tong quat

### Ly thuyet can bo sung/sua

Phan GNN hien viet dung theo dang message passing tong quat. Tuy nhien code dang chay dung Graph Attention Network (`GATConv`), nen nen noi ro aggregation duoc hoc bang attention:

```tex
h_i' =
\sigma\left(
\sum_{j\in\mathcal{N}(i)}
\alpha_{ij} W h_j
\right)
```

voi:

```tex
\alpha_{ij} =
\text{softmax}_j(e_{ij})
```

`alpha_ij` la trong so attention cho biet nut hang xom `j` anh huong den nut `i` manh den muc nao.

### Code tuong ung

- Khoi tao GAT:
  - `HALO_1/drl_moudle/modules/torch_layers.py:130`
  - `self.gnn = GATConv(self.encoder_dim, self._features_dim, num_heads=1, activation=torch.nn.ReLU())`

- Forward GNN:
  - `HALO_1/drl_moudle/modules/torch_layers.py:140-152`
  - `z1 = self.encoder(node_features)`
  - `sub_graph = dgl.add_self_loop(sub_graph)`
  - `h1 = self.gnn(sub_graph, z1)`
  - `return h1 + z1`

- `GNNFiLMLayer` co message passing tu viet tay nhung khong phai duong chay hien tai:
  - `HALO_1/drl_moudle/modules/gnn_models.py:59-76`
  - Trong `GraphExtractor`, phan dung `GNNFiLMLayer` dang bi comment tai `HALO_1/drl_moudle/modules/torch_layers.py:122-128`

### Nen them vao ly thuyet

Nen sua cau "AGGREGATE va UPDATE duoc trien khai bang FNN" thanh: "Trong trien khai nay, aggregation duoc thuc hien bang GATConv cua DGL, trong do moi canh/hang xom duoc gan trong so attention hoc duoc. GNN-FiLM la mot phuong an co trong code nhung khong duoc kich hoat trong cau hinh hien tai."

## 20. Bo sung AI: Action mask anh huong entropy va PPO update

### Ly thuyet can bo sung

Khi dung action mask, entropy khong tinh tren toan bo `N^2` hanh dong ma tinh tren phan phoi sau khi mask. Do do entropy khuyen khich exploration trong tap hanh dong hop le:

```tex
S[\pi_\theta^{mask}](s)
= -\sum_{a\in\mathcal{A}_{valid}(s)}
\pi_\theta^{mask}(a|s)
\log \pi_\theta^{mask}(a|s)
```

PPO ratio cung phai duoc tinh tu log-prob voi cung mask tai thoi diem update:

```tex
r_t(\theta)
= \exp(
\log \pi_\theta^{mask}(a_t|s_t)
- \log \pi_{\theta_{old}}^{mask}(a_t|s_t)
)
```

### Code tuong ung

- Luu old log-prob khi rollout co mask:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:409-410`
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:467-475`

- Khi train, evaluate action voi `action_masks=rollout_data.action_masks`:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:516-520`

- Entropy cua masked categorical:
  - `HALO_1/drl_moudle/modules/distributions.py:37-46`

- PPO ratio dung log-prob sau mask:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:528-534`

### Nen them vao ly thuyet

Nen them vao phan PPO de tranh hieu nham rang entropy va ratio duoc tinh tren tat ca action. Voi bai toan nay, PPO update tren phan phoi da duoc rang buoc boi mask.

## 21. Dong hoc vi sai: forward kinematics

### Cong thuc trong ly thuyet

```tex
v = (v_R + v_L) / 2
```

```tex
\omega = (v_R - v_L) / B
```

### Code tuong ung

Repo dung `radius` nhu nua khoang cach hai banh. Vi vay:

```python
v = 0.5 * (v_left + v_right)
yaw_rate = 0.5 * (v_right - v_left) / radius
```

Neu so voi cong thuc `omega = (v_R - v_L) / B` thi `B = 2 * radius`.

- Trong input cho OCP planner:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:293-298`

- Trong ROS main:
  - `HALO_1/src/ocp_planner/src/main.cc:37-40`

- Trong robot trajectory RK2:
  - `HALO_1/drl_moudle/crowd_sim/envs/utils/robot.py:96-98`

- Trong state update robot:
  - `HALO_1/drl_moudle/crowd_sim/envs/utils/agent.py:244-248`

## 22. Dong hoc vi sai: update trang thai tu gia toc banh

### Code tuong ung voi mo phong robot

Action cua robot la:

```python
ActionDiff(al, ar)
```

trong do `al`, `ar` la gia toc banh trai/phai.

- Cap nhat van toc banh:
  - `HALO_1/drl_moudle/crowd_sim/envs/utils/agent.py:210-219`
  - `vel_left = self.v_left + left_acc * self.time_step`
  - `vel_right = self.v_right + right_acc * self.time_step`

- Gioi han van toc banh:
  - `HALO_1/drl_moudle/crowd_sim/envs/utils/agent.py:215-219`

- Tinh quang duong banh:
  - `HALO_1/drl_moudle/crowd_sim/envs/utils/agent.py:221-225`

- Tinh thay doi goc:
  - `HALO_1/drl_moudle/crowd_sim/envs/utils/agent.py:226-228`
  - `d_theta = (s_right - s_left) / (2 * self.radius)`

- Cap nhat vi tri:
  - `HALO_1/drl_moudle/crowd_sim/envs/utils/agent.py:237-243`

### Ghi chu

Day la dynamic/kinematic simulation bang gia toc banh, khong phai PID velocity tracking.

## 23. OCP/MPC dynamics va RK2

### Cong thuc trong code

State MPC:

```tex
x = [X, Y, \phi, v, r]
```

Input:

```tex
u = [acc, dr, slack]
```

Lien tuc:

```tex
\dot{x} = [v cos(phi), v sin(phi), r, acc, dr]
```

### Code tuong ung

- Model lien tuc:
  - `HALO_1/src/ocp_planner/src/mpc.cc:187-198`

- RK2:
  - `HALO_1/src/ocp_planner/src/mpc.cc:209-218`

- Rang buoc dynamics trong optimization:
  - `HALO_1/src/ocp_planner/src/mpc.cc:280-290`

- Params MPC:
  - `HALO_1/src/ocp_planner/src/types.h:29-35`
  - `kNP = 10`, `kDT = 0.25`, max vel/acc.

## 24. Inverse kinematics tu acc/dr sang gia toc banh

### Cong thuc gan voi code

Trong C++ planner, input MPC la:

```tex
acc = gia toc tinh tien
dr = gia toc yaw-rate
```

Output ve Python/ROS la:

```tex
a_L = acc - dr * radius
```

```tex
a_R = acc + dr * radius
```

voi `radius = 0.3`.

### Code tuong ung

- ROS response:
  - `HALO_1/src/ocp_planner/src/main.cc:118-127`
  - `res->al = acc - dr * 0.3`
  - `res->ar = acc + dr * 0.3`

- Python output tu pybind:
  - `HALO_1/drl_moudle/algorithms/mpc_ppo.py:276-284`

- MPC constraints tren van toc/gia toc tung banh:
  - `HALO_1/src/ocp_planner/src/mpc.cc:319-322`
  - `-1.0 <= X(3,k) - X(4,k)*0.3 <= 1.0`
  - `-1.0 <= X(3,k) + X(4,k)*0.3 <= 1.0`
  - `-1.0 <= U(0,k) - U(1,k)*0.3 <= 1.0`
  - `-1.0 <= U(0,k) + U(1,k)*0.3 <= 1.0`

## 25. Cong thuc PID gia so va dynamic scaling PWM

### Cong thuc trong ly thuyet

```tex
\Delta u(k) =
K_p[e(k)-e(k-1)]
+ K_iT_s e(k)
+ K_d/T_s[e(k)-2e(k-1)+e(k-2)]
```

```tex
\alpha = u_{max} / max(|u_L|, |u_R|)
```

### Tinh trang trong code hien tai

Khong thay implementation truc tiep cua:

- `e_v = v_ref - v`
- `e_omega = omega_ref - omega`
- incremental PID voi `Kp`, `Ki`, `Kd`
- PWM saturation `u_max`
- dynamic scaling `alpha`
- back-calculation anti-windup cho `u_v`, `u_omega`

Phan gan nhat trong repo la:

- Gioi han van toc banh trong simulation:
  - `HALO_1/drl_moudle/crowd_sim/envs/utils/agent.py:215-219`

- Gioi han input/velocity trong MPC:
  - `HALO_1/src/ocp_planner/src/mpc.cc:319-322`

- Chuyen `acc/dr` sang `al/ar`:
  - `HALO_1/src/ocp_planner/src/main.cc:118-127`

Do do, neu dua phan PID vao luan van thi nen ghi ro day la co so cho tang dieu khien thap/hardware, con trong `HALO_1` ban hien tai dang dung MPC/OCP sinh `al/ar`, sau do simulation cap nhat robot bang dynamic differential model.

## 26. Tom tat mapping nhanh

| Ly thuyet | Code chinh |
|---|---|
| `S_t`, `A_t`, `R_{t+1}`, `S_{t+1}` | `algorithms/mpc_ppo.py:406-425`, `crowd_sim.py:1331-1439` |
| `G_t = sum gamma^k R` | `algorithms/mpc_ppo.py:435-442` |
| GAE `delta`, `A_hat` | `modules/mpc_ppo_core.py:87-122` |
| Policy `pi(a|s)` | `modules/mpc_ppo_core.py:198-220`, `modules/distributions.py:244-287` |
| Value `V(s)` | `modules/graph_ppo_core.py:323`, `modules/mpc_ppo_core.py:213-220` |
| PPO ratio | `algorithms/mpc_ppo.py:528-529` |
| PPO clip loss | `algorithms/mpc_ppo.py:531-534` |
| Value loss | `algorithms/mpc_ppo.py:541-550` |
| Entropy | `algorithms/mpc_ppo.py:553-560` |
| Total loss | `algorithms/mpc_ppo.py:562` |
| Local-goal action space `N^2` | `train_ppo.py:83-90`, `algorithms/mpc_ppo.py:676-685` |
| Action Masking | `crowd_sim.py:1167-1240`, `modules/distributions.py:18-35` |
| Masked PPO update | `algorithms/mpc_ppo.py:516-520`, `algorithms/mpc_ppo.py:528-534` |
| Reward shaping | `crowd_sim.py:904-925`, `crowd_sim.py:1431-1435` |
| Privileged trajectory reward | `crowd_sim.py:1360-1367`, `crowd_sim.py:943-1073` |
| Graph nodes/edges | `modules/crowd_graph.py:128-316` |
| GNN/GAT message passing | `modules/torch_layers.py:140-152` |
| GATConv architecture | `modules/torch_layers.py:130` |
| `H = concat(robot, GNN)` | `modules/graph_ppo_core.py:376-415` |
| Local goal action | `algorithms/mpc_ppo.py:676-685` |
| MPC dynamics | `src/ocp_planner/src/mpc.cc:187-218` |
| Wheel velocity/acc relation | `algorithms/mpc_ppo.py:293-298`, `src/ocp_planner/src/main.cc:118-127` |
| PID/dynamic scaling | Chua co code truc tiep trong repo |

## 27. Noi dung de xuat dua thang vao bao cao thiet ke

Phan nay viet theo huong co the dua vao chuong thiet ke/thuc nghiem cua luan van. Cac cong thuc bam sat code AI hien co trong `HALO_1`, dac biet la `train_ppo.py`, `mpc_ppo.py`, `mpc_ppo_core.py`, `crowd_graph.py`, `torch_layers.py`, `distributions.py` va `crowd_sim.py`.

### 27.1. Tong quan kien truc AI

Trong he thong de xuat, tac nhan hoc tang cuong khong truc tiep xuat lenh dieu khien banh xe. Thay vao do, mang Actor hoc mot chinh sach roi rac tren tap cac muc tieu cuc bo (local goals). Voi moi trang thai quan sat, moi truong duoc ma hoa thanh mot do thi tuong tac gom robot, nguoi di bo, vat can tron va vat can dang tuong. Do thi nay duoc dua qua mang GNN/GAT de trich xuat dac trung khong gian - tuong tac. Dac trung sau GNN duoc noi voi dac trung rieng cua robot, sau do dua vao hai nhanh Actor va Critic cua PPO.

Actor sinh phan phoi xac suat tren tap local goals:

```tex
\pi_\theta(a_t|s_t), \quad a_t \in \{0,1,\dots,N^2-1\}
```

Critic uoc luong gia tri trang thai:

```tex
V_\phi(s_t)
```

Hanh dong duoc Actor chon ra la chi so cua local goal. Local goal nay duoc chuyen sang he toa do ban do va dua vao bo lap ke hoach cuc bo de sinh gia toc banh trai/phai cho robot. Theo cach thiet ke nay, PPO dam nhan nhiem ra quyet dinh chien luoc o muc local-goal, con bo lap ke hoach dam nhan viec tao dieu khien kha thi o muc thap.

Code tuong ung:

- Tao model PPO: `HALO_1/drl_moudle/train_ppo.py:98-111`
- Actor chon action va planner sinh dieu khien: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:406-425`
- Chuyen raw state thanh graph: `HALO_1/drl_moudle/modules/utils.py:67-71`

### 27.2. Khong gian hanh dong local-goal roi rac

Khong gian hanh dong cua tac nhan duoc xay dung bang cach roi rac hoa mot vung local goal trong he toa do gan voi robot. Gia su moi truc toa do duoc chia thanh `N` gia tri, tong so local goals la:

```tex
|\mathcal{A}| = N^2
```

Tap hanh dong co the viet:

```tex
\mathcal{A} =
\{a_{ij} \mid i,j = 0,1,\dots,N-1\}
```

Moi action index `a` duoc anh xa thanh toa do local goal:

```tex
i = \left\lfloor \frac{a}{N} \right\rfloor,
\quad
j = a \bmod N
```

```tex
g(a) = (x_i, y_j)
```

Trong do:

```tex
x_i, y_j \in [g_{min}, g_{max}]
```

Voi cau hinh mac dinh trong code:

```tex
N = 9,\quad |\mathcal{A}| = 9^2 = 81
```

Dieu nay co nghia Actor sinh phan phoi xac suat tren 81 local goals:

```tex
\pi_\theta(a|s),\quad a \in \{0,\dots,80\}
```

Code tuong ung:

- `action_dim=9`: `HALO_1/drl_moudle/train_ppo.py:200`
- `Discrete(action_dim * action_dim)`: `HALO_1/drl_moudle/train_ppo.py:83-90`
- `map_action_to_goal`: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:676-685`

### 27.3. Bieu dien trang thai bang do thi tuong tac

Moi truong xung quanh robot duoc bieu dien bang mot do thi:

```tex
\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)
```

Trong do tap nut gom cac thuc the:

```tex
\mathcal{V}_t =
\mathcal{V}^{robot}
\cup \mathcal{V}^{human}
\cup \mathcal{V}^{obstacle}
\cup \mathcal{V}^{wall}
```

Moi nut `i` co vector dac trung:

```tex
\mathbf{h}_i^0 \in \mathbb{R}^{d}
```

Trong code, vector dac trung nut bao gom one-hot loai nut va cac dac trung hinh hoc/dong hoc. Doi voi robot, dac trung gom van toc banh trai, van toc banh phai, khoang cach den dich, van toc mong muon va huong robot tuong doi so voi goal. Doi voi human/obstacle/wall, dac trung duoc mo rong voi thong tin tu Velocity Obstacle/RVO nhu vi tri VO, bien trai/phai cua VO, khoang cach nho nhat va thoi gian du doan va cham.

Tap canh bieu dien cac quan he tuong tac:

```tex
\mathcal{E}_t =
\{h2r, o2r, w2r, o2h, w2h, h2h\}
```

Trong do `h2r` la human-to-robot, `o2r` la obstacle-to-robot, `w2r` la wall-to-robot, va `h2h` la tuong tac giua nguoi di bo.

Code tuong ung:

- Khai bao relation types: `HALO_1/drl_moudle/modules/crowd_graph.py:21`
- Tao feature node: `HALO_1/drl_moudle/modules/crowd_graph.py:128-224`
- Tao edge: `HALO_1/drl_moudle/modules/crowd_graph.py:226-316`

### 27.4. Trich xuat dac trung bang Graph Attention Network

Trong thiet ke nay, GNN duoc hien thuc bang Graph Attention Network (GAT). Truoc tien, node feature ban dau duoc dua qua encoder:

```tex
\mathbf{z}_i = f_{enc}(\mathbf{h}_i^0)
```

Sau do, GAT tong hop thong tin tu cac nut lan can thong qua trong so attention:

```tex
\mathbf{h}_i^{GAT} =
\sigma
\left(
\sum_{j \in \mathcal{N}(i)}
\alpha_{ij}\mathbf{W}\mathbf{z}_j
\right)
```

Trong do `alpha_ij` la he so attention bieu dien muc do anh huong cua nut `j` den nut `i`:

```tex
\alpha_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{k\in\mathcal{N}(i)}\exp(e_{ik})}
```

Code con cong them residual connection giua embedding dau vao va dau ra GAT:

```tex
\mathbf{h}_i =
\mathbf{h}_i^{GAT} + \mathbf{z}_i
```

Residual connection giup bao toan thong tin cuc bo cua nut va lam qua trinh huan luyen on dinh hon.

Code tuong ung:

- Khoi tao `GATConv`: `HALO_1/drl_moudle/modules/torch_layers.py:130`
- Encoder va residual: `HALO_1/drl_moudle/modules/torch_layers.py:147-152`

Ghi chu cho bao cao: trong code co class `GNNFiLMLayer`, tuy nhien duong chay hien tai cua `GraphExtractor` dang dung `GATConv`. Vi vay neu viet bao cao theo code hien tai, nen goi la GAT/GNN feature extractor thay vi GNN-FiLM.

### 27.5. Tao vector quan sat co kich thuoc co dinh

Sau khi trich xuat embedding cho toan bo do thi, he thong lay embedding cua nut robot lam dai dien cho trang thai tuong tac xung quanh robot:

```tex
\mathbf{h}_{0}^{GNN}
```

Vector nay duoc noi voi dac trung rieng cua robot:

```tex
\mathbf{h}_{robot}
=
[v_L, v_R, d_g, v_{pref}, \theta_{rel}]
```

Vector quan sat cuoi cung dua vao Actor va Critic la:

```tex
\mathbf{H}_t =
\text{Concat}
\left(
\mathbf{h}_{robot},
\mathbf{h}_{0}^{GNN}
\right)
```

Day la buoc bien doi trang thai co so luong doi tuong thay doi thanh mot vector co kich thuoc co dinh, phu hop voi mang fully-connected cua Actor-Critic.

Code tuong ung:

- Single graph: `HALO_1/drl_moudle/modules/graph_ppo_core.py:381-390`
- Batch graph: `HALO_1/drl_moudle/modules/graph_ppo_core.py:392-415`
- `features_dim + 5`: `HALO_1/drl_moudle/modules/graph_ppo_core.py:250-251`

### 27.6. Actor-Critic network

Sau khi co vector quan sat `H_t`, Actor va Critic duoc xay dung thanh hai nhanh MLP rieng:

```tex
\mathbf{z}^{\pi}_t = f_{\pi}(\mathbf{H}_t)
```

```tex
\mathbf{z}^{V}_t = f_{V}(\mathbf{H}_t)
```

Actor tao logits cho phan phoi categorical tren tap hanh dong roi rac:

```tex
\mathbf{l}_t = W_{\pi}\mathbf{z}^{\pi}_t + b_{\pi}
```

```tex
\pi_\theta(a|s_t)
=
\text{Softmax}(\mathbf{l}_t)_a
```

Critic xuat mot gia tri vo huong:

```tex
V_\phi(s_t)
=
W_V\mathbf{z}^{V}_t + b_V
```

Trong code, cau hinh mac dinh cua hai nhanh Actor va Critic la hai lop an 256-256:

```tex
f_{\pi}: [256, 256],
\quad
f_V: [256, 256]
```

Code tuong ung:

- Kien truc MLP mac dinh: `HALO_1/drl_moudle/modules/graph_ppo_core.py:235-237`
- Actor/value MLP extractor: `HALO_1/drl_moudle/modules/torch_layers.py:52-104`
- Actor head va value head: `HALO_1/drl_moudle/modules/graph_ppo_core.py:316-323`

### 27.7. Action masking trong policy

De dam bao tac nhan khong chon cac local goal khong kha thi, he thong ap dung action mask phu thuoc trang thai. Goi mask la:

```tex
m_t(a) \in \{0,1\}
```

Trong do:

```tex
m_t(a)=0
```

neu local goal tuong ung voi action `a` khong hop le, vi du nam ngoai bien, qua gan vat can, qua gan tuong, nam trong vat can da giac, hoac vuot tam toi uu cua planner.

Chinh sach sau khi mask duoc chuan hoa lai:

```tex
\pi_\theta^{m}(a|s_t)
=
\frac{m_t(a)\pi_\theta(a|s_t)}
{\sum_{a'\in\mathcal{A}}m_t(a')\pi_\theta(a'|s_t)}
```

Neu `m_t(a)=0`, xac suat chon action do bang 0:

```tex
\pi_\theta^{m}(a|s_t)=0
```

Trong implementation, thay vi nhan truc tiep voi mask, code thay doi logits:

```tex
l_a =
\begin{cases}
-10^8, & m_t(a)=0\\
10^8, & m_t(a)=2\\
l_a, & \text{nguoc lai}
\end{cases}
```

Gia tri `m_t(a)=2` duoc dung nhu mot co che uu tien dac biet cho action gan goal khi robot da o gan dich.

Code tuong ung:

- Tao mask: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1167-1240`
- Mask logits: `HALO_1/drl_moudle/modules/distributions.py:18-35`
- Dua mask vao policy rollout: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:407-410`
- Dung mask khi PPO update: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:516-520`

### 27.8. Ham phan thuong thiet ke cho dieu huong an toan

Tin hieu phan thuong trong he thong duoc thiet ke de can bang giua tien do den dich, an toan va chat luong dinh huong cua robot. Reward mot buoc co dang tong quat:

```tex
r_t =
100
\left(
r_{arr}
+ w_g r_{goal}
+ w_\theta r_\theta
+ r_{col}
+ w_s r_{safe}
+ w_{rvo}r_{rvo}
\right)
+ r_{traj}
```

Thanh phan tien do den dich duoc tinh bang do giam khoang cach toi goal:

```tex
r_{goal}
=
\|p_t - p_g\|_2
-
\|p_{t+1} - p_g\|_2
```

Trong do `p_t` la vi tri robot tai thoi diem `t`, `p_{t+1}` la vi tri du doan sau khi thuc hien action, va `p_g` la vi tri dich.

Thanh phan dinh huong khuyen khich robot quay ve phia goal:

```tex
r_\theta
=
\frac{
\cos(\theta_t)\cos(\theta_g)
+ \sin(\theta_t)\sin(\theta_g)
- 1
}
{\|p_t-p_g\|_2 + d_\theta}
```

Trong do `theta_t` la huong hien tai cua robot, `theta_g` la goc tu robot den goal, va `d_theta` la hang so lam mem.

Thanh phan va cham:

```tex
r_{col} =
\begin{cases}
r_{collision}, & \text{neu va cham}\\
0, & \text{nguoc lai}
\end{cases}
```

Thanh phan an toan phat khi khoang cach nho hon nguong discomfort:

```tex
r_{safe}
=
\sum_i
\min(0, d_i - d_{safe})
```

Thanh phan RVO phat cac hanh vi co nguy co va cham trong tuong lai:

```tex
r_{rvo}
=
\sum_i \mathbb{I}_{VO_i}
\left(
c_0 - \frac{1}{t_i^{exp}+1} - c_1
\right)
```

Trong do `I_{VO_i}` cho biet robot co nam trong velocity obstacle cua doi tuong `i` hay khong, va `t_i^{exp}` la thoi gian va cham du doan.

Code tuong ung:

- Goal/theta/collision/safety reward: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:904-925`
- RVO reward: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1136-1165`
- Reward sau scale: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1431-1435`

### 27.9. Privileged trajectory reward

Ngoai reward mot buoc, code con co co che danh gia them quang duong du doan cua robot neu bat privileged learning. Gia su planner sinh ra chuoi trang thai du doan:

```tex
\tau_t =
\{s_{t+1}, s_{t+2}, \dots, s_{t+K-1}\}
```

Reward phu theo trajectory duoc tinh:

```tex
r_{traj}
=
\sum_{i=1}^{K-1}
\gamma_{PL}^{i}
r_{eval}(s_{t+i})
```

Trong do `gamma_PL` la he so chiet khau rieng cho trajectory reward. Ham `r_eval` danh gia trang thai du doan dua tren huong robot, va cham, khoang cach an toan va RVO risk. Co che nay giup tac nhan nhan duoc tin hieu hoc tu chat luong ngan han cua trajectory do planner sinh ra, thay vi chi dua vao reward tai buoc ke tiep.

Code tuong ung:

- Bat `use_PL`: `HALO_1/drl_moudle/train_ppo.py:91-94`
- Tham so `PL_traj_length`, `PL_traj_gamma`: `HALO_1/drl_moudle/train_ppo.py:204-206`
- Cong trajectory reward: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1360-1367`
- Ham `eval_state`: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:943-1073`

### 27.10. Uoc luong advantage bang GAE

Sau khi thu thap rollout, PPO tinh advantage bang Generalized Advantage Estimation. Sai so TD tai moi buoc:

```tex
\delta_t =
r_t
+ \gamma V_\phi(s_{t+1})
- V_\phi(s_t)
```

Khi episode ket thuc, thanh phan bootstrap bi loai bo bang bien `next_non_terminal`:

```tex
\delta_t =
r_t
+ \gamma \cdot \text{nonterminal}_{t+1}
V_\phi(s_{t+1})
- V_\phi(s_t)
```

Advantage GAE:

```tex
\hat{A}_t =
\delta_t
+ \gamma\lambda \cdot \text{nonterminal}_{t+1}
\hat{A}_{t+1}
```

Target cho Critic:

```tex
\hat{R}_t =
\hat{A}_t + V_\phi(s_t)
```

Code tuong ung:

- GAE: `HALO_1/drl_moudle/modules/mpc_ppo_core.py:87-122`
- Goi tinh GAE sau rollout: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:481-485`

### 27.11. Ham mat mat PPO co action mask

Trong PPO, rollout duoc thu bang policy cu `pi_theta_old`, sau do policy moi `pi_theta` duoc cap nhat bang ty le xac suat:

```tex
r_t(\theta)
=
\frac{
\pi_\theta^m(a_t|s_t)
}{
\pi_{\theta_{old}}^m(a_t|s_t)
}
```

Trong code, ty le nay duoc tinh tu log-prob:

```tex
r_t(\theta)
=
\exp
\left(
\log\pi_\theta^m(a_t|s_t)
-
\log\pi_{\theta_{old}}^m(a_t|s_t)
\right)
```

Ham muc tieu clipped surrogate:

```tex
L_t^{CLIP}(\theta)
=
\min
\left(
r_t(\theta)\hat{A}_t,
\text{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)
\hat{A}_t
\right)
```

Do code dung gradient descent tren loss, policy loss duoc lay dau am:

```tex
L_{policy}
=
-\mathbb{E}_t[L_t^{CLIP}(\theta)]
```

Value loss:

```tex
L_{V}
=
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)-\hat{R}_t
\right)^2
\right]
```

Entropy cua policy sau mask:

```tex
S[\pi_\theta^m](s_t)
=
-
\sum_{a\in\mathcal{A}_{valid}(s_t)}
\pi_\theta^m(a|s_t)
\log \pi_\theta^m(a|s_t)
```

Tong loss duoc minimize:

```tex
L_{total}
=
L_{policy}
+ c_v L_V
+ c_e L_{entropy}
```

Trong do code dat:

```tex
L_{entropy} = -\mathbb{E}_t[S[\pi_\theta^m](s_t)]
```

nen khi cong `c_e L_entropy`, qua trinh minimize se khuyen khich entropy cao hon.

Code tuong ung:

- Ratio: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:528-529`
- Clip loss: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:531-534`
- Value loss: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:541-550`
- Entropy loss: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:553-560`
- Total loss: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:562`

### 27.12. Chu trinh huan luyen AI

Chu trinh huan luyen trong code co the tom tat nhu sau:

1. Moi truong tra ve observation gom robot state, human states, obstacle states, wall states va predicted human paths.
2. Observation duoc chuyen thanh graph `G_t`.
3. GAT trich xuat embedding tu graph.
4. Actor sinh phan phoi masked categorical tren 81 local goals.
5. Lay mau action `a_t`, map thanh local goal `g(a_t)`.
6. Planner nhan local goal va sinh gia toc banh `a_L, a_R`.
7. Environment thuc hien action, tinh reward va done.
8. Rollout buffer luu `s_t, a_t, r_t, V(s_t), log pi(a_t|s_t), mask_t`.
9. Sau `n_steps`, tinh GAE va return target.
10. PPO update Actor-Critic bang clipped objective, value loss va entropy regularization.

Code tuong ung:

- Khoi tao env/model: `HALO_1/drl_moudle/train_ppo.py:73-111`
- Collect rollout: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:391-488`
- Train PPO: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:490-605`
