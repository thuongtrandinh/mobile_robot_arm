# Controller Async Cascade Pipeline Guide

Tai lieu nay mo ta kien truc duy nhat dang duoc duy tri trong package `controller`:

- Python RL bridge chi tao `sub_goal` va goi service `/ocp_plann`.
- C++ planner service chi cap nhat reference path (A* + smooth + profile).
- C++ MPC timer chay doc lap de giai MPC tu reference da cache.
- C++ cmd timer chay doc lap de publish `/diff_cont/cmd_vel` o tan so co dinh.

## 1) Data Flow

1. `rl_ocp_policy_bridge.py` (10 Hz)
   - Lay state/sensor
   - Suy luan policy RL
   - Gui `sub_goal` den `/ocp_plann`

2. `opt_planner` service callback (10 Hz theo nhan request)
   - Chi goi `UpdateReferenceOnly(...)`
   - Cap nhat cache reference path

3. `MpcExecTimerCallback` (20 Hz)
   - Doc state + reference cache
   - Goi `SolveMpcFromCachedReference(...)`
   - Cap nhat cmd cache

4. `CmdPublishTimerCallback` (20 Hz)
   - Publish cmd cache len `/diff_cont/cmd_vel`

## 2) Frequency Settings

Trong `config/mpc_tuning.yaml`:

- `rl_ocp_policy_bridge.policy_hz: 10.0`
- `rl_ocp_policy_bridge.service_hz: 10.0`
- `opt_planner.planner.mpc_exec_hz: 20.0`
- `opt_planner.planner.cmd_publish_hz: 20.0`

## 3) Notes

- Khong con mode `bridge_cmd`/`bridge_free`.
- Khong con `pipeline_mode`.
- Khong con Python publish cmd_vel.
- Khong con co `planner.async_pipeline_enabled`.

## 4) Quick Check

- `ros2 service list | grep /ocp_plann`
- `ros2 topic hz /diff_cont/cmd_vel`
- `ros2 topic hz /planner/debug_markers`
