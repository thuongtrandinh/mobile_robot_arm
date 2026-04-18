/**
  ******************************************************************************
  * @file    planner.h
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2023/05/05
  * @brief   ocp planner as rl backend for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#ifndef PLANNER_H
#define PLANNER_H
#include <chrono>
#include <iostream>

#include "types.h"
#include "lookahead.h"

#ifdef ROS_BUILD
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#endif

#include "mpc.h"


namespace robot_plann {

class Planner {
 public:
  explicit Planner(int verbose = 0)
    : Planner(CreateDefaultMpcParams(), verbose) {}

  explicit Planner(const MpcParams::Ptr &mpc_params, int verbose = 0)
    : visual_flag_(false),
      has_map_(false),
      move_forward_(true),
      verbose_(verbose) {
    mpc_params_ = std::make_shared<MpcParams>(*mpc_params);

    astar_planner_ = std::make_unique<AStar>(verbose);

    path_smoother_ = std::make_unique<SmoothCorner>();
    path_smoother_->SetDeltaMax(0.1);

    vel_planner_ = std::make_unique<LookAhead>();
    vel_planner_->SetParams(
        mpc_params_->max_linear_vel,
        mpc_params_->max_linear_acc,
        mpc_params_->max_angular_vel);
    ocp_planner_ = std::make_unique<Mpc>(verbose);

    ocp_planner_->SetParams(mpc_params_);
#ifdef ROS_BUILD
    a_start_smooth_path_ = std::make_shared<nav_msgs::msg::Path>();
#endif
    // update map
    cv::Mat map = this->CreateMap();
    this->SetMap(map);
    
  }

  void SetMap(const cv::Mat &map) {
    map_ = map;
    astar_planner_->InitMap(map_);
    has_map_ = true;
  }

  MpcReturn PlannExec(const JointState &state, Eigen::Vector2d &sub_goal);

  robot_plann::MPCOutputForPython RunSlover(const robot_plann::MPCInputForPython& input);


  bool visual_flag_;

#ifdef ROS_BUILD
  inline nav_msgs::msg::Path GetAStarSmoothPath() const {
    return *a_start_smooth_path_;
  }
#endif

  inline std::vector<Point> GetAStarPath() const {
    return astar_path_;
  }

  inline int GetMpcHorizonSteps() const {
    return static_cast<int>(mpc_params_->np);
  }

  inline double GetMpcDt() const {
    return mpc_params_->dt;
  }

  inline double GetWheelHalfTrack() const {
    return mpc_params_->wheel_half_track;
  }

  static Wall ClipWall(double x1, double y1, double x2, double y2, 
                      double x_min = -5, double x_max = 5);

  
 private:
  void UpdateCostMap(const JointState &state);
  cv::Mat CreateMap(); 
  bool CheckAround(Eigen::Vector2d &pos);
  bool CheckNavGoal(Eigen::Vector2d &sub_goal,
                    const Eigen::Vector2d &nav_goal, 
                    const Eigen::Vector2d &pos, 
                    const std::vector<Eigen::Vector2d> &vertices);
  
  bool SimpleRayCast(Eigen::Vector2d &goal, const Eigen::Vector2d &pos);
  cv::Point MapCoord2ImgIdx(const Eigen::Vector2d &pt, bool vis = false) const;
  Eigen::Vector2d ImgIdx2MapCoord(const cv::Point &idx, bool vis = false) const;
  Eigen::Vector2d PidCalc(const JointState &state);
  static void PybindInputDataChange(const robot_plann::MPCInputForPython& input,
                                    robot_plann::JointState &ob_state);

  // planner sub-modules
  std::unique_ptr<AStar> astar_planner_;

  std::unique_ptr<SmoothCorner> path_smoother_;
  std::unique_ptr<LookAhead> vel_planner_;

  std::unique_ptr<Mpc> ocp_planner_;

#ifdef ROS_BUILD
  std::shared_ptr<nav_msgs::msg::Path> a_start_smooth_path_;
#endif
  std::vector<Point> astar_path_{};

  cv::Mat map_, cost_map_, visual_map_;
  double window_center_x_ = 0.0;
  double window_center_y_ = 0.0;
  bool has_map_;
  bool move_forward_;
  int verbose_;
  MpcParams::Ptr mpc_params_;

  static MpcParams::Ptr CreateDefaultMpcParams() {
    auto params = std::make_shared<MpcParams>();
    params->dt = kDT;
    params->np = kNP;
    params->max_linear_vel = kMaxLinearVel;
    params->max_linear_acc = kMaxLinearAcc;
    params->max_angular_vel = kMaxAngularVel;
    params->max_angular_acc = kMaxAngularAcc;
    params->wheel_half_track = 0.3;
    params->local_obst_num = 8;
    return params;
  }

};

}  // namespace robot_plann

# endif  // PLANNER_H