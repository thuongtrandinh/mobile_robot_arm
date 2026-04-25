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
#include <string>

#include "types.h"
#include "lookahead.h"

#ifdef ROS_BUILD
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#endif

namespace robot_plann {

class Planner {
 public:
  explicit Planner(int verbose = 0)
    : Planner(CreateDefaultParams(), verbose) {}

  explicit Planner(const ReferencePlannerParams::Ptr &params, int verbose = 0)
    : visual_flag_(false),
      has_map_(false),
      move_forward_(true),
      verbose_(verbose) {
    params_ = std::make_shared<ReferencePlannerParams>(*params);

    astar_planner_ = std::make_unique<AStar>(verbose);

    path_smoother_ = std::make_unique<SmoothCorner>();
    path_smoother_->SetDeltaMax(0.1);

    vel_planner_ = std::make_unique<LookAhead>();
    vel_planner_->SetParams(
        params_->max_linear_vel,
        params_->max_linear_acc,
        params_->max_angular_vel);
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

  bool UpdateReferenceOnly(const JointState &state, Eigen::Vector2d &sub_goal);

  bool visual_flag_;

#ifdef ROS_BUILD
  inline nav_msgs::msg::Path GetAStarSmoothPath() const {
    return *a_start_smooth_path_;
  }

  nav_msgs::msg::OccupancyGrid GetLocalCostMap(
      const std::string &frame_id = "map") const;
#endif

  inline std::vector<Point> GetAStarPath() const {
    return astar_path_;
  }

  inline int GetReferenceHorizonSteps() const {
    return static_cast<int>(params_->horizon_steps);
  }

  inline double GetReferenceDt() const {
    return params_->dt;
  }

  void SetReferenceMode(const std::string &mode) {
    use_direct_goal_xref_ = (mode == "direct_goal_xref");
  }

  void SetDebugDirectXrefParams(double v_max, double v_min,
                                double slowdown_distance, double kp_dist) {
    debug_xref_v_max_ = v_max;
    debug_xref_v_min_ = v_min;
    debug_xref_slowdown_distance_ = slowdown_distance;
    debug_xref_kp_dist_ = kp_dist;
  }

  void UpdateParams(const ReferencePlannerParams::Ptr &params) {
    if (!params) {
      return;
    }
    params_ = std::make_shared<ReferencePlannerParams>(*params);
    vel_planner_->SetParams(
        params_->max_linear_vel,
        params_->max_linear_acc,
        params_->max_angular_vel);
  }

  static bool ClipLine(double &x1, double &y1, double &x2, double &y2,
                       double x_min, double x_max, double y_min,
                       double y_max);

  
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
  std::vector<Point> BuildDirectReferenceTrajectory(
      const JointState &state, const Eigen::Vector2d &goal) const;
  double ComputeDirectReferenceSpeed(double remaining_distance) const;
  // planner sub-modules
  std::unique_ptr<AStar> astar_planner_;

  std::unique_ptr<SmoothCorner> path_smoother_;
  std::unique_ptr<LookAhead> vel_planner_;

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
  ReferencePlannerParams::Ptr params_;
  bool use_direct_goal_xref_ = false;
  double debug_xref_v_max_ = 0.6;
  double debug_xref_v_min_ = 0.05;
  double debug_xref_slowdown_distance_ = 1.5;
  double debug_xref_kp_dist_ = 0.8;

  static ReferencePlannerParams::Ptr CreateDefaultParams() {
    auto params = std::make_shared<ReferencePlannerParams>();
    params->dt = kDT;
    params->horizon_steps = kNP;
    params->max_linear_vel = kMaxLinearVel;
    params->max_linear_acc = kMaxLinearAcc;
    params->max_angular_vel = kMaxAngularVel;
    params->max_angular_acc = kMaxAngularAcc;
    params->local_obst_num = 8;
    return params;
  }

};

}  // namespace robot_plann

# endif  // PLANNER_H
