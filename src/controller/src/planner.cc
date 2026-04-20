/**
  ******************************************************************************
  * @file    planner.cc
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2023/05/05
  * @brief   ocp planner as rl backend for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#include "planner.h"

namespace robot_plann {

cv::Point Planner::MapCoord2ImgIdx(const Eigen::Vector2d &pt, bool vis) const {
  if (cost_map_.empty()) {
    throw std::string("Cost map is empty");
  }

  const int width = cost_map_.cols;
  const int height = cost_map_.rows;
  cv::Point idx;
  idx.x = static_cast<int>(std::round((pt(0) - window_center_x_) / kMapResol)) + width / 2;
  idx.y = static_cast<int>(std::round((window_center_y_ - pt(1)) / kMapResol)) + height / 2;

  if (idx.x < 0 || idx.x >= width || idx.y < 0 || idx.y >= height) {
    std::stringstream err;
    err << "Point out of sliding window: " << pt(0) << ", " << pt(1)
        << " -> (" << idx.x << ", " << idx.y << ") in [0," << width
        << ")x[0," << height << ")";
    throw err.str();
  }

  if (vis) {
    idx.x = static_cast<int>(std::round(idx.x * kVisualScale));
    idx.y = static_cast<int>(std::round(idx.y * kVisualScale));
  }
  return idx;
}

Eigen::Vector2d Planner::ImgIdx2MapCoord(const cv::Point &idx, bool vis) const {
  if (cost_map_.empty()) {
    throw std::string("Cost map is empty");
  }

  cv::Point raw = idx;
  if (vis) {
    raw.x = static_cast<int>(std::round(raw.x / kVisualScale));
    raw.y = static_cast<int>(std::round(raw.y / kVisualScale));
  }

  const int width = cost_map_.cols;
  const int height = cost_map_.rows;
  if (raw.x < 0 || raw.x >= width || raw.y < 0 || raw.y >= height) {
    std::stringstream err;
    err << "Image index out of sliding window: (" << raw.x << ", " << raw.y
        << ") in [0," << width << ")x[0," << height << ")";
    throw err.str();
  }

  Eigen::Vector2d pt;
  pt(0) = (raw.x - width / 2) * kMapResol + window_center_x_;
  pt(1) = window_center_y_ - (raw.y - height / 2) * kMapResol;
  return pt;
}

#ifdef ROS_BUILD
nav_msgs::msg::OccupancyGrid Planner::GetLocalCostMap(
    const std::string &frame_id) const {
  nav_msgs::msg::OccupancyGrid msg;
  msg.header.frame_id = frame_id;

  if (cost_map_.empty()) {
    msg.info.resolution = static_cast<float>(kMapResol);
    msg.info.width = 0;
    msg.info.height = 0;
    msg.info.origin.orientation.w = 1.0;
    return msg;
  }

  const int width = cost_map_.cols;
  const int height = cost_map_.rows;

  msg.info.resolution = static_cast<float>(kMapResol);
  msg.info.width = static_cast<uint32_t>(width);
  msg.info.height = static_cast<uint32_t>(height);
  msg.info.origin.position.x = window_center_x_ - 0.5 * width * kMapResol;
  msg.info.origin.position.y = window_center_y_ - 0.5 * height * kMapResol;
  msg.info.origin.position.z = 0.0;
  msg.info.origin.orientation.x = 0.0;
  msg.info.origin.orientation.y = 0.0;
  msg.info.origin.orientation.z = 0.0;
  msg.info.origin.orientation.w = 1.0;

  msg.data.assign(static_cast<size_t>(width * height), 0);
  // Convert OpenCV image coordinates (y-down) to OccupancyGrid (y-up).
  for (int row = 0; row < height; ++row) {
    const int src_row = height - 1 - row;
    const int row_offset = row * width;
    for (int col = 0; col < width; ++col) {
      const int gray = static_cast<int>(cost_map_.at<u_char>(src_row, col));
      int occ = static_cast<int>(std::lround((255.0 - gray) * 100.0 / 255.0));
      occ = std::max(0, std::min(100, occ));
      msg.data[static_cast<size_t>(row_offset + col)] = static_cast<int8_t>(occ);
    }
  }

  return msg;
}
#endif

bool Planner::UpdateReferenceOnly(const JointState &state,
                                  Eigen::Vector2d &sub_goal) {
  if (!has_map_) return false;
  try {
    this->UpdateCostMap(state);
  } catch (const std::string &e) {
    if (verbose_ >= 1) {
      std::cout << "Cost map update failed! " << e << std::endl;
    }
    return false;
  }
  
  Eigen::Vector2d start_pt = {state.robot.px, state.robot.py};
  const Eigen::Vector2d goal_pt = {state.robot.gx, state.robot.gy};

  if (!CheckAround(start_pt)) {
    if (verbose_ >= 1) {
      std::cout << "Failed to escape from obstacle! " << std::endl;
    }
    return false;
  } 


  if (!CheckNavGoal(sub_goal, goal_pt, start_pt, state.rect.vertices)) {
    if (verbose_ >= 1) {
      std::cout << "Sub goal is unreachable!!!" << std::endl;
    }
    return false;
  }
  
  if (verbose_ > 1) std::cout << "DEBUG: check done!" << std::endl;

  auto astar_path = astar_planner_->SearchPath(
      cost_map_, start_pt, sub_goal, window_center_x_, window_center_y_);
  if (astar_path.size() < 2) {
    if (verbose_ >= 1) { 
      std::cout << "Invalid A* path with len = " << astar_path.size() << 
          std::endl;
    }
    return false;
  }

  auto smooth_path = path_smoother_->SmoothSharpCorner(cost_map_, astar_path);

  auto final_path = vel_planner_->UpdateVelocity(smooth_path, state.robot.v);

  {
#ifdef ROS_BUILD
    a_start_smooth_path_->header.frame_id = "map";
    a_start_smooth_path_->poses.clear();
    geometry_msgs::msg::PoseStamped pose_stamped;
    for (const Point &p: final_path) {
      pose_stamped.pose.position.x = p.x;
      pose_stamped.pose.position.y = p.y;
      pose_stamped.pose.position.z = p.v;
      a_start_smooth_path_->poses.push_back(pose_stamped);
    }
#endif
    astar_path_.clear();
    astar_path_ = final_path;
  }

  return true;
}

MpcReturn Planner::SolveMpcFromCachedReference(const JointState &state) {
  if (astar_path_.size() < 2) {
    if (verbose_ >= 1) {
      std::cout << "Invalid cached reference trajectory with len = "
                << astar_path_.size() << std::endl;
    }
    auto mpc_stages = MpcStages();
    Eigen::Vector2d pid_acc = this->PidCalc(state);
    mpc_stages[0].uk.acc = pid_acc(0);
    mpc_stages[0].uk.dr = pid_acc(1);
    return {mpc_stages, true};
  }

  auto final_path = astar_path_;
  auto revised_state = state;

  double phi_0 = atan2(final_path[1].y - final_path[0].y,
                       final_path[1].x - final_path[0].x);

  double yaw_error = state.robot.yaw - phi_0;
  Unwrap(yaw_error);

  if (abs(yaw_error) > M_PI / 2) {
    revised_state.robot.yaw -= M_PI;
    Unwrap(revised_state.robot.yaw);
    revised_state.robot.v = -revised_state.robot.v;
    move_forward_ = false;
  } else {
    move_forward_ = true;
  }

  while (final_path.size() < static_cast<size_t>(GetMpcHorizonSteps())) {
    final_path.push_back(final_path.back());
  }
  
  if (verbose_ > 1) std::cout << "DEBUG: geo plann done!" << std::endl;

  auto mpc_return = ocp_planner_->RunMpc(revised_state, final_path);

  if (!mpc_return.success) {
    // Keep robot moving toward goal when NLP is temporarily infeasible.
    auto fallback_stages = MpcStages();
    Eigen::Vector2d pid_acc = this->PidCalc(revised_state);
    fallback_stages[0].uk.acc = pid_acc(0);
    fallback_stages[0].uk.dr = pid_acc(1);
    if (!move_forward_) {
      fallback_stages[0].uk.acc = -fallback_stages[0].uk.acc;
    }
    return {fallback_stages, true};
  }

  // std::cout << "===" << std::endl;
  if (visual_flag_) {
    for (int i = 0; i < final_path.size(); i++) {
      Eigen::Vector2d path_pt(final_path[i].x, final_path[i].y);
      cv::Point visual_pt = MapCoord2ImgIdx(path_pt, true);
      cv::circle(visual_map_, visual_pt, 3, cv::Scalar(200, 0, 0), 2);
    }
    if (mpc_return.success) {
      for (int k = 0; k < mpc_return.stages.size(); k++) {
        Eigen::Vector2d path_pt(mpc_return.stages[k].xk.X, 
                                mpc_return.stages[k].xk.Y);
        
        cv::Point visual_pt = MapCoord2ImgIdx(path_pt, true);
        cv::circle(visual_map_, visual_pt, 3, cv::Scalar(0, 0, 200), 2);
      }
    }

    cv::imshow("debug!", visual_map_);
    cv::waitKey(5);
  }
  if (mpc_return.success && !move_forward_) {
    for (std::size_t t = 0; t < mpc_return.stages.size(); ++t) {
      mpc_return.stages[t].uk.acc = -mpc_return.stages[t].uk.acc;
    }
    // mpc_return.stages[0].uk.acc = -mpc_return.stages[0].uk.acc;
  }
  return mpc_return;
}

bool Planner::CheckAround(Eigen::Vector2d &pos) {
  cv::Point pixel;
  try{
    pixel = MapCoord2ImgIdx(pos);
  } catch (const std::string &e) {
    std::cout << e << std::endl;
    return false;
  }
  if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) return true;
  
  auto is_free = [this](int x, int y) {
    return x > 0 && x < cost_map_.cols && y > 0 && y < cost_map_.rows &&
           cost_map_.at<u_char>(y, x) > 240;
  };

  int x_right_free = 0, x_left_free = 0, y_up_free = 0, y_down_free = 0;
  for (int i = 0; i < 15; i++) {
    if (is_free(pixel.x + i, pixel.y)) x_right_free++;
    if (is_free(pixel.x, pixel.y - i)) y_up_free++;
    if (is_free(pixel.x - i, pixel.y)) x_left_free++;
    if (is_free(pixel.x, pixel.y + i)) y_down_free++;
  }

  if (x_right_free > std::max({y_up_free, x_left_free, y_down_free})) {
    // should escape from x right which has the least occupancy pixel counts
    while (++pixel.x < cost_map_.cols) {
      if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) {
        pos = ImgIdx2MapCoord(pixel);
        if (verbose_ >= 3) std::cout << "Escape from x right!" << std::endl;
        return true;
      }
    }
  } else if (y_up_free > std::max({x_left_free, y_down_free})) {
    while (--pixel.y > 0) {
      if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) {
        pos = ImgIdx2MapCoord(pixel);
        if (verbose_ >= 3) std::cout << "Escape from y up!" << std::endl;
        return true;
      }      
    }
  } else if (x_left_free > y_down_free) {
    while (--pixel.x > 0) {
      if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) {
        pos = ImgIdx2MapCoord(pixel);
        if (verbose_ >= 3) std::cout << "Escape from x left!" << std::endl;
        return true;
      }      
    }
  } else {
    while (++pixel.y < cost_map_.rows) {
      if (255 - cost_map_.at<u_char>(pixel.y, pixel.x) <= 250) {
        pos = ImgIdx2MapCoord(pixel);
        if (verbose_ >= 3) std::cout << "Escape from y down!" << std::endl;
        return true;
      }      
    }
  }
  return false;  // failed to escape
}

bool Planner::CheckNavGoal(Eigen::Vector2d &sub_goal,
                           const Eigen::Vector2d &nav_goal,
                           const Eigen::Vector2d &pos,
                           const std::vector<Eigen::Vector2d> &vertices) {
  (void)nav_goal;
  (void)vertices;

  // 1. Kiểm tra khoảng cách: Nếu điểm RL cấp quá gần, từ chối để RL lấy điểm mới.
  // Tuyệt đối không tự ý đẩy điểm hướng về nav_goal (Global Goal) nữa.
  float distance = (sub_goal - pos).norm();
  if (distance < 0.05) {
    return false;
  }

  // 2. Tôn trọng tuyệt đối điểm của RL.
  // Chỉ cần kiểm tra xem có đường ngắm thẳng (Line of sight) từ Robot tới điểm đó không.
  try {
    (void)MapCoord2ImgIdx(sub_goal);
  } catch (const std::string &) {
    if (!this->SimpleRayCast(sub_goal, pos)) {
      if (verbose_ >= 1) {
        std::cout << "RL point blocked by wall, Raycast failed. Requesting new point." << std::endl;
      }
      return false;  // Điểm RL bị khuất tường hoàn toàn -> Trả về false để từ chối
    }
  }

  // 3. Đã xóa toàn bộ logic Polygon Fix và Circular Fix.
  // Điểm RL đã an toàn, cho phép giải MPC!
  return true;
}

void Planner::UpdateCostMap(const JointState &state) {
  // TODO:
  window_center_x_ = state.robot.px;
  window_center_y_ = state.robot.py;
  cost_map_ = map_.clone();
  std::size_t skipped_invalid_points = 0;

  // Clip walls against the local sliding window to keep both endpoints valid.
  const double half_w = (cost_map_.cols / 2.0) * kMapResol;
  const double half_h = (cost_map_.rows / 2.0) * kMapResol;
  const double x_min = window_center_x_ - half_w + 0.02;
  const double x_max = window_center_x_ + half_w - 0.02;
  const double y_min = window_center_y_ - half_h + 0.02;
  const double y_max = window_center_y_ + half_h - 0.02;

  for (const auto &wall : state.walls) {
    double x1 = wall.first.x;
    double y1 = wall.first.y;
    double x2 = wall.second.x;
    double y2 = wall.second.y;

    if (ClipLine(x1, y1, x2, y2, x_min, x_max, y_min, y_max)) {
      try {
        cv::Point p1 = MapCoord2ImgIdx({x1, y1});
        cv::Point p2 = MapCoord2ImgIdx({x2, y2});
        cv::line(cost_map_, p1, p2, cv::Scalar(0), 1);
      } catch (const std::string &) {
        skipped_invalid_points++;
      }
    }
  }

  std::vector<cv::Point> contour;
  contour.reserve(state.rect.vertices.size());
  for (const auto &vertex : state.rect.vertices) {
    try {
      contour.push_back(MapCoord2ImgIdx(vertex));
    } catch (const std::string &) {
      skipped_invalid_points++;
    }
  }
  std::vector<std::vector<cv::Point>> contours = {contour};
  if (!contour.empty()) {
    cv::drawContours(cost_map_, contours, -1, cv::Scalar(0), 1);
  }

  for (auto &iter: state.obst) {
    try {
      cv::circle(cost_map_, MapCoord2ImgIdx({iter.px, iter.py}), 
                 (int)round(iter.radius / kMapResol), cv::Scalar(0), 1);
    } catch (const std::string &) {
      skipped_invalid_points++;
    }
  }

  if (verbose_ >= 2 && skipped_invalid_points > 0) {
    std::cout << "Skip " << skipped_invalid_points
              << " invalid map points in UpdateCostMap" << std::endl;
  }

  visual_map_ = cost_map_.clone();
  static cv::Size vis_size(kVisualScale * cost_map_.cols, 
                           kVisualScale * cost_map_.rows);
  
  cv::resize(visual_map_, visual_map_, vis_size);
  cv::cvtColor(visual_map_, visual_map_, cv::COLOR_GRAY2BGR);
  // map inflation
  static int inflation_pixel = (int)round(kInflationRadius / kMapResol) + 2;  // + noise fix 2 pixel
  
  cv::Mat kernal = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(inflation_pixel, inflation_pixel));
  cv::erode(cost_map_, cost_map_, kernal);

  // cv::imshow("debug!", cost_map_);
  // cv::waitKey(5);
}

bool Planner::SimpleRayCast(Eigen::Vector2d &goal, 
                            const Eigen::Vector2d &pos) {
  Eigen::Vector2d ray = goal - pos;
  double distance = ray.norm();
  goal = pos;

  if (distance < 1e-6) {
    return false;
  }

  for (double step = 0.0; step < distance; step += 0.1) {
    double t = step / distance;
    Eigen::Vector2d candidate = pos + t * ray;

    try {
      (void)MapCoord2ImgIdx(candidate);
    } catch (const std::string &) {
      break;
    }
    goal = candidate;
  }

  return ((goal - pos).norm() < 0.1)? false: true;
}

Eigen::Vector2d Planner::PidCalc(const JointState &state) {
  Eigen::Vector3d robot_pose;
  robot_pose << state.robot.px, state.robot.py, state.robot.yaw;
  
  Eigen::Vector2d desired_pos;
  desired_pos << state.robot.gx, state.robot.gy;

  double alpha = 
      atan2(desired_pos(1) - robot_pose(1), desired_pos(0) - robot_pose(0)) - 
      robot_pose(2);
  
  Unwrap(alpha);

  double forward = (alpha <= M_PI / 2 && alpha > -M_PI / 2)? 1.0: -1.0;
  double dist_error = forward * 
      EuclideanNorm(desired_pos(0) - robot_pose(0), 
                    desired_pos(1) - robot_pose(1));

  if (abs(dist_error) < 0.05) alpha = 0;

  Eigen::Vector2d acc_ctrl;
  Eigen::Vector2d desired_vel;
  desired_vel << 0.8 * dist_error, 2.0 * alpha;

  const double dt = GetMpcDt();
  acc_ctrl(0) = (desired_vel(0) - state.robot.v) / (dt * 0.5);
  acc_ctrl(1) = (desired_vel(1) - state.robot.yaw_rate) / (dt * 0.5);

  return acc_ctrl;
}

cv::Mat Planner::CreateMap() {
  cv::Mat map(400, 240, CV_8UC1, cv::Scalar(255));  // TODO: change to 10 x 20

  std::cout << "create map" << std::endl;
  return std::move(map);
};

robot_plann::MPCOutputForPython Planner::RunSlover(
  const robot_plann::MPCInputForPython& input) {
  
  robot_plann::MPCOutputForPython ans{};
  ans.astar_path.clear();
  ans.control_vars.clear();
  ans.success = false;
  if (input.valid == false) {
    return ans;
  }
  robot_plann::JointState ob_state;

  PybindInputDataChange(input, ob_state);
  Eigen::Vector2d sub_goal = {input.sub_goal.x, input.sub_goal.y};

    auto start_stamp = std::chrono::high_resolution_clock::now();
    const bool ref_ok = this->UpdateReferenceOnly(ob_state, sub_goal);
    auto mpc_return = ref_ok ? this->SolveMpcFromCachedReference(ob_state)
                 : MpcReturn{MpcStages(), false};
    auto end_stamp = std::chrono::high_resolution_clock::now();
    double time_cost =
      std::chrono::duration<double, std::milli>(end_stamp - start_stamp)
        .count();

  std::vector<robot_plann::Point> astar_path = this->GetAStarPath();
  for (int i = 0; i < astar_path.size(); ++i) {
    robot_plann::Point pt;
    pt.x = astar_path.at(i).x;
    pt.y = astar_path.at(i).y;
    pt.v = 0.0;
    ans.astar_path.push_back(pt);

  }

  robot_plann::ControlVar cur_control_var{};
  if (!mpc_return.success) {
    ans.al = 0.0;
    ans.ar = 0.0;
    ans.revised_goal.x = input.sub_goal.x;
    ans.revised_goal.y = input.sub_goal.y;
    std::cout << "Ocp plann failed!" << std::endl;
    cur_control_var.al = ans.al;
    cur_control_var.ar = ans.ar;
    for (int i = 0; i < GetMpcHorizonSteps(); ++i) {
      ans.control_vars.push_back(cur_control_var);
    }

  } else {
    const double half_track = GetWheelHalfTrack();
    ans.al = mpc_return.stages[0].uk.acc - mpc_return.stages[0].uk.dr * half_track;
    ans.ar = mpc_return.stages[0].uk.acc + mpc_return.stages[0].uk.dr * half_track;
    ans.revised_goal.x = input.sub_goal.x;
    ans.revised_goal.y = input.sub_goal.y;
   
    for (int i = 0; i < GetMpcHorizonSteps(); ++i) {
      cur_control_var.al =
          mpc_return.stages.at(i).uk.acc - mpc_return.stages.at(i).uk.dr * half_track;
      cur_control_var.ar =
          mpc_return.stages.at(i).uk.acc + mpc_return.stages.at(i).uk.dr * half_track;

      ans.control_vars.push_back(cur_control_var);
    }
    ans.success = true;
    std::cout << "Time cost: " << time_cost << std::endl;

  }

  return std::move(ans);
}


void Planner::PybindInputDataChange(const robot_plann::MPCInputForPython& input,
                               robot_plann::JointState &ob_state) {
  
  const auto &robot_state = input.ob.robot;
  ob_state.robot.px = robot_state.px;
  ob_state.robot.py = robot_state.py;
  ob_state.robot.yaw = robot_state.yaw;
  ob_state.robot.v = robot_state.v;
  ob_state.robot.yaw_rate = robot_state.yaw_rate;

  ob_state.robot.v_pref = robot_state.v_pref;
  ob_state.robot.radius = robot_state.radius;
  ob_state.robot.gx = robot_state.gx;
  ob_state.robot.gy = robot_state.gy;

  for (const auto &hum_iter : input.ob.hum) {
    robot_plann::HumanState hum_state;
    hum_state.px = hum_iter.px;
    hum_state.py = hum_iter.py;
    hum_state.vx = hum_iter.vx;
    hum_state.vy = hum_iter.vy;
    hum_state.radius = hum_iter.radius;

    ob_state.hum.push_back(hum_state);
  }

  for (const auto &obst_iter : input.ob.obst) {
    robot_plann::ObstacleState obst_state;
    obst_state.px = obst_iter.px;
    obst_state.py = obst_iter.py;
    obst_state.radius = obst_iter.radius;
    ob_state.obst.push_back(obst_state);
  }

  ob_state.rect.vertices.clear();
  // for (const auto &poly_iter : input.ob.rect) {
  for (const auto &vertex : input.ob.rect.vertices) {
    Eigen::Vector2d pt(vertex.x, vertex.y);
    ob_state.rect.vertices.push_back(pt);
  }

  ob_state.walls.clear();
  for (const auto &input_wall : input.ob.walls) {
    Wall wall;
    wall.first.x = input_wall.sx;
    wall.first.y = input_wall.sy;
    wall.second.x = input_wall.ex;
    wall.second.y = input_wall.ey;
    ob_state.walls.push_back(wall);
  }

}

// Cohen-Sutherland line clipping against an axis-aligned box.
bool Planner::ClipLine(double &x1, double &y1, double &x2, double &y2,
                       double x_min, double x_max, double y_min,
                       double y_max) {
  auto compute_outcode = [&](double x, double y) {
    int code = 0;
    if (x < x_min) code |= 1;       // LEFT
    else if (x > x_max) code |= 2;  // RIGHT
    if (y < y_min) code |= 4;       // BOTTOM
    else if (y > y_max) code |= 8;  // TOP
    return code;
  };

  int outcode1 = compute_outcode(x1, y1);
  int outcode2 = compute_outcode(x2, y2);

  while (true) {
    if (!(outcode1 | outcode2)) {
      return true;
    }
    if (outcode1 & outcode2) {
      return false;
    }

    const int outcode_out = outcode1 ? outcode1 : outcode2;
    double x = 0.0;
    double y = 0.0;

    if (outcode_out & 8) {
      const double dy = y2 - y1;
      if (std::abs(dy) < 1e-12) return false;
      x = x1 + (x2 - x1) * (y_max - y1) / dy;
      y = y_max;
    } else if (outcode_out & 4) {
      const double dy = y2 - y1;
      if (std::abs(dy) < 1e-12) return false;
      x = x1 + (x2 - x1) * (y_min - y1) / dy;
      y = y_min;
    } else if (outcode_out & 2) {
      const double dx = x2 - x1;
      if (std::abs(dx) < 1e-12) return false;
      y = y1 + (y2 - y1) * (x_max - x1) / dx;
      x = x_max;
    } else {
      const double dx = x2 - x1;
      if (std::abs(dx) < 1e-12) return false;
      y = y1 + (y2 - y1) * (x_min - x1) / dx;
      x = x_min;
    }

    if (outcode_out == outcode1) {
      x1 = x;
      y1 = y;
      outcode1 = compute_outcode(x1, y1);
    } else {
      x2 = x;
      y2 = y;
      outcode2 = compute_outcode(x2, y2);
    }
  }
}

}  // namespace robot_plann