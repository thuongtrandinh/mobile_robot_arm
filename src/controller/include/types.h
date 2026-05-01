/**
  ******************************************************************************
  * @file    types.h
  * @author  Alex Liu 
  * @version V1.0.0
  * @date    2023/05/05
  * @brief   ocp planner as rl backend for MobiRo @ tib_k331
  ******************************************************************************
  * @attention
  *
  ******************************************************************************
  */
#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <memory>


namespace robot_plann {

static constexpr float kMapResol = 0.05;
static constexpr float kHalfMapHeight = 10.0;
static constexpr float kHalfMapWidth = 6.0;

static constexpr float kVisualScale = 3.0;
static constexpr float kInflationRadius = 0.3;  // typically equivalent to the robot's radius
static constexpr int kNP = 20;
static constexpr float kDT = 0.1;

static constexpr double kMaxLinearVel  = 1.0;
static constexpr double kMaxLinearAcc  = 1.0;
static constexpr double kMaxAngularVel = 1.0;
static constexpr double kMaxAngularAcc = 1.0;

// st search
static constexpr double kSTHalfLocalRange = kDT * (kNP - 1) * 1.0;
static constexpr double kSTHalfLocalRangeSquare = 
                        kSTHalfLocalRange * kSTHalfLocalRange;

static constexpr std::size_t kThreadNum = 8;

struct Point {
  double x;
  double y;
  double v;
};

struct RobotState {
  double px;
  double py;
  double yaw;
  double v;
  double yaw_rate;

  double radius;
  double gx;
  double gy;
  double v_pref = 1.0;
};

typedef std::vector<Point> Trajectory;
typedef std::pair<Point, Point> Wall;

struct ForPythonWall {
  double sx;
  double sy;
  double ex;
  double ey;
};


struct HumanState {
  double px;
  double py;
  double vx;
  double vy;
  double radius;
  // Temp: Consider point.v as a predictive probability
  std::vector<Trajectory> pred_trajectorys; 
  // std::vector<double> prob_pred_trajectorys;

};

struct ObstacleState {
  double px;
  double py;
  double radius;
};

struct PolygonState {
  std::vector<Eigen::Vector2d> vertices;
};

struct PolygonStateForPython {
  std::vector<Point> vertices;
};

struct JointState {
  RobotState robot;
  std::vector<HumanState> hum;
  std::vector<ObstacleState> obst;
  PolygonState rect;
  std::vector<Wall> walls;
};

struct JointStateForPython {
  RobotState robot;
  std::vector<HumanState> hum;
  std::vector<ObstacleState> obst;
  PolygonStateForPython rect;
  std::vector<ForPythonWall> walls;
};

struct ReferencePlannerParams {
  uint16_t horizon_steps;
  double dt;
  double max_linear_vel;
  double max_angular_vel;
  double max_linear_acc;
  double max_angular_acc;
  int local_obst_num;
  double polygon_clearance_margin = 0.03;
  double wall_constraint_activation_distance = 2.0;

  typedef std::shared_ptr<ReferencePlannerParams> Ptr;
};

struct Position {
  Point position;
};

struct Pose {
  Position pose;
};
struct Path {
  std::vector<Pose> poses;
};

struct HyperPlane {
  double nx;
  double ny;
  double b;
};

struct DynaObstacle {
  double id;
  double distance;
  double radius;
  std::vector<Trajectory> pred_trajectorys;
};

struct ObstacleArray {
  std::vector<DynaObstacle> dyna_obstacles{};
};

}  // namespace robot_plann

#endif  // TYPES_H
