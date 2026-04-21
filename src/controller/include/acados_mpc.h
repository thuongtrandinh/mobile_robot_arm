#ifndef ACADOS_MPC_H
#define ACADOS_MPC_H

#include <memory>
#include <vector>

#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_ddmr.h"
#include "types.h"

namespace robot_plann {

class AcadosMpc {
 public:
  explicit AcadosMpc(int verbose = 0);
  ~AcadosMpc();

  void SetParams(const MpcParams::Ptr &params);

  MpcReturn RunMpc(const JointState &state, std::vector<robot_plann::Point> &path);

 private:
  bool UpdateRuntimeWeightsAndConstraints();

  ddmr_solver_capsule *acados_capsule_;
  ocp_nlp_config *nlp_config_;
  ocp_nlp_dims *nlp_dims_;
  ocp_nlp_in *nlp_in_;
  ocp_nlp_out *nlp_out_;
  ocp_nlp_solver *nlp_solver_;
  void *nlp_opts_;

  MpcParams::Ptr params_;
  int verbose_;
  bool initialized_;
};

}  // namespace robot_plann

#endif  // ACADOS_MPC_H
