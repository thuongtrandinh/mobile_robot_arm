/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_ddmr_H_
#define ACADOS_SOLVER_ddmr_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define DDMR_NX     7
#define DDMR_NZ     0
#define DDMR_NU     22
#define DDMR_NP     203
#define DDMR_NP_GLOBAL     0
#define DDMR_NBX    1
#define DDMR_NBX0   7
#define DDMR_NBU    22
#define DDMR_NSBX   0
#define DDMR_NSBU   0
#define DDMR_NSH    65
#define DDMR_NSH0   0
#define DDMR_NSG    0
#define DDMR_NSPHI  0
#define DDMR_NSHN   0
#define DDMR_NSGN   0
#define DDMR_NSPHIN 0
#define DDMR_NSPHI0 0
#define DDMR_NSBXN  0
#define DDMR_NS     65
#define DDMR_NS0    0
#define DDMR_NSN    0
#define DDMR_NG     0
#define DDMR_NBXN   0
#define DDMR_NGN    0
#define DDMR_NY0    29
#define DDMR_NY     29
#define DDMR_NYN    5
#define DDMR_N      20
#define DDMR_NH     72
#define DDMR_NHN    0
#define DDMR_NH0    0
#define DDMR_NPHI0  0
#define DDMR_NPHI   0
#define DDMR_NPHIN  0
#define DDMR_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct ddmr_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *expl_vde_forw;
    external_function_external_param_casadi *expl_vde_forw_p;
    external_function_external_param_casadi *expl_ode_fun;
    external_function_external_param_casadi *expl_vde_adj;




    // cost






    // constraints
    external_function_external_param_casadi *nl_constr_h_fun_jac;
    external_function_external_param_casadi *nl_constr_h_fun;









} ddmr_solver_capsule;

ACADOS_SYMBOL_EXPORT ddmr_solver_capsule * ddmr_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int ddmr_acados_free_capsule(ddmr_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int ddmr_acados_create(ddmr_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int ddmr_acados_reset(ddmr_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of ddmr_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int ddmr_acados_create_with_discretization(ddmr_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int ddmr_acados_update_time_steps(ddmr_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int ddmr_acados_update_qp_solver_cond_N(ddmr_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int ddmr_acados_update_params(ddmr_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int ddmr_acados_update_params_sparse(ddmr_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int ddmr_acados_set_p_global_and_precompute_dependencies(ddmr_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int ddmr_acados_solve(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int ddmr_acados_setup_qp_matrices_and_factorize(ddmr_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int ddmr_acados_free(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void ddmr_acados_print_stats(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int ddmr_acados_custom_update(ddmr_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT ocp_nlp_in *ddmr_acados_get_nlp_in(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *ddmr_acados_get_nlp_out(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *ddmr_acados_get_sens_out(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *ddmr_acados_get_nlp_solver(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *ddmr_acados_get_nlp_config(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *ddmr_acados_get_nlp_opts(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *ddmr_acados_get_nlp_dims(ddmr_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *ddmr_acados_get_nlp_plan(ddmr_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_ddmr_H_
