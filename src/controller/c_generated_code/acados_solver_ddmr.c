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

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "ddmr_model/ddmr_model.h"


#include "ddmr_constraints/ddmr_constraints.h"



#include "acados_solver_ddmr.h"

#define NX     DDMR_NX
#define NZ     DDMR_NZ
#define NU     DDMR_NU
#define NP     DDMR_NP
#define NP_GLOBAL     DDMR_NP_GLOBAL
#define NY0    DDMR_NY0
#define NY     DDMR_NY
#define NYN    DDMR_NYN

#define NBX    DDMR_NBX
#define NBX0   DDMR_NBX0
#define NBU    DDMR_NBU
#define NG     DDMR_NG
#define NBXN   DDMR_NBXN
#define NGN    DDMR_NGN

#define NH     DDMR_NH
#define NHN    DDMR_NHN
#define NH0    DDMR_NH0
#define NPHI   DDMR_NPHI
#define NPHIN  DDMR_NPHIN
#define NPHI0  DDMR_NPHI0
#define NR     DDMR_NR

#define NS     DDMR_NS
#define NS0    DDMR_NS0
#define NSN    DDMR_NSN

#define NSBX   DDMR_NSBX
#define NSBU   DDMR_NSBU
#define NSH0   DDMR_NSH0
#define NSH    DDMR_NSH
#define NSHN   DDMR_NSHN
#define NSG    DDMR_NSG
#define NSPHI0 DDMR_NSPHI0
#define NSPHI  DDMR_NSPHI
#define NSPHIN DDMR_NSPHIN
#define NSGN   DDMR_NSGN
#define NSBXN  DDMR_NSBXN



// ** solver data **

ddmr_solver_capsule * ddmr_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(ddmr_solver_capsule));
    ddmr_solver_capsule *capsule = (ddmr_solver_capsule *) capsule_mem;

    return capsule;
}


int ddmr_acados_free_capsule(ddmr_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int ddmr_acados_create(ddmr_solver_capsule* capsule)
{
    int N_shooting_intervals = DDMR_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return ddmr_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int ddmr_acados_update_time_steps(ddmr_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "ddmr_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for ddmr_acados_create: step 1
 */
void ddmr_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = LINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = ERK;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* ddmr_acados_create_setup_dimensions(ddmr_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 7;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for ddmr_acados_create: step 3
 */
void ddmr_acados_create_setup_functions(ddmr_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {
        // constraints.constr_type == "BGH" and dims.nh > 0
        capsule->nl_constr_h_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun_jac[i], ddmr_constr_h_fun_jac_uxt_zt);
        }
        capsule->nl_constr_h_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun[i], ddmr_constr_h_fun);
        }
    



    
        // explicit ode
        capsule->expl_vde_forw = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_vde_forw[i], ddmr_expl_vde_forw);
        }

        

        capsule->expl_ode_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_ode_fun[i], ddmr_expl_ode_fun);
        }

        capsule->expl_vde_adj = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_vde_adj[i], ddmr_expl_vde_adj);
        }

    
    } // N > 0

#undef MAP_CASADI_FNC
}


/**
 * Internal function for ddmr_acados_create: step 5
 */
void ddmr_acados_create_set_default_parameters(ddmr_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        ddmr_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for ddmr_acados_create: step 5
 */
void ddmr_acados_setup_nlp_in(ddmr_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        ddmr_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double time_step = 0.1;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.1;
        cost_scaling[1] = 0.1;
        cost_scaling[2] = 0.1;
        cost_scaling[3] = 0.1;
        cost_scaling[4] = 0.1;
        cost_scaling[5] = 0.1;
        cost_scaling[6] = 0.1;
        cost_scaling[7] = 0.1;
        cost_scaling[8] = 0.1;
        cost_scaling[9] = 0.1;
        cost_scaling[10] = 0.1;
        cost_scaling[11] = 0.1;
        cost_scaling[12] = 0.1;
        cost_scaling[13] = 0.1;
        cost_scaling[14] = 0.1;
        cost_scaling[15] = 0.1;
        cost_scaling[16] = 0.1;
        cost_scaling[17] = 0.1;
        cost_scaling[18] = 0.1;
        cost_scaling[19] = 0.1;
        cost_scaling[20] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_forw", &capsule->expl_vde_forw[i]);
        
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_ode_fun", &capsule->expl_ode_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_adj", &capsule->expl_vde_adj[i]);
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 5;
    W_0[1+(NY0) * 1] = 5;
    W_0[2+(NY0) * 2] = 1;
    W_0[3+(NY0) * 3] = 1;
    W_0[4+(NY0) * 4] = 1;
    W_0[5+(NY0) * 5] = 2;
    W_0[6+(NY0) * 6] = 0.5;
    W_0[7+(NY0) * 7] = 0.0005000000000000001;
    W_0[8+(NY0) * 8] = 0.0005000000000000001;
    W_0[9+(NY0) * 9] = 0.0001;
    W_0[10+(NY0) * 10] = 0.0001;
    W_0[11+(NY0) * 11] = 0.0001;
    W_0[12+(NY0) * 12] = 0.0001;
    W_0[13+(NY0) * 13] = 0.0001;
    W_0[14+(NY0) * 14] = 0.0001;
    W_0[15+(NY0) * 15] = 0.0001;
    W_0[16+(NY0) * 16] = 0.0001;
    W_0[17+(NY0) * 17] = 0.0001;
    W_0[18+(NY0) * 18] = 0.0001;
    W_0[19+(NY0) * 19] = 0.0001;
    W_0[20+(NY0) * 20] = 0.0001;
    W_0[21+(NY0) * 21] = 0.0001;
    W_0[22+(NY0) * 22] = 0.0001;
    W_0[23+(NY0) * 23] = 0.0001;
    W_0[24+(NY0) * 24] = 0.0001;
    W_0[25+(NY0) * 25] = 0.0001;
    W_0[26+(NY0) * 26] = 0.0001;
    W_0[27+(NY0) * 27] = 0.0001;
    W_0[28+(NY0) * 28] = 0.0001;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* Vx_0 = calloc(NY0*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_0[0+(NY0) * 0] = 1;
    Vx_0[1+(NY0) * 1] = 1;
    Vx_0[2+(NY0) * 2] = 1;
    Vx_0[3+(NY0) * 3] = 1;
    Vx_0[4+(NY0) * 4] = 1;
    Vx_0[5+(NY0) * 5] = 1;
    Vx_0[6+(NY0) * 6] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);
    double* Vu_0 = calloc(NY0*NU, sizeof(double));
    // change only the non-zero elements:
    Vu_0[7+(NY0) * 0] = 1;
    Vu_0[8+(NY0) * 1] = 1;
    Vu_0[9+(NY0) * 2] = 1;
    Vu_0[10+(NY0) * 3] = 1;
    Vu_0[11+(NY0) * 4] = 1;
    Vu_0[12+(NY0) * 5] = 1;
    Vu_0[13+(NY0) * 6] = 1;
    Vu_0[14+(NY0) * 7] = 1;
    Vu_0[15+(NY0) * 8] = 1;
    Vu_0[16+(NY0) * 9] = 1;
    Vu_0[17+(NY0) * 10] = 1;
    Vu_0[18+(NY0) * 11] = 1;
    Vu_0[19+(NY0) * 12] = 1;
    Vu_0[20+(NY0) * 13] = 1;
    Vu_0[21+(NY0) * 14] = 1;
    Vu_0[22+(NY0) * 15] = 1;
    Vu_0[23+(NY0) * 16] = 1;
    Vu_0[24+(NY0) * 17] = 1;
    Vu_0[25+(NY0) * 18] = 1;
    Vu_0[26+(NY0) * 19] = 1;
    Vu_0[27+(NY0) * 20] = 1;
    Vu_0[28+(NY0) * 21] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 5;
    W[1+(NY) * 1] = 5;
    W[2+(NY) * 2] = 1;
    W[3+(NY) * 3] = 1;
    W[4+(NY) * 4] = 1;
    W[5+(NY) * 5] = 2;
    W[6+(NY) * 6] = 0.5;
    W[7+(NY) * 7] = 0.0005000000000000001;
    W[8+(NY) * 8] = 0.0005000000000000001;
    W[9+(NY) * 9] = 0.0001;
    W[10+(NY) * 10] = 0.0001;
    W[11+(NY) * 11] = 0.0001;
    W[12+(NY) * 12] = 0.0001;
    W[13+(NY) * 13] = 0.0001;
    W[14+(NY) * 14] = 0.0001;
    W[15+(NY) * 15] = 0.0001;
    W[16+(NY) * 16] = 0.0001;
    W[17+(NY) * 17] = 0.0001;
    W[18+(NY) * 18] = 0.0001;
    W[19+(NY) * 19] = 0.0001;
    W[20+(NY) * 20] = 0.0001;
    W[21+(NY) * 21] = 0.0001;
    W[22+(NY) * 22] = 0.0001;
    W[23+(NY) * 23] = 0.0001;
    W[24+(NY) * 24] = 0.0001;
    W[25+(NY) * 25] = 0.0001;
    W[26+(NY) * 26] = 0.0001;
    W[27+(NY) * 27] = 0.0001;
    W[28+(NY) * 28] = 0.0001;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* Vx = calloc(NY*NX, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(NY) * 0] = 1;
    Vx[1+(NY) * 1] = 1;
    Vx[2+(NY) * 2] = 1;
    Vx[3+(NY) * 3] = 1;
    Vx[4+(NY) * 4] = 1;
    Vx[5+(NY) * 5] = 1;
    Vx[6+(NY) * 6] = 1;
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    double* Vu = calloc(NY*NU, sizeof(double));
    // change only the non-zero elements:
    Vu[7+(NY) * 0] = 1;
    Vu[8+(NY) * 1] = 1;
    Vu[9+(NY) * 2] = 1;
    Vu[10+(NY) * 3] = 1;
    Vu[11+(NY) * 4] = 1;
    Vu[12+(NY) * 5] = 1;
    Vu[13+(NY) * 6] = 1;
    Vu[14+(NY) * 7] = 1;
    Vu[15+(NY) * 8] = 1;
    Vu[16+(NY) * 9] = 1;
    Vu[17+(NY) * 10] = 1;
    Vu[18+(NY) * 11] = 1;
    Vu[19+(NY) * 12] = 1;
    Vu[20+(NY) * 13] = 1;
    Vu[21+(NY) * 14] = 1;
    Vu[22+(NY) * 15] = 1;
    Vu[23+(NY) * 16] = 1;
    Vu[24+(NY) * 17] = 1;
    Vu[25+(NY) * 18] = 1;
    Vu[26+(NY) * 19] = 1;
    Vu[27+(NY) * 20] = 1;
    Vu[28+(NY) * 21] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 100;
    W_e[1+(NYN) * 1] = 100;
    W_e[2+(NYN) * 2] = 1;
    W_e[3+(NYN) * 3] = 1;
    W_e[4+(NYN) * 4] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    double* Vx_e = calloc(NYN*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_e[0+(NYN) * 0] = 1;
    Vx_e[1+(NYN) * 1] = 1;
    Vx_e[2+(NYN) * 2] = 1;
    Vx_e[3+(NYN) * 3] = 1;
    Vx_e[4+(NYN) * 4] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    free(Vx_e);




    // slacks
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    Zl[0] = 99999;
    Zl[1] = 99999;
    Zl[2] = 99999;
    Zl[3] = 99999;
    Zl[4] = 99999;
    Zl[5] = 99999;
    Zl[6] = 99999;
    Zl[7] = 99999;
    Zl[8] = 99999;
    Zl[9] = 99999;
    Zl[10] = 99999;
    Zl[11] = 99999;
    Zl[12] = 99999;
    Zl[13] = 99999;
    Zl[14] = 99999;
    Zl[15] = 99999;
    Zl[16] = 99999;
    Zl[17] = 99999;
    Zl[18] = 99999;
    Zl[19] = 99999;
    Zl[20] = 99999;
    Zl[21] = 99999;
    Zl[22] = 99999;
    Zl[23] = 99999;
    Zl[24] = 99999;
    Zl[25] = 99999;
    Zl[26] = 99999;
    Zl[27] = 99999;
    Zl[28] = 99999;
    Zl[29] = 99999;
    Zu[0] = 99999;
    Zu[1] = 99999;
    Zu[2] = 99999;
    Zu[3] = 99999;
    Zu[4] = 99999;
    Zu[5] = 99999;
    Zu[6] = 99999;
    Zu[7] = 99999;
    Zu[8] = 99999;
    Zu[9] = 99999;
    Zu[10] = 99999;
    Zu[11] = 99999;
    Zu[12] = 99999;
    Zu[13] = 99999;
    Zu[14] = 99999;
    Zu[15] = 99999;
    Zu[16] = 99999;
    Zu[17] = 99999;
    Zu[18] = 99999;
    Zu[19] = 99999;
    Zu[20] = 99999;
    Zu[21] = 99999;
    Zu[22] = 99999;
    Zu[23] = 99999;
    Zu[24] = 99999;
    Zu[25] = 99999;
    Zu[26] = 99999;
    Zu[27] = 99999;
    Zu[28] = 99999;
    Zu[29] = 99999;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);



    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(7 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);












    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    idxbu[4] = 4;
    idxbu[5] = 5;
    idxbu[6] = 6;
    idxbu[7] = 7;
    idxbu[8] = 8;
    idxbu[9] = 9;
    idxbu[10] = 10;
    idxbu[11] = 11;
    idxbu[12] = 12;
    idxbu[13] = 13;
    idxbu[14] = 14;
    idxbu[15] = 15;
    idxbu[16] = 16;
    idxbu[17] = 17;
    idxbu[18] = 18;
    idxbu[19] = 19;
    idxbu[20] = 20;
    idxbu[21] = 21;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -5;
    ubu[0] = 5;
    lbu[1] = -5;
    ubu[1] = 5;
    ubu[2] = 1000;
    ubu[3] = 1000;
    ubu[4] = 1000;
    ubu[5] = 1000;
    ubu[6] = 1000;
    ubu[7] = 1000;
    ubu[8] = 1000;
    ubu[9] = 1000;
    ubu[10] = 1000;
    ubu[11] = 1000;
    ubu[12] = 1000;
    ubu[13] = 1000;
    ubu[14] = 1000;
    ubu[15] = 1000;
    ubu[16] = 1000;
    ubu[17] = 1000;
    ubu[18] = 1000;
    ubu[19] = 1000;
    ubu[20] = 1000;
    ubu[21] = 1000;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);






    /* Path constraints */

    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 4;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    lbx[0] = -1;
    ubx[0] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);


    // set up nonlinear constraints for stage 1 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;
    lh[0] = -1;
    lh[1] = -1;
    lh[2] = -1;
    lh[3] = -1;
    lh[16] = 0.95;
    lh[17] = 0.4;
    lh[18] = 0.4;
    lh[19] = 0.4;
    lh[20] = 0.4;
    lh[21] = 0.4;
    lh[22] = 0.4;
    lh[23] = 0.4;
    lh[24] = 0.4;
    lh[25] = 0.4;
    lh[26] = 0.4;
    lh[27] = 0.4;
    lh[28] = 0.4;
    lh[29] = 0.4;
    lh[30] = 0.4;
    lh[31] = 0.4;
    lh[32] = 0.4;
    lh[33] = 0.4;
    lh[34] = 0.4;
    lh[35] = 0.4;
    lh[36] = 0.4;
    uh[0] = 1;
    uh[1] = 1;
    uh[2] = 1;
    uh[3] = 1;
    uh[4] = 1000000000;
    uh[5] = 1000000000;
    uh[6] = 1000000000;
    uh[7] = 1000000000;
    uh[8] = 1000000000;
    uh[9] = 1000000000;
    uh[10] = 1000000000;
    uh[11] = 1000000000;
    uh[12] = 1000000000;
    uh[13] = 1000000000;
    uh[14] = 1000000000;
    uh[15] = 1000000000;
    uh[16] = 1.05;
    uh[17] = 1000000000;
    uh[18] = 1000000000;
    uh[19] = 1000000000;
    uh[20] = 1000000000;
    uh[21] = 1000000000;
    uh[22] = 1000000000;
    uh[23] = 1000000000;
    uh[24] = 1000000000;
    uh[25] = 1000000000;
    uh[26] = 1000000000;
    uh[27] = 1000000000;
    uh[28] = 1000000000;
    uh[29] = 1000000000;
    uh[30] = 1000000000;
    uh[31] = 1000000000;
    uh[32] = 1000000000;
    uh[33] = 1000000000;
    uh[34] = 1000000000;
    uh[35] = 1000000000;
    uh[36] = 1000000000;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i-1]);
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i-1]);
        
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "uh", uh);
        
        
    }
    free(luh);








    // set up soft bounds for nonlinear constraints
    int* idxsh = malloc(NSH * sizeof(int));
    idxsh[0] = 6;
    idxsh[1] = 7;
    idxsh[2] = 8;
    idxsh[3] = 9;
    idxsh[4] = 10;
    idxsh[5] = 11;
    idxsh[6] = 12;
    idxsh[7] = 13;
    idxsh[8] = 14;
    idxsh[9] = 15;
    idxsh[10] = 17;
    idxsh[11] = 18;
    idxsh[12] = 19;
    idxsh[13] = 20;
    idxsh[14] = 21;
    idxsh[15] = 22;
    idxsh[16] = 23;
    idxsh[17] = 24;
    idxsh[18] = 25;
    idxsh[19] = 26;
    idxsh[20] = 27;
    idxsh[21] = 28;
    idxsh[22] = 29;
    idxsh[23] = 30;
    idxsh[24] = 31;
    idxsh[25] = 32;
    idxsh[26] = 33;
    idxsh[27] = 34;
    idxsh[28] = 35;
    idxsh[29] = 36;
    double* lush = calloc(2*NSH, sizeof(double));
    double* lsh = lush;
    double* ush = lush + NSH;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxsh", idxsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lsh", lsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ush", ush);
    }
    free(idxsh);
    free(lush);



    /* terminal constraints */




















}


static void ddmr_acados_create_set_opts(ddmr_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    double newton_tol_val = 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_tol", &newton_tol_val);

    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 20;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for ddmr_acados_create: step 7
 */
void ddmr_acados_set_nlp_out(ddmr_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for ddmr_acados_create: step 9
 */
int ddmr_acados_create_precompute(ddmr_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int ddmr_acados_create_with_discretization(ddmr_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != DDMR_N && !new_time_steps) {
        fprintf(stderr, "ddmr_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, DDMR_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    ddmr_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = ddmr_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    ddmr_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    ddmr_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    ddmr_acados_create_setup_functions(capsule);
    ddmr_acados_setup_nlp_in(capsule, N, new_time_steps);
    ddmr_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = ddmr_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int ddmr_acados_update_qp_solver_cond_N(ddmr_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from ddmr_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = ddmr_acados_create_precompute(capsule);
    return status;
}


int ddmr_acados_reset(ddmr_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int ddmr_acados_update_params(ddmr_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 98;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int ddmr_acados_update_params_sparse(ddmr_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int ddmr_acados_set_p_global_and_precompute_dependencies(ddmr_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, ddmr_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int ddmr_acados_solve(ddmr_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int ddmr_acados_setup_qp_matrices_and_factorize(ddmr_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int ddmr_acados_free(ddmr_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->expl_vde_forw[i]);
        
        external_function_external_param_casadi_free(&capsule->expl_ode_fun[i]);
        external_function_external_param_casadi_free(&capsule->expl_vde_adj[i]);
    }
    free(capsule->expl_vde_adj);
    free(capsule->expl_vde_forw);
    
    free(capsule->expl_ode_fun);

    // cost

    // constraints
    for (int i = 0; i < N-1; i++)
    {
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);



    return 0;
}


void ddmr_acados_print_stats(ddmr_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[1616];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int ddmr_acados_custom_update(ddmr_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *ddmr_acados_get_nlp_in(ddmr_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *ddmr_acados_get_nlp_out(ddmr_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *ddmr_acados_get_sens_out(ddmr_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *ddmr_acados_get_nlp_solver(ddmr_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *ddmr_acados_get_nlp_config(ddmr_solver_capsule* capsule) { return capsule->nlp_config; }
void *ddmr_acados_get_nlp_opts(ddmr_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *ddmr_acados_get_nlp_dims(ddmr_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *ddmr_acados_get_nlp_plan(ddmr_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
