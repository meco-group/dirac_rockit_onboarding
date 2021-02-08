from pylab import *
from rockit import *
from casadi import vertcat

# OCP based on https://web.casadi.org/blog/mpc-simulink/
# SQP method 'sqpmethod 'based on https://www.syscop.de/files/2015ws/numopt/numopt_0.pdf
# QP method 'qrqp'
#     active-set method
#     implementation paper http://www.optimization-online.org/DB_FILE/2018/05/6642.pdf
#     see more implementation comments in https://github.com/casadi/casadi/blob/3.5.5/casadi/core/runtime/casadi_qp.hpp
#          sparse linear algebra has commetns in https://github.com/casadi/casadi/blob/3.5.5/casadi/core/runtime/casadi_qr.hpp
#     we will probably switch to an interior point qp solver, or osqp for some applications


ocp = Ocp(T=2.0) # 10 seconds horizon

x1 = ocp.state()
x2 = ocp.state()
u  = ocp.control()

start_x = vertcat(0.1,0)

# ==============
# System model
# ==============

e = 1 - x2**2

ocp.set_der(x1, e * x1 - x2 + u)
ocp.set_der(x2, x1)

# ===============================
# Define optimal control problem
# ===============================

# Least-squares objective
ocp.add_objective(ocp.integral(x1**2+x2**2+u**2))
ocp.add_objective(ocp.at_tf(x1**2))

# Path constraints
ocp.subject_to(x1 >= -0.25)
ocp.subject_to(-1 <= (u <= 1 ))

# Boundary condition
x0 = ocp.parameter(2, 1)
x = vertcat(x1,x2)
ocp.subject_to(ocp.at_t0(x)==x0)

# convert Optimal Control -> Nonlinear constraint programming nx*(N+1)+nu*N

ocp.method(MultipleShooting(N=2,intg='rk'))

options = {}
options["qpsol"] = 'qrqp';
options["expand"] =True
options["qpsol_options"] = {"print_iter": False, "print_header": False}
options["print_iteration"] = False
options["print_header"] = False
options["print_status"] = False
options["print_time"] = False
ocp.solver('sqpmethod',options)

ocp.set_value(x0, start_x)

# ===============================
# A single solve
# ===============================

sol = ocp.solve()


[t,usol] = sol.sample(u, grid='integrator',refine=10)

figure()
plot(t,usol)
xlabel('t [s]')
title('Control action [N]')
grid(True)

ocp.spy()

show()


# ===============================
# An MPC Function
# ===============================

# rockit -> CasADi
MPC_step = ocp.to_function('MPC_step', [x0], [ocp.value(ocp.at_t0(u))], ["x0"], ["u_optimal"])

MPC_step.generate('MPC_step.c',{"main":True})

raise Exception()

current_x = start_x

# For plant simulation
plant = ocp._method.discrete_system(ocp)

for i in range(100):
    # Compute optimal action
    optimal_action = MPC_step(current_x)

    # Apply to plant
    current_x = plant(x0=current_x,u=optimal_action,T=10.0/20)["xf"]

    # Add some noise
    if i>50:
        current_x = current_x + vertcat([0,0.1*np.random.rand()])

    print(optimal_action)

# mex MPC_step.c -largeArrayDims
MPC_step.generate('MPC_step.c',{'mex':True})

