"""Fingering Convection Test Script

Following Gaurad & Brummell (2015). 

"""
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

from filter_field import filter_field

import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = (40., 10.)
Prandtl = 1.
tau = 1/10.
R0 = 5.
R0_inv = 1./R0

N = np.sqrt(Prandtl*(R0-1.))
logger.info("Inverse buoyancy frequency = {}".format(1/N))

# Create bases and domain
x_basis = de.Fourier('x', 128, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 32, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics with 
problem = de.IVP(domain, variables=['p','mu','u','w','T', 'mu_z', 'T_z', 'u_z', 'w_z'])
problem.parameters['Pr'] = Prandtl
problem.parameters['tau'] = tau
problem.parameters['R0_inv'] = R0_inv
problem.parameters['Lz'] = Lz
problem.parameters['Lx'] = Lx

problem.substitutions['vol_avg(A)']   = 'integ(A)/(Lx*Lz)'

problem.add_equation("dt(u) + Pr*(dx(p) - (dx(dx(u)) + dz(u_z)))            = -(u*dx(u) + w*u_z)")
problem.add_equation("dt(w) + Pr*(dz(p) - (T - mu) - (dx(dx(w)) + dz(w_z))) = -(u*dx(w) + w*w_z)")
problem.add_equation("dt(T) - (dx(dx(T)) + dz(T_z)) + w                     = -(u*dx(T) + w*T_z)")
problem.add_equation("dt(mu) - tau*(dx(dx(mu)) + dz(mu_z)) + R0_inv*w       = -(u*dx(mu) + w*mu_z)")
problem.add_equation("dx(u) + w_z = 0")
problem.add_equation("T_z - dz(T) = 0")
problem.add_equation("mu_z - dz(mu) = 0")
problem.add_equation("u_z - dz(u) = 0")
problem.add_equation("w_z - dz(w) = 0")
problem.add_bc("left(T) = 0")
problem.add_bc("left(mu) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(T) = 0")
problem.add_bc("right(mu) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")


# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
T = solver.state['T']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
pert =  1e-3 * noise*np.sin(np.pi*z/Lz)
T['g'] = pert
filter_field(T)

# Initial timestep
dt = 0.001

# Integration parameters
solver.stop_sim_time = 10000#0
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1., max_writes=50)
snapshots.add_system(solver.state)

timeseries = solver.evaluator.add_file_handler('timeseries',iter=1, max_writes=np.inf)
timeseries.add_task("vol_avg(sqrt(u*u))",name='urms')
timeseries.add_task("vol_avg(sqrt(w*w))",name='wrms')
timeseries.add_task("vol_avg(sqrt(T*T))",name='Trms') 
timeseries.add_task("vol_avg(sqrt(mu*mu))",name='mrms')
timeseries.add_task("vol_avg(sqrt(p*p))",name='prms')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=0.8,
                     max_change=1.5, min_change=0.5, max_dt=3.5, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("0.5*(u*u + w*w)", name="Ekin")
flow.add_property("u", name="u")
flow.add_property("w", name="w")

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)#, trim=True)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e, Kinetic Energy: %e, u max: %e, w max: %e' %(solver.iteration, solver.sim_time, dt, flow.volume_average('Ekin'), flow.max('u'), flow.max('w')))
        #logger.info('urms = %f' %flow.max('urms'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
