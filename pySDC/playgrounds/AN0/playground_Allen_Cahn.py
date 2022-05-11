import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.playgrounds.AN0.generic_imex_MPI import generic_imex_MPI
from pySDC.playgrounds.AN0.AllenCahn_MPIFFT import allencahn_imex
from pySDC.playgrounds.AN0.HookClass_Qd import Q_hook

import matplotlib.pyplot as plt
import numpy as np

num_nodes = 5






def run_simulation(spectral=None, ml=None, nprocs_space=None, sweeper_class=None, QI=None, index=None):

    """
    A test program to do SDC, MLSDC and PFASST runs for the 2D NLS equation

    Args:
        spectral (bool): run in real or spectral space
        ml (bool): single or multiple levels
        num_procs (int): number of parallel processors
    """


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


    world_rank = comm.Get_rank()
    world_size = comm.Get_size()


    assert world_size%nprocs_space == 0, 'Total number of processors must be multiple of number of processors in space'  

    # split world communicator to create space-communicators
    if nprocs_space is not None:
        color = int(world_rank / nprocs_space)
    else:
        color = int(world_rank / 1)

    space_comm = comm.Split(color=color)
    space_comm.Set_name('Space-Comm')
    space_size = space_comm.Get_size()
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    if nprocs_space is not None:
        color = int(world_rank % nprocs_space)
    else:
        color = int(world_rank / world_size)
    time_comm = comm.Split(color=color)
    time_comm.Set_name('Time-Comm')
    time_size = time_comm.Get_size()
    time_rank = time_comm.Get_rank()





    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-12
    level_params['dt'] = 1E-03 
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [num_nodes]

    if sweeper_class == 'generic_imex_MPI':
        assert sweeper_params['num_nodes'][0] == time_size, 'Need %s processors in time' % sweeper_params['num_nodes']   

    sweeper_params['initial_guess'] = 'spread'

    sweeper_params['QI'] = [QI]  
    sweeper_params['comm'] = time_comm 

    # initialize problem parameters
    problem_params = dict()
    if ml:
        problem_params['nvars'] = [(128, 128), (64, 64)]
    else:
        problem_params['nvars'] = [(32, 32)]

    problem_params['spectral'] = spectral
    problem_params['comm'] = space_comm
    problem_params['time_comm'] = time_comm

    #problem_params['RL_both'] = True             
    problem_params['dt'] = level_params['dt']    
    #problem_params['imex'] = False 

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = Q_hook  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30 if rank == 0 else 99


    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    description['problem_class'] = allencahn_imex #advectiondiffusion_imex
    description['sweeper_class'] = sweeper_class #generic_imex_MPI # imex_1st_order




    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    #description['space_transfer_class'] = fft_to_fft



    # set time parameters
    t0 = 0.0
    Tend = 24*1e-3 


    # instantiate controller
    MPI.COMM_WORLD.Barrier()    
    wt = MPI.Wtime()

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    MPI.COMM_WORLD.Barrier()    
    run_time = MPI.Wtime()
    # call main function to get things done...


    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    MPI.COMM_WORLD.Barrier()    
    wt = MPI.Wtime() - wt
    wt2 = MPI.Wtime() - run_time

    uex = P.u_exact(Tend)
    err = abs(uex - uend)




    if rank == 0:
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        niters = np.array([item[1] for item in iter_counts])
        out = f'   Min/Mean/Max number of iterations: ' \
              f'{np.min(niters):4.2f} / {np.mean(niters):4.2f} / {np.max(niters):4.2f}'
        print(out)

        out = f'Error: {err:6.4e}'
        print(out)

        timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
        out = f'Time to solution: {timing[0][1]:6.4f} sec.'
        print(out)


        print('Mean number of iterations %s' %( np.mean(niters)))






def main():
    """
    Little helper routine to run the whole thing

    Note: This can also be run with "mpirun -np 2 python B_pySDC_with_mpi4pyfft.py"
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()


    n_space = int(world_size/num_nodes)

    run_RL=True
    run_MIN3=True
    run_LU=True

    wt = MPI.Wtime()
   
    if run_RL:
        if rank ==0: print("############ Opt/RL")
        MPI.COMM_WORLD.Barrier()
        run_simulation(spectral=True, ml=False, nprocs_space=n_space, sweeper_class = generic_imex_MPI, QI='Opt', index=0)
        MPI.COMM_WORLD.Barrier()


    if run_MIN3:
        if rank ==0: print("############ MIN3")
        MPI.COMM_WORLD.Barrier()
        run_simulation(spectral=True, ml=False, nprocs_space=n_space, sweeper_class = generic_imex_MPI, QI='MIN3', index=1)
        MPI.COMM_WORLD.Barrier()


    if run_LU:     
        if rank ==0: print("############ LU")
        MPI.COMM_WORLD.Barrier()    
        run_simulation(spectral=True, ml=False, nprocs_space=world_size, sweeper_class = imex_1st_order, QI='LU', index=2)
        MPI.COMM_WORLD.Barrier()      

    wt = MPI.Wtime() - wt
    if rank == 0: print("***** runtime", wt)

if __name__ == "__main__":
    main()
