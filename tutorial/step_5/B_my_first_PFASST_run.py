import numpy as np

from pySDC.implementations.problem_classes.HeatEquation_1D_FD_forced import heat1d_forced
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh_1D import mesh_to_mesh_1d_dirichlet
from pySDC.implementations.controller_classes.allinclusive_classic_nonMPI import allinclusive_classic_nonMPI

from pySDC.plugins.stats_helper import filter_stats, sort_stats


def main():
    """
    A simple test program to setup a full step hierarchy
    """

    # initialize level parameters
    level_params = {}
    level_params['restol'] = 1E-10
    level_params['dt'] = 0.25

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['do_LU'] = True
    # sweeper_params['do_GS'] = True

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1      # diffusion coefficient
    problem_params['freq'] = 8      # frequency for the test value
    problem_params['nvars'] = [511,255]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = heat1d_forced                    # pass problem class
    description['problem_params'] = problem_params                  # pass problem parameters
    description['dtype_u'] = mesh                                   # pass data type for u
    description['dtype_f'] = rhs_imex_mesh                          # pass data type for f
    description['sweeper_class'] = imex_1st_order                   # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params                  # pass sweeper parameters
    description['level_params'] = level_params                      # pass level parameters
    description['step_params'] = step_params                        # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_1d_dirichlet # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params    # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 4.0

    nsteps = int(Tend/level_params['dt'])
    num_proc_list = [2**i for i in range(int(np.log2(nsteps)+1))]

    for num_proc in num_proc_list:
        print('Working with %2i processes...' %num_proc)
        controller = allinclusive_classic_nonMPI(num_procs=num_proc, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        # compute exact solution and compare
        uex = P.u_exact(Tend)
        err = abs(uex - uend)
        print('Error: %12.8e' % (err))

        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        for item in iter_counts:
            print('Number of iterations for time %4.2f: %2i' % item)
        print()

if __name__ == "__main__":
    main()
