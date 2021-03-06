from mpi4py import MPI
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres
import scipy.sparse.linalg as spla
import sys
from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh, rhs_comp2_mesh
import matplotlib.pyplot as plt


class gmres_counter(object):
    def __init__(self, disp=False):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))





class grayscott_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 2D with finite differences and periodic BC

    Attributes:
        A: second-order FD discretization of the 2D laplace operator
        dx: distance between two spatial nodes (same for both directions)
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'D0', 'D1','f','k', 'newton_maxiter', 'newton_tol', 'lin_tol', 'lin_maxiter']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        #if len(problem_params['nvars']) != 2:
        #    raise ProblemError('this is a 2d example, got %s' % problem_params['nvars'])
        #if problem_params['nvars'][0] != 2*problem_params['nvars'][1]:
        #    raise ProblemError('nvars is not compatible with system, got %s' % problem_params['nvars'])
        #if problem_params['nvars'][0] % 2 != 0:
        #    raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(grayscott_fullyimplicit, self).__init__(problem_params['nvars'], dtype_u, dtype_f, problem_params)

        # compute dx and get discretization matrix A
        #self.nv = self.params.nvars[1]
        self.dx = 1. / self.params.nvars[1]  ##################################################################################
        #print(self.dx)
        #exit()
        self.A = self.__get_A(self.params.nvars, self.dx)
        self.xvalues = np.array([i * self.dx  for i in range(self.params.nvars[1])])


        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0
        self.warning_count = 0
        self.linear_count = 0
        self.time_for_solve = 0

    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (list): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        zero_pos = 2
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(([N[1] - i - 1 for i in reversed(range(zero_pos - 1))],
                                  [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N[1]))

        A = sp.diags(dstencil, doffsets, shape=(N[1], N[1]), format='csc')
        A = sp.kron(A, sp.eye(N[1])) + sp.kron(sp.eye(N[1]), A)
        A *= 1.0 / (dx ** 2)

        return A

    # noinspection PyTypeChecker
    def solve_system(self, rhs, factor, u_0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """
        #print("u0", abs(u_0))
        #print("rhs", abs(rhs))
        #print("factor", abs(factor))                

        
	#print(u0.values.shape)
        u0 = self.dtype_u(u_0).values[0,:,:].flatten()
        u1 = self.dtype_u(u_0).values[1,:,:].flatten()   
        
        u=self.dtype_u(u_0).values.flatten()  
             
        z = self.dtype_u(self.init, val=0.0).values.flatten()



        Id = sp.eye(self.params.nvars[1]*self.params.nvars[2])

        counter = gmres_counter()
        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:


            g0 = u0 - factor * (self.params.D0*self.A.dot(u0) - u0 * u1** 2 + self.params.f*(1-u0)) - rhs.values[0,:,:].flatten()           
            g1 = u1 - factor * (self.params.D1*self.A.dot(u1) + u0 * u1** 2 - (self.params.f+self.params.k)*u1) - rhs.values[1,:,:].flatten()

            
            g=np.zeros([2*self.params.nvars[1]*self.params.nvars[1]])
                
            g[:self.params.nvars[1]*self.params.nvars[1]] += (g0)
            g[self.params.nvars[1]*self.params.nvars[1]:] += (g1)  
            

                        
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break


            #dg=np.zeros([2*self.params.nvars[1]*self.params.nvars[1], 2*self.params.nvars[1]*self.params.nvars[1]])
            
            #dg[:self.params.nvars[1]*self.params.nvars[1], :self.params.nvars[1]*self.params.nvars[1]] += (Id - factor * (self.params.D0*self.A + sp.diags((-u1 ** 2 -self.params.f), offsets=0))) #00

            #dg[:self.params.nvars[1]*self.params.nvars[1], self.params.nvars[1]*self.params.nvars[1]:] += - factor * (sp.diags(( -2.0*u0*u1 ), offsets=0)) #01            
            
            #dg[self.params.nvars[1]*self.params.nvars[1]:, self.params.nvars[1]*self.params.nvars[1]:] += (Id - factor * (self.params.D1*self.A + sp.diags((2.0*u0*u1-(self.params.f+self.params.k)), offsets=0))) #11
            
            #dg[self.params.nvars[1]*self.params.nvars[1]:, :self.params.nvars[1]*self.params.nvars[1]] +=  - factor *  sp.diags(u1**2, offsets=0) #10           


            dg = sp.bmat([[(Id - factor * (self.params.D0*self.A + sp.diags((-u1 ** 2 -self.params.f), offsets=0))), - factor * (sp.diags(( -2.0*u0*u1 ), offsets=0))],[- factor *  sp.diags(u1**2, offsets=0), (Id - factor * (self.params.D1*self.A + sp.diags((2.0*u0*u1-(self.params.f+self.params.k)), offsets=0))) ]])


            #print("das system dg gray sott", dg)
            #print("is it", sp.issparse(dg))
        
            t1 = MPI.Wtime()  
                                
            u -= gmres(dg, g, x0=z, tol=self.params.lin_tol, maxiter=self.params.lin_maxiter, callback=counter)[0]

            t2 = MPI.Wtime()             

            self.newton_itercount += 1
            self.linear_count += counter.niter
            self.time_for_solve += t2-t1 
            if counter.niter== self.params.lin_maxiter:
                self.warning_count += 1          
            u0=u.reshape(self.params.nvars)[0,:,:].flatten()
            u1=u.reshape(self.params.nvars)[1,:,:].flatten()
            n += 1


        
        me = self.dtype_u(self.init)
        me.values = u.reshape(self.params.nvars)

        #print("me", abs(me))                
        #exit()

        self.newton_ncalls += 1

        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        #print(abs(u))
        f = self.dtype_f(self.init)
        v0 = u.values[0,:,:].flatten()
        v1 = u.values[1,:,:].flatten()        
        
        #print("abs ", abs(abs(- v0 * v1** 2 + self.params.f*(1-v0))))
        
        #f0 = (- v0 * v1* v1)  + (1-v0)*self.params.f
        #f1 = ( v0 * v1**2 - self.params.k*v1  )  # - (self.params.f+self.params.k)*v1
        
        f0 = self.params.D0*self.A.dot(v0) - v0 * v1**2 + self.params.f*(1-v0)
        f1 = self.params.D1*self.A.dot(v1) + v0 * v1**2  - (self.params.f+self.params.k)*v1
        
        f.values[0,:,:] = f0.reshape([self.params.nvars[1], self.params.nvars[2]])
        f.values[1,:,:] = f1.reshape([self.params.nvars[1], self.params.nvars[2]])        


        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)

        me = self.dtype_u(self.init)
        dx = 1.0 / self.params.nvars[1]

        
        #for i in range(self.params.nvars[1],self.params.nvars[1]):
        #    for j in range(self.params.nvars[1],self.params.nvars[1]):
        #        #print("laeut", 1.0 - 0.5 * np.power(np.sin(np.pi * i * dx / 100) *np.sin(np.pi * j * dx / 100), 100))
        #        me.values[0, i, j] = 1.0 - 0.5 * np.power(np.sin(np.pi * i * dx / 100) *np.sin(np.pi * j * dx / 100), 100)
        #        me.values[1, i, j] = 0.25 * np.power(np.sin(np.pi * i * dx / 100) *np.sin(np.pi * j * dx / 100), 100)

        #fname = 'anfangswerte.npz'
        #loaded = np.load(fname) 
        
        #me.values[0,:,:] = loaded['u_0'].reshape(32,32,2)[:,:,0]
        #me.values[1,:,:] = loaded['u_0'].reshape(32,32,2)[:,:,1]        

        #plt.imshow(me.values[0,:,:], interpolation='bilinear')
        #plt.savefig('anfangswertmsh.pdf')
        #exit()                    
        #print("zweites", me.values[0,14:17,14:17])        
        #exit()
        #print(self.params.nvars[1])
        
        for i in range(self.params.nvars[1]):
            for j in range(self.params.nvars[1]):            
                if( (self.xvalues[i]-0.5)*(self.xvalues[i]-0.5) + (self.xvalues[j]-0.5)*(self.xvalues[j]-0.5) < 0.0025):
                    me.values[0,i, j] = 0.5
                    me.values[1,i, j] = 0.25
                else:
                    me.values[0,i, j] = 1.0
                    me.values[1,i, j] = 0.0                    


        return me


