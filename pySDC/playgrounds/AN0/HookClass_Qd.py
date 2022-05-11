import numpy as np

from pySDC.core.Hooks import hooks

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right

import json


class Q_hook(hooks):
    def __init__(self):
        """
        Initialization of particles output
        """

        super(Q_hook, self).__init__()


        self.coll = CollGaussRadau_Right(3, 0, 1)




    def pre_run(self, step, level_number):
        """
        Overwrite default routine called before time-loop starts
        Args:
            step: the current step
            level_number: the current level number
        """
        super(Q_hook, self).pre_run(step, level_number)

        
        L = step.levels[level_number]

        problem = L.prob

        

        if L.sweep.params.QI=='Opt':


            self.opt_path = "models/5l"+str(problem.time_rank)+".json" 

            f=open(self.opt_path, "r")
            data=json.load(f)
            q_d_opt_=np.array(data["q_d_opt"])
            q_d_opt=q_d_opt_
            f.close()   

            problem.QD = np.ndarray(shape=problem.Laplace().shape, dtype=float) 

            for idx, x_ in np.ndenumerate(problem.Laplace()):

                x=  -x_*L.dt

                if (x>-100 and x<-0.01):
                    index=int(10000-int(-x*100))  
                    problem.QD[idx] =q_d_opt[index]
                else:
                    problem.QD[idx] =q_d_opt[0] 




        elif L.sweep.params.QI=='Inv':
                problem.QD = np.ndarray(shape=problem.Laplace().shape, dtype=float) 
                num_nodes=3
                for idx, x in np.ndenumerate(problem.Laplace()):
                    coll = CollGaussRadau_Right(num_nodes, 0, 1)
                    Q = coll.Qmat[1:, 1:]
                    lam= -x*problem.nu*L.dt
                    C = np.eye(num_nodes) - lam * Q
                    new=np.linalg.inv(C)@ np.ones(num_nodes, dtype=np.float64)

                    if (abs(-1+new[problem.time_rank])>1e-14):
                        problem.QD[idx] = (1./lam- coll.nodes[problem.time_rank]/(-1.+new[problem.time_rank])) 
                    else:
                        problem.QD[idx] = 0


    def post_step(self, step, level_number):
        """
        Default routine called after each iteration
        Args:
            step: the current step
            level_number: the current level number
        """
        super(Q_hook, self).post_step(step, level_number)

        L = step.levels[level_number]
        return None
