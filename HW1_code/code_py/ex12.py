from IPython import display
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class RBF_Network:
    def __init__(self, x, y, x_test, y_test, rho, sigma, N):
        self.P = x.shape[0]
        self.feat = x.shape[1]
        self.x = x # (P,2)
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        # Hyper-parameters
        self.N = N
        self.rho = rho
        self.sigma = 1/(sigma**2)
        # parameters
        np.random.seed(1960415)
        self.v = np.random.normal(size=(1, self.N))
        self.c = np.random.normal(size=(self.feat, self.N))
        self.omega = np.concatenate((self.v, self.c)).flatten()
        
    def _squaredL2norm(self, x):
        return np.sum(x**2)

    def _rbf(self, x, c, sigma):
        '''
        RBF function.

        :param x: input -> (P,2)
        :param c: centers -> (2,N)
        :param N: number of neurons of the hidden layer
        :param sigma: the spread sigma > 0 in the RBF function

        :return output -> (P,N)
        '''
        diff = np.expand_dims(x,axis=2) - c # (186, 2, N)
        return np.exp(-1 * sigma * np.sum(diff**2, axis=1)) # (186, N)


    def forward_pass(self, x, omega):
        '''
        Compute the function f(x), RBF Network.

        :param x: input -> (P,2)
        :param omega: flatten concatenation of the parameters v and c

        :return f(x) -> (P,1)
        '''
        # parameter
        omega_ = omega.copy()
        v = omega_[:self.N].reshape(self.N,1) # (N, 1)
        c = omega_[self.N:].reshape(self.feat,self.N) # (2, N)

        return self._rbf(x,c,self.sigma) @ v # (P, 1)

    def _error_function(self, omega):
        '''
        Compute the regularized training error: E(omega;pi) = MSE + L2 Normalization.
        Where omega represents the parameters and pi the hyper-parameters.

        :param omega: flatten concatenation of the parameters v and c

        :retun E(omega;pi)
        '''
        # parameter
        omega_ = omega.copy()
        v = omega_[:self.N].reshape(self.N,1)
        c = omega_[self.N:].reshape(self.feat, self.N)

        err = (1/(2*self.P))*np.sum((self.forward_pass(self.x, omega)-self.y)**2)
        reg = (0.5 * self.rho)*(self._squaredL2norm(v)+self._squaredL2norm(c))
        return err+reg
    
    def _fun_grad(self, omega):
        '''
        Compute the gradients of loss function wrt variables v and c.
        P - number of samples
        2 - features, a.k.a x and y of the function
        N - number of neurons
        
        :param omega: flatten concatenation of v and c
        
        :return output -> flatten concatenation of dE_dv, dE_dc
        '''
        # parameter
        omega_ = omega.copy()
        v = omega_[:self.N].reshape(self.N, 1)
        c = omega_[self.N:].reshape(self.feat, self.N)

        diff = np.expand_dims(self.x,axis=2) - c # (P, 2, N)
        rbf = self._rbf(self.x, c,self.sigma) # (P, N)
        fun = rbf @ v # (P, 1)
        E_fun = (1/self.P)*(fun-self.y) # (P, 1)

        drbf_dc = (2/self.sigma**2)*rbf*np.swapaxes(diff,0,1) # (P,N)*(2,P,N) --> (2,P,N)
        red_P = np.tensordot(E_fun,drbf_dc,axes=([0],[1])).reshape(2,self.N) # solving the sample summation 
        
        loss_c = v.T*red_P # (2,N)
        loss_v = rbf.T @ E_fun # (N,P)@(P, 1) --> (N,1)

        reg_v = self.rho * v # (N,1)
        reg_c = self.rho * c # (2,N)

        E_v = (loss_v + reg_v).flatten() # (N,1)
        E_c = (loss_c + reg_c).flatten() # (2,N)

        return np.concatenate((E_v, E_c))

    def optim(self,grad=True,verbose=False):
        '''
        Optimize the objective function by gradient-based approach

        :param grad: gradient function itself
        :param verbose: True, if it needs to print the optimisation time spent
        :return: output of the minimize function, optimal v and c parameter values, concatenation of the v and c,
        objective function value, validation loss, training loss and optimisation time
        '''
        if grad:
            opt_time = time.time()
            opt_pass = minimize(self._error_function,self.omega,method='CG', jac=self._fun_grad, tol=1e-5)
            opt_time = time.time()-opt_time
            if verbose:
                print('Time spent to optimize the function:',opt_time,'sec')
        else:
            opt_time = time.time()
            opt_pass = minimize(self._error_function,self.omega,method='CG')
            opt_time = time.time()-opt_time
            if verbose:
                print('Time spent to optimize the function:',opt_time,'sec')
        
        opt_v = opt_pass.x[:self.N].reshape(self.N,1)
        opt_c = opt_pass.x[self.N:].reshape(self.feat, self.N)

        val_loss = (1 / (2 * len(self.x_test))) * np.sum((self.forward_pass(self.x_test, opt_pass.x) - self.y_test) ** 2)
        train_loss = (1 / (2 * self.P)) * np.sum((self.forward_pass(self.x, opt_pass.x) - self.y) ** 2)

        return opt_pass, opt_v, opt_c, opt_pass.x, opt_pass.fun, val_loss, train_loss, opt_time

    def plot3D(self, omega):
        '''
        Plot the approximation of the function.

        :param omega: flatten concatenation of v and c

        :return plot
        '''
        x_1 = np.linspace(-2, 2, 50)
        x_2 = np.linspace(-3, 3, 50)
        x_1 = x_1.repeat(50)
        x_2 = np.tile(x_2, 50)
        new_x = np.concatenate((x_1.reshape(2500,1), x_2.reshape(2500,1)),axis = 1)
        new_y = self.forward_pass(new_x, omega)
        a_1_ = np.reshape(x_1, (50, 50))
        a_2_ = np.reshape(x_2, (50, 50))
        y_ = np.reshape(new_y, (50, 50))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(a_1_, a_2_, y_, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
        plt.show()

    def gradient_norm(self, optimal_omega):
        '''
        Compute the value of the norm of the gradient at the initial and the final points

        :param optimal_omega: optimal omega value/ output of the optim() function

        :return L2 norm of the initial and final point's gradient
        '''
        initial_gradient = self._fun_grad(self.omega)
        final_gradient = self._fun_grad(optimal_omega)

        return np.linalg.norm(initial_gradient), np.linalg.norm(final_gradient)

    def objective_func_value(self, optimal_omega):
        '''
        Compute the objective function of the initial and the optimal variables

        :param optimal_omega: flatten concatenation of optimal v and c
        :return: Initial and the final objective function values
        '''
        initial_obj = self._error_function(self.omega)
        final_obj = self._error_function(optimal_omega)

        return initial_obj, final_obj

    def initial_train_error(self):
        '''
        Compute the initial training error

        :return: Initial training error
        '''
        initial_train_loss = (1 / (2 * self.P)) * np.sum((self.forward_pass(self.x, self.omega) - self.y) ** 2)

        return initial_train_loss
