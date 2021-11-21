from IPython import display
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class MLP_Network:

    def __init__(self, x, y, x_test, y_test, rho, sigma, N):
        self.P = x.shape[0]
        self.x = x  # (P, 2)
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        # Hyper-parameters
        self.N = N
        self.rho = rho
        self.sigma = sigma
        # parameters
        np.random.seed(1960415)
        self.v = np.random.normal(size=(1, self.N))
        self.b = np.random.normal(size=(1, self.N))
        self.w = np.random.normal(size=(self.x.shape[1], self.N))
        self.omega = np.concatenate((self.v, self.b, self.w)).flatten()

    def tanh(self, t):
        '''
        :param t: input of the tanh
        :return tanh(t)
        '''
        result = (np.exp(2 * self.sigma * t) - 1) / (np.exp(2 * self.sigma * t) + 1)
        return result

    def forward_pass(self, x, omega):
        '''
        Compute the function f(x), MLP Network.

        :param x: input -> (P,2)
        :param omega: flatten concatenation of the parameters v, b and w

        :return f(x) -> (P,1)
        '''
        omega_ = omega.copy()
        v = omega_[:self.N].reshape(1, self.N)
        b = omega_[self.N:2 * self.N].reshape(1, self.N)
        w = omega_[2 * self.N:].reshape(x.shape[1], self.N)
        result = np.dot(self.tanh(np.dot(x, w) + b), v.T)
        return result

    def error_function(self, omega):
        '''
        Compute the regularized training error: E(omega;pi) = MSE + L2 Normalization.
        Where omega represents the parameters and pi the hyper-parameters.

        :param omega: flatten concatenation of the parameters v, b and w

        :return E(omega;pi)
        '''
        omega_ = omega.copy()
        v = omega_[:self.N].reshape(1, self.N)
        b = omega_[self.N:2*self.N].reshape(1, self.N)
        w = omega_[2*self.N:].reshape(self.x.shape[1], self.N)
        err = np.sum((self.forward_pass(self.x, omega)-self.y)**2)
        error = (1/(2*self.P))*err+(self.rho/2)*np.sum(omega_**2)
        return error

    def fun_grad(self, omega):

        '''
        Compute the gradients of loss function wrt variables v, b, and w.

        :param omega: flatten concatenation of v, b, and w -> 4 x N

        :return output -> flatten concatenation of dE_dv, dE_db, and dE_dw -> 4 x N
        '''

        omega_ = omega.copy()
        v = omega_[:self.N].reshape(1, self.N)
        b = omega_[self.N:2 * self.N].reshape(1, self.N)
        w = omega_[2 * self.N:].reshape(self.x.shape[1], self.N)

        # The following variables will be used in the upcoming functions
        # So, for better organisation we are defining them here
        z = np.dot(self.x, w) + b
        h = self.tanh(z)
        y_hat = np.dot(h, v.T)
        loss = (1 / (2 * self.P)) * np.sum((y_hat - self.y) ** 2) + (self.rho / 2) * np.sum(omega_**2)

        # Chain rule to calculate gradients
        dE_dy = (1 / self.P) * (y_hat - self.y)  # (186,1)
        dy_dh = v  # (1, N)
        dy_dv = h  # (186,N)
        dh_dz = np.divide((4 * self.sigma * np.exp(2 * self.sigma * (z))),
                          (np.exp(2 * self.sigma * (z)) + 1) ** 2)  # (186,N)
        dz_dw = self.x  # (186,2)
        dz_db = 1

        dE_dv = np.dot(dE_dy.T, dy_dv) + self.rho * v  # (1,N)
        dE_dh = np.dot(dE_dy, dy_dh)  # (186,N)
        dE_dz = dE_dh * dh_dz  # (186,N)
        dE_db = np.sum(dE_dz * dz_db, axis=0, keepdims=True) + self.rho * b  # (1,N)
        dE_dw = np.dot(dz_dw.T, dE_dz) + self.rho * w  # (2,N)

        return np.concatenate((dE_dv, dE_db, dE_dw)).flatten()

    def optim(self,grad=True,verbose=False):
        '''
        Optimize the objective function by gradient-based approach

        :param grad: gradient function itself
        :param verbose: True, if it needs to print the optimisation time spent
        :return: output of the minimize function, optimal v, b and w parameter values, concatenation of the v, b and w,
        objective function value, validation loss, training loss and optimisation time
        '''
        if grad:
            opt_time = time.time()
            opt_pass = minimize(self.error_function,self.omega,method='CG', jac=self.fun_grad,tol=1e-5)
            opt_time = time.time()-opt_time
            if verbose:
                print('Time spent to optimize the function:',opt_time,'sec')
        else:
            opt_time = time.time()
            opt_pass = minimize(self.error_function,self.omega,method='CG')
            opt_time = time.time() - opt_time
            if verbose:
                print('Time spent to optimize the function:',opt_time,'sec')
            
        opt_v = opt_pass.x[:self.N].reshape(1,self.N)
        opt_b = opt_pass.x[self.N:(2*self.N)].reshape(1,self.N)
        opt_w = opt_pass.x[2*self.N:].reshape(self.x.shape[1],self.N)

        val_loss = (1 / (2 * len(self.x_test))) * np.sum((self.forward_pass(self.x_test, opt_pass.x) - self.y_test) ** 2)
        train_loss = (1 / (2 * self.P)) * np.sum((self.forward_pass(self.x, opt_pass.x) - self.y) ** 2)

        return opt_pass, opt_v, opt_b, opt_w, opt_pass.x, opt_pass.fun, val_loss, train_loss, opt_time

    def Plot3D(self,omega):
        '''
        Plot the approximation of the function.

        :param omega: flatten concatenation of v, b and w

        :return plot
        '''
        x_1 = np.linspace(-2, 2, 50)
        x_2 = np.linspace(-3, 3, 50)
        x_1 = x_1.repeat(50)
        x_2 = np.tile(x_2, 50)
        new_x = np.concatenate((x_1.reshape(2500, 1), x_2.reshape(2500, 1)), axis=1)
        omega_ = omega.copy()
        v = omega_[:self.N].reshape(1, self.N)
        b = omega_[self.N:2 * self.N].reshape(1, self.N)
        w = omega_[2 * self.N:].reshape(new_x.shape[1], self.N)

        new_y = np.dot(self.tanh(np.dot(new_x, w) + b), v.T)
        a_1_ = np.reshape(x_1, (50, 50))
        a_2_ = np.reshape(x_2, (50, 50))
        y_ = np.reshape(new_y, (50, 50))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(a_1_, a_2_, y_, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.show()

    def gradient_norm(self, optimal_omega):
        '''
        Compute the value of the norm of the gradient at the initial and the final points

        :param optimal_omega: optimal omega value/ output of the optim() function

        :return L2 norm of the initial and final point's gradient
        '''
        initial_gradient = self.fun_grad(self.omega)
        final_gradient = self.fun_grad(optimal_omega)

        return np.linalg.norm(initial_gradient), np.linalg.norm(final_gradient)

    def objective_func_value(self, optimal_omega):
        '''
        Compute the objective function of the initial and the optimal variables

        :param optimal_omega: flatten concatenation of optimal v, b and w
        :return: Initial and the final objective function values
        '''
        initial_obj = self.error_function(self.omega)
        final_obj = self.error_function(optimal_omega)

        return initial_obj, final_obj

    def initial_train_error(self):
        '''
        Compute the initial training error

        :return: Initial training error
        '''
        initial_train_loss = (1 / (2 * self.P)) * np.sum((self.forward_pass(self.x, self.omega) - self.y) ** 2)

        return initial_train_loss

