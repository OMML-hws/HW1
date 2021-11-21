from IPython import display
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import truncnorm
import cvxpy as cvx
import numpy as np
from code_py.trunc_normal_sampling import get_truncated_normal
from tqdm import tqdm


class MLP_Extreme_Opt(object):

    def __init__(self, x_train, y_train,x_test,y_test, rho, sigma, N):  # bound_list):

        self.N = N
        self.x = x_train
        self.y = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.rho = rho
        self.sigma = sigma


    def tanh(self, t):
        """
        :param t: input of the tanh

        :return tanh(t)
        """
        result = (np.exp(2 * self.sigma * t) - 1) / (np.exp(2 * self.sigma * t) + 1)
        return result

    def forward_pass(self,b,w):
        """
        Compute the forward_pass of the first layer plus activation function. 
        (! no dot product with v)

        :param b: bias
        :param w: weights

        :return forward_pass
        """
        return self.tanh(np.dot(self.x, w) + b)

    def prediction(self, x, b, w, v):
        """
        Compute the new y given x

        :param x: input
        :param b: bias
        :param w: weights
        :param v: weights

        return new_y
        """
        return self.tanh(np.dot(x, w) + b) @ v.T

    def train_loss(self,b,w,v):
        """
        :param b: bias
        :param w: weights
        :param v: weights of the second layer

        :return train_loss
        """
        return (1/(2 * self.x.shape[0])) * np.sum(((self.forward_pass(b,w)@v.T)-self.y)**2)

    def validation_loss(self,b,w,v):
        """
        :param b: bias
        :param w: weights
        :param v: weights of the second layer

        :return val_loss
        """
        forward_pass    = self.tanh(np.dot(self.x_test,w) +b)
        loss = (1/(2 * self.x_test.shape[0])) * np.sum(((forward_pass@v.T)-self.y_test)**2)
        return loss
         


    def convex_training_error_opt(self,b,w,verbose):
        """
        function to compute the minimization of the quadratic convex function.
        Solver: ECOS

        :param w: weights
        :param b: bias
        :param verbose: To see the output from the solvers

        :return loss, v, cvx.Problem
        """
        v = cvx.Variable((1, self.N))
        reg_error = (cvx.norm(v, 2))**2
        mat = self.forward_pass(b,w)
        train_err = (1 / (2 * self.x.shape[0])) * cvx.sum_squares((mat@v.T)-self.y)
        cost = train_err + 0.5 * self.rho * reg_error
        objective = cvx.Minimize(cost)
        prob = cvx.Problem(objective)
        loss = prob.solve(solver=cvx.ECOS,verbose=verbose) # ECOS default option for this problem

        return loss,v.value,prob

    
    def Random_Sampling(self,n_sample):
        """
        Repeat the random choice of the weights w and b from a trunced norm to find the best configuration.
        
        :param n_sample: number of iteration

        :return best b and w
        """
        trunc_norm_params = {
            "low" : -3,
            "upp" :  3,
            "mean":  0,
            "sd"  :  1
        }

        sampling_history = {}
        print("sampling on truncated normal to look for the best b,w setting...")
        for _ in tqdm(range(n_sample)):
            w_rnd = get_truncated_normal(trunc_norm_params,self.x.shape[1]*self.N).reshape(self.x.shape[1],self.N)
            b_rnd = get_truncated_normal(trunc_norm_params,self.N).reshape(1, self.N)

            ext_MLP_net = MLP_Extreme_Opt(x_train=self.x,y_train=self.y,x_test=self.x_test,y_test=self.y_test,rho= self.rho,sigma=self.sigma,N=self.N) 
            loss,_,_    = ext_MLP_net.convex_training_error_opt(b=b_rnd,w=w_rnd,verbose=False)
            omega = np.concatenate((b_rnd, w_rnd)).flatten()
            sampling_history[loss] = omega
            #print(loss)
        sampling_history_ = sorted(sampling_history.items())
        best_loss = sampling_history_[0][0]
        
        best_omega = sampling_history_[0][1]
        b = best_omega[:self.N].reshape(1, self.N)
        w = best_omega[self.N:].reshape(self.x.shape[1], self.N)
        print("Best Loss: ", best_loss)
        return b,w


    def Plot3D(self,omega):
        """
        Plot the approximation of the function.

        :param omega: flatten concatenation of v, b and w

        :return plot
        """
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