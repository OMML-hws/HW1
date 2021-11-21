from IPython import display
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cvx
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

class RBF_Extreme_Opt(object):
    def __init__(self, x, y, x_test, y_test, rho, sigma, N, c_flag=False):
        self.P = x.shape[0]
        self.feat = x.shape[1]
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.N = N
        self.rho = rho
        self.sigma = 1/( sigma**2 )
        if c_flag:
            self.c = 0
        else:
            self.c = KMeans(n_clusters=self.N, random_state=1).fit(self.x).cluster_centers_.T

    def set_sigma(self, x):
        """
        simple function to compute the squared l2 norm.

        :param x: input matrix

        :return squared l2 norm of x
        """
        self.sigma = 1/( (2*x)**2 )

    def Random_Sampling(self,n_sample):
        """
        Repeat the random choice of the centers from the train set to find the best configuration.
        
        :param n_sample: number of iteration

        :return best centers
        """
        sampling_history = {}
        print("sampling to look for the best c setting...")
        for _ in tqdm(range(n_sample)):
            c = self.x[np.random.choice(self.P, size=self.N, replace=False)].T

            ext_RBF_net = RBF_Extreme_Opt(x=self.x,y=self.y,x_test=self.x_test,y_test=self.y_test,rho=self.rho,sigma=self.sigma,N=self.N) 
            loss,_ ,_   = ext_RBF_net.convex_training_error_opt(c,verbose=False)
            sampling_history[loss] = c
            #print(loss)

        sampling_history_ = sorted(sampling_history.items())
        best_loss = sampling_history_[0][0]
        
        best_c = sampling_history_[0][1]
        print("Best Loss: ", best_loss)
        return best_c 


    # set methods
    def setc(self, new_c):
        self.c = new_c

    def setv(self, new_v):
        self.v = new_v

    # 
    def rbf(self, x, c):
        '''
        RBF function.

        :param x: input -> (P,2)
        :param c: centers -> (2,N)
        :param sigma: the spread sigma > 0 in the RBF function

        :return output -> (P,N)
        '''
        diff = np.expand_dims(x,axis=2) - c
        out = np.exp(-1 * self.sigma * np.sum(diff**2, axis=1))
        return out

    def forward_pass(self, x, c):
        '''
        Compute the function f(x), RBF Network.
        Note that we do note compute the final dot product.

        :param x: input -> (P,2)
        :param c: centers

        :return f(x) -> (P,1)
        '''
        return self.rbf(x,c) # (P,N)

    def train_loss(self,v,c):
        """
        Compute the train loss

        :param v: parameter v
        :param c: centers of the RBF

        :return train_loss
        """
        return (1/(2 * self.P)) * np.sum(((self.forward_pass(self.x, c)@v.T)-self.y)**2)

    def prediction(self, x, c, v):
        """
        Compute the new y given x

        :param x: input
        :param c: centers
        :param v: weights

        return new_y
        """
        return self.rbf(x,c) @ v.T

    def validation_loss(self,v,c):
        """
        Compute the val loss

        :param v: parameter v
        :param c: centers of the RBF

        :return val_loss
        """
        loss = (1/(2 * len(self.x_test))) * np.sum(((self.forward_pass(self.x_test, c)@v.T)-self.y_test)**2)
        return loss

    def convex_training_error_opt(self, c, verbose=False):
        """
        function to compute the minimization of the quadratic convex function.
        Solver: ECOS

        :param c: centers
        :param verbose: To see the output from the solvers

        :return loss, v, cvx.Problem
        """
        v = cvx.Variable((1, self.N))
        reg_error = (cvx.norm(v, 2))**2
        mat = self.forward_pass(self.x, c)
        train_err = (1 / (2 * self.x.shape[0])) * cvx.sum_squares((mat@v.T)-self.y)
        cost = train_err + 0.5 * self.rho * reg_error
        objective = cvx.Minimize(cost)
        prob = cvx.Problem(objective) 
        loss = prob.solve(solver=cvx.ECOS,verbose=verbose) # cvx.ECOS is the default solver for SOCP
        return loss,v.value,prob

    def Plot3D(self,omega):
        """
        Plot the approximation of the function.

        :param omega: flatten concatenation of v and c

        :return plot
        """
        x_1 = np.linspace(-2, 2, 50)
        x_2 = np.linspace(-3, 3, 50)
        x_1 = x_1.repeat(50)
        x_2 = np.tile(x_2, 50)
        new_x = np.concatenate((x_1.reshape(2500, 1), x_2.reshape(2500, 1)), axis=1)

        omega_ = omega.copy()
        v = omega_[:self.N].reshape(1, self.N)
        c = omega_[self.N:].reshape(new_x.shape[1], self.N)


        new_y = np.dot(self.rbf(new_x,c), v.T)
        a_1_ = np.reshape(x_1, (50, 50))
        a_2_ = np.reshape(x_2, (50, 50))
        y_ = np.reshape(new_y, (50, 50))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(a_1_, a_2_, y_, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.show()
