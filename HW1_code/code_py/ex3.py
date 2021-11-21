import time
import numpy as np
import cvxpy as cvx
from IPython import display
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from code_py.trunc_normal_sampling import get_truncated_normal

class Two_Blocks_Optimization:
    np.random.seed(7)
    def __init__(self, x, y, x_test, y_test, rho, sigma, N):
        self.N = N
        self.P = x.shape[0]
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.rho = rho
        self.sigma = sigma
        
    def tanh(self, t):
        """
        tanh activation function definition
        :params t: input vector
        """
        result = (np.exp(2 * self.sigma * t) - 1) / (np.exp(2 * self.sigma * t) + 1)
        return result
    
    def forward_pass_cvx_block(self,w,b):
        """
        forward pass function
        :params w: weights matrix
        :params b: bias vector
        :output -> forward pass matrix
        """
        return self.tanh(np.dot(self.x,w) + b)
    
    def optime_cvx_block(self,w,b,verbose_flag=False):
        """
        Convex function optimization block
        :params w: cvxpy symbolic variable -- vector to optimize
        :
        """
        #params v: cvxpy symbolic variable -- vector to optimize
        v = cvx.Variable((1, self.N))
        #regularization parameter definition -- 
        # removed terms depended by w and b since the partial derivative =0 respect to v
        reg_error = (cvx.norm(v, 2))**2
        #returning the mat from the forward pass
        mat = self.forward_pass_cvx_block(w,b)
        train_err = (1 / (2 * self.x.shape[0])) * cvx.sum_squares((mat@v.T)-self.y)
        #define the objective function to minimize as combination of the train error and regularization error
        cost = train_err + 0.5 * self.rho * reg_error
        #calling the Minimize method
        objective = cvx.Minimize(cost)
        prob = cvx.Problem(objective)
        #solving the quadratic convex error
        loss = prob.solve(solver=cvx.ECOS,verbose=verbose_flag)
        n_iter = prob.solver_stats.num_iters
        return loss,v.value,n_iter
    
    def _squaredL2norm(self, x):
        """
        Function to compute the L2 (euclidean) norm
        :params x: generic n_dim vector
        """
        return np.sum(x**2)
    
    def forward_pass(self, x, omega, args):
        """
        forward_pass for the non convex block
        :params x: input data
        :params omega: flatten vector of v,b,w
        :params arg: external argument that we need to keep fixed while optimizing w and b 
        :output -> forward pass matrix
        """
        omega_ = omega.copy()
        v = args[0]
        b = omega_[:self.N].reshape(1, self.N)
        w = omega_[self.N:].reshape(self.x.shape[1], self.N)
        result = np.dot(self.tanh(np.dot(x, w) + b), v.T)
        return result

    def error_function(self, omega,arg):
        """
        error function of the non convex block
        :params omega: flatten vector of v,b,w
        :params arg: external argument that we need to keep fixed while optimizing w and b 
        :output -> Least Square Error 
        """
        v= arg[0]
        omega_ = omega.copy()
        b = omega_[:self.N].reshape(1, self.N)
        w = omega_[self.N:].reshape(self.x.shape[1],self.N)
        fwd_pass = np.dot(self.tanh(np.dot(self.x, w) + b), v.T)
        err = (1/(2*self.P))*np.sum((fwd_pass-self.y)**2)
        reg = 0.5*self.rho*(self._squaredL2norm(v)+self._squaredL2norm(w)+self._squaredL2norm(b))
        return err+reg

    def fun_grad(self, omega, arg):
        """
        function to evaluate the gradient
        :params omega: flatten vector of v,b,w
        :params arg: external argument that we need to keep fixed while optimizing w and b
        :output -> gradient vector
        """
        v = arg[0]
        omega_ = omega.copy()
        b = omega_[:self.N].reshape(1, self.N)
        w = omega_[self.N:].reshape(self.x.shape[1], self.N)
        # The following variables will be used in the upcoming functions
        # So, for better organisation we are defining them here
        z = np.dot(self.x, w) + b
        h = self.tanh(z)
        y_hat = np.dot(h, v.T)
        loss = (1 / (2 * self.P)) * np.sum((y_hat - self.y) ** 2) + (self.rho / 2) * (
                    np.sum(v ** 2) + np.sum(w ** 2) + np.sum(b ** 2))
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

        return np.concatenate((dE_db, dE_dw)).flatten()

    def optim(self,omega,v,grad=True,verbose=False):
        np.random.seed(7)
        """
        optimizaton method function to optimize the non convex block 
        :params omega: flatten vector of v,b,w 
        :params v: value of the vector v -- convex block
        :params grad: Bool -> if True, the gradient function will be called
        :params verbose: allowing printing during the optimization process

        :outuput -> opt object ,opt b, opt w, opt x , opt loss, val_loss, train_loss
        """

        if grad:
            opt_time = time.time()
            opt_pass = minimize(self.error_function,omega,method='CG', jac=self.fun_grad,tol=1e-4, args = [v])
            if verbose== True:
                print('Time spent to optimize the function:',time.time()-opt_time,'sec')
        else:
            opt_time = time.time()
            opt_pass = minimize(self.error_function,omega,method='CG', args = [v])
            if verbose== True:
                print('Time spent to optimize the function:',time.time()-opt_time,'sec')
        
        opt_b = opt_pass.x[:self.N].reshape(1, self.N)
        opt_w = opt_pass.x[self.N:].reshape(self.x.shape[1],self.N)

        train_loss = (1 / (2 * self.P)) * np.sum((self.forward_pass(self.x, opt_pass.x, [v]) - self.y) ** 2)
        val_loss = (1 / (2 * len(self.x_test))) * np.sum((self.forward_pass(self.x_test, opt_pass.x, [v]) - self.y_test) ** 2)
        
        return opt_pass,opt_b, opt_w, opt_pass.x, opt_pass.fun, val_loss, train_loss


    def Early_Stopping(self, init_class, min_delta, patiente,max_iter):
        np.random.seed(7)
        """
        Early Stopping function
        :params init_class: an initialised class 
        :params min_delta: threshold below which Early Stoppign is activated, monitoring validation loss
        :params patiente: number of epochs to wait during min_delta condition is verified
        :params max_iter: maximum number of iterations to wait before forcing 

        output -> dict{
                :params omega: flatten vector with v,b,w concatenated 
                :params n_fun_eval: n of function evaluations 
                :params n_grad_eval: n of gradient evaluations
                :params execution_time: elapsed time from the calling of the solver (seconds)
                :params val_loss: Validations Loss
                :params train_loss: Number of train
                :params outer_success: Numbers of outer success of the non convex optimization block
                :params n_iterations: Number of iterations of the non-convex optimization block
                :params n_fun_eval_cvx: Number of iterations of the convex block optimization
                }
        """
        
        #defining usefull variables
        val_loss_plot   = []
        train_loss_plot = []
        success_counter = []
        Early_Stopping= False
        current = float("inf")
        count = 0 
        iterations = 0
        n_fun_eval = 0
        n_fun_eval_cvx = 0 
        n_grad_eval = 0
        
        #define the parameter for the sampling
        trunc_norm_params = {
            "low" : -3,
            "upp" :  3,
            "mean":  0,
            "sd"  :  1
        }
        #sampling from the above interval
        w = get_truncated_normal(trunc_norm_params,self.x.shape[1]*self.N).reshape(self.x.shape[1],self.N)
        b = np.ones((1, self.N))
        v = get_truncated_normal(trunc_norm_params,self.N).reshape(1, self.N)
        omega = np.concatenate((b,w))
        
        start_time = time.time()
        while Early_Stopping == False:
            
            #Block1: non convex block -> w and b optimization keeping v fixed
            opt_pass,opt_b, opt_w, opt_pass.x, opt_pass.fun, _, _ =init_class.optim(omega,v)
            #Block2: convex optimization -> v optimization keeping w and b fixed
            loss,opt_v,n_iter=init_class.optime_cvx_block(opt_w,opt_b)

            #train loss computation
            train_loss = (1 / (2 * self.P)) * np.sum(((np.dot(self.tanh(np.dot(self.x, w) + b), v.T)) - self.y) ** 2)
            #val loss computation
            val_loss = (1 / (2 * 64)) * np.sum(((np.dot(self.tanh(np.dot(self.x_test, w) + b), v.T)) - self.y_test) ** 2)


            success_counter.append(opt_pass.success)
            n_fun_eval_cvx+=n_iter
            n_fun_eval  += opt_pass.nfev
            n_grad_eval += opt_pass.njev

            # minimizing pass 
            if (np.abs(val_loss-current)>=min_delta) and iterations<=max_iter:
                current = val_loss
                count = 0
                Early_Stopping = False

            # ES activation and updating counter
            elif (np.abs(val_loss-current)<=min_delta) and iterations<=max_iter:
                count+=1
                #print(count)
                Early_Stopping = False
                if count==patiente:
                    Early_Stopping = True
                    print("Simulation Stopped due to Early Stopping")
                    print("The optimised v,b,w are stored in omega")
                    self.omega = np.concatenate((v, b, w)).flatten()
                    execution_time= (time.time() - start_time)
                    summary = {
                        "omega":self.omega,
                        "n_fun_eval":n_fun_eval,
                        "n_grad_eval":n_grad_eval,
                        "execution_time":execution_time,
                        "val_loss":val_loss_plot,
                        "train_loss":train_loss_plot,
                        "outer_success":sum(success_counter),
                        "n_iterations":iterations,
                        "n_fun_eval_cvx":n_fun_eval_cvx}
                    return self.omega,summary
            #forcing the ES due to number of iterations
            elif iterations >max_iter:
                Early_Stopping = True
                print("Simulation Stopped due to Max iterations")
                self.omega = np.concatenate((v, b, w)).flatten()
                execution_time= (time.time() - start_time)
                summary = {
                    "omega":self.omega,
                    "n_fun_eval":n_fun_eval,
                    "n_grad_eval":n_grad_eval,
                    "execution_time":execution_time,
                    "val_loss":val_loss_plot,
                    "train_loss":train_loss_plot,
                    "outer_success":sum(success_counter),
                    "n_iterations":iterations}
                return self.omega,summary


            iterations+=1    
            # if (iterations+1) % 5 == 0:
            #     #print(loss)
            #     print (f'Iteration [{iterations+1}]')
                
            w,b,v=opt_w,opt_b,opt_v
            val_loss_plot.append(val_loss)
            train_loss_plot.append(train_loss)


    
    def Plot3D(self,omega):
        """
        Plot3D: function to plot the example function
        :params omega: flatten vector with best values of v,b,w concatenated, selected from the optimization 
        :output -> a 3D plot
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
        #compute the forward pass using w,b and v. 
        new_y = np.dot(self.tanh(np.dot(new_x, w) + b), v.T)
        a_1_ = np.reshape(x_1, (50, 50))
        a_2_ = np.reshape(x_2, (50, 50))
        y_ = np.reshape(new_y, (50, 50))
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(a_1_, a_2_, y_, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.show()

    def train_loss_Plot(self,val_loss_plot,train_loss_plot):
        """
        train loss plot
        :params val_loss_plot: list with all the validation loss per epoch
        :params train_loss: list with all the validation loss per epoch
        """
        plt.style.use('ggplot')
        xx = np.arange(0,len(val_loss_plot)-1)
        plt.plot(xx, val_loss_plot[1:], '-ok', color='red',markersize=3,label="Validation loss")
        plt.plot(xx, train_loss_plot[1:], '-ok', color='black',markersize=3,label="Train loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title('Validation vs Train Loss')
        plt.legend(loc="upper right")
        plt.show()
        