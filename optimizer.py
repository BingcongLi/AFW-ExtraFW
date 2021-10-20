import numpy as np


class Optimizer:
    def __init__(self, x_init, loss, constraint):
        """
        Base class for all optimizers
        Args:
            x_init (np array): initialized x. It is assumed to be in the constraint set.
            loss (class): loss function
            constraint (class): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
        """

        self.x = x_init.copy()
        self.loss_fn = loss 
        self.constraint = constraint
        self.n_iter = 0
    
    
    def step(self):
        raise NotImplementedError
        
    

class ExtraFW(Optimizer):
    def __init__(self, x_init, loss, constraint):
        """
        ExtraFW with parameter-free step size
        see paper: https://ojs.aaai.org/index.php/AAAI/article/view/17012

        Args:
            x_init (np array): initialized x. It is assumed to be in the constraint set.
            loss (class): loss function
            constraint (class): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
        """
        
        super(ExtraFW, self).__init__(x_init, loss, constraint)
        self.y = x_init.copy()
        self.v = x_init.copy()
        # here we use a lazy approach to initialize g, since we have \delta_0 = 1
        self.g = np.zeros((self.loss_fn.dim,1))
        
        
    def step(self):
        # step 1: 1st gradient calculation
        delta = 2 / (self.n_iter + 2)
        self.y = (1 - delta) * self.x + delta * self.v
        grad_y = self.loss_fn.grad(self.y)
        
        # step 2: solve the first fw subproblem
        g_hat = (1 - delta) * self.g + delta * grad_y 
        v_hat = self.constraint.fw_subprob(g_hat)
        
        # step 4: update and 2nd gradient calculation
        self.x = (1 - delta) * self.x + delta * v_hat
        grad_x = self.loss_fn.grad(self.x)
        self.g = (1 - delta) * self.g + delta * grad_x 
        self.v = self.constraint.fw_subprob(self.g)
        
        self.n_iter += 1
        loss_fn_val = self.loss_fn.function_value(self.x)
        return loss_fn_val

    

class AFW(Optimizer):
    def __init__(self, x_init, loss, constraint):
        """
        accelerated FW (AFW) with parameter-free step size
        see paper: https://ieeexplore.ieee.org/abstract/document/9457128

        Args:
            x_init (np array): initialized x. It is assumed to be in the constraint set.
            loss (class): loss function
            constraint (class): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
        """
        
        super(AFW, self).__init__(x_init, loss, constraint)
        
        self.y = x_init.copy()
        # here we use a lazy approach to initialize g, since we have \delta_0 = 1
        self.v = np.zeros((self.loss_fn.dim,1))
        self.g = np.zeros((self.loss_fn.dim,1))

        
    def step(self):
        # step 1: gradient calculation
        delta = 2 / (self.n_iter+2)
        self.y = (1 - delta) * self.y + delta * self.v
        grad_y = self.loss_fn.grad(self.y)
        
        # step 2: solve the fw subproblem
        self.g = (1 - delta) * self.g + delta * grad_y 
        self.v = self.constraint.fw_subprob(self.g)
            
        # step 3: update
        self.x = (1 - delta) * self.x + delta * self.v
        self.n_iter += 1
        loss_fn_val = self.loss_fn.function_value(self.x)
        return loss_fn_val


    
class FW(Optimizer):
    def __init__(self, x_init, loss, constraint):
        """
        vanilla FW with parameter-free step size

        Args:
            x_init (np array): initialized x. It is assumed to be in the constraint set.
            loss (class): loss function
            constraint_type (class): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
        """
        super(FW, self).__init__(x_init, loss, constraint)
        
        
    def step(self):
        # step 1: gradient calculation
        grad_x = self.loss_fn.grad(self.x)
        
        # step 2: solve the fw subproblem
        v = self.constraint.fw_subprob(grad_x)
        
        # step 3: determine the step size   
        delta = 2 / (self.n_iter + 2)
     
        # step 4: update
        self.x = (1 - delta) * self.x + delta * v
        self.n_iter += 1
        loss_fn_val = self.loss_fn.function_value(self.x)
        return loss_fn_val
      
        
        
class GD(Optimizer):
    def __init__(self, x_init, loss, constraint):
        """
        gradient descent (GD) using step size 1/L

        Args:
            x_init (np array): initialized x. It is assumed to be in the constraint set.
            loss (class): loss function
            constraint (str): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
        """
        super(GD, self).__init__(x_init, loss, constraint)
        # standard step size to use
        self.lr = 1/self.loss_fn.L
        
    def step(self):
        # step 1: gradient calculation
        grad_x = self.loss_fn.grad(self.x)
        
        # step 2: update and projection
        self.x = self.constraint.projection(self.x - self.lr*grad_x)
        loss_fn_val = self.loss_fn.function_value(self.x)
        self.n_iter += 1
        return loss_fn_val
    


class NAG(Optimizer):
    def __init__(self, x_init, loss, constraint):
        """
        Nesterov's accelerated gradient method (NAG)

        Args:
            x_init (np array): initialized x. It is assumed to be in the constraint set.
            loss (class): loss function
            constraint_type (str): constraint set, currently only 'l1', 'l2', 'n_supp' are supported
            R (float): radius of constraint set, i.e., \| x \| <= R
        """
        super(NAG, self).__init__(x_init, loss, constraint)
        self.y = x_init.copy()
        self.v = x_init.copy()
        # standard step size
        self.lr = 1 / self.loss_fn.L
        
        
    def step(self):
        # step 1: gradient calculation
        delta = 2 / (self.n_iter + 2)
        self.y = delta * self.v + (1 - delta) * self.x
        grad_y = self.loss_fn.grad(self.y)
        
        # step 2: update and projection
        self.x = self.constraint.projection(self.y - self.lr * grad_y)
        alpha = (self.n_iter + 2) / (2*self.loss_fn.L)
        self.v = self.constraint.projection(self.v - alpha * grad_y)
        
        loss_fn_val = self.loss_fn.function_value(self.x)
        self.n_iter += 1
        return loss_fn_val

    
