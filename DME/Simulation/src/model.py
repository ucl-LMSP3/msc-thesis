import copy
import torch
import torch.nn as nn
import numpy as np
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.models import GPRegression
from DME.Simulation.src.utils import _zero_mean_function, get_mean_params, get_kernel_params


class DMEGP(nn.Module):
    def __init__(self, input_dim, random_effects_dim, mean_fn=None):
        super(DMEGP, self).__init__()
        self.input_dim = input_dim
        self.random_effects_dim = random_effects_dim
        self.mean_fn = mean_fn

        # define kernel function
        kernel = RBF(self.random_effects_dim, lengthscale=torch.ones(self.random_effects_dim)) 

        # define gaussian process regression model
        self.gp_model = GPRegression(X=torch.ones(1, self.random_effects_dim),
                                     y=None,
                                     kernel=kernel,
                                     mean_function=_zero_mean_function)

    def forward(self, train_y, train_Z, test_Z, loss_fn, lr, n_adapt):
        """
        Return p(Y_test|train_y, train_Z, test_Z).
        """
        train_on_gpu = torch.cuda.is_available()
        device = torch.device('cuda' if train_on_gpu else 'cpu')
        if train_on_gpu:
            train_y, train_Z, test_Z = train_y.to(device), train_Z.to(device), test_Z.to(device)

        gp_clone = self.define_new_GP()
        params_clone = copy.deepcopy(self.gp_model.state_dict())
        gp_clone.load_state_dict(params_clone)
        gp_clone.set_data(train_Z, train_y)
        optimizer = torch.optim.Adam(get_kernel_params(gp_clone), lr=lr)
        for _ in range(n_adapt):
            optimizer.zero_grad()
            loss = loss_fn(gp_clone.model, gp_clone.guide)
            loss.backward()
            optimizer.step()
        pred_y, var_y = gp_clone(test_Z, noiseless=False)
        return pred_y, var_y, loss

    def forward_mean(self, test_X):
        """
        Return \mu(test_X) that is predictions of global mean function.
        """
        train_on_gpu = torch.cuda.is_available()
        device = torch.device('cuda' if train_on_gpu else 'cpu')
        if train_on_gpu:
            test_X = test_X.to(device)
        mean_preds = self.mean_fn(test_X)
        mean_preds = torch.transpose(mean_preds, -1, -2)
        return mean_preds

    def step(self, train_X, train_y, train_Z, mean_optim, gp_optim, loss_fn_fixed,
             loss_fn, lr, n_adapt):
        """
        Optimise the model by alternating between optimisinng the mean function
        and the GP.
        """
        train_on_gpu = torch.cuda.is_available()
        device = torch.device('cuda' if train_on_gpu else 'cpu')
        if train_on_gpu:
            train_X, train_y, train_Z = train_X.to(device), train_y.to(device), train_Z.to(device)
        # Mean function
        mean_optim.zero_grad()
        loss_fixed = loss_fn_fixed(self.mean_fn(train_X), train_y)
        loss_fixed.backward()
        mean_optim.step()
        
        # GP
        train_y = train_y.view(-1)
        self.gp_model.set_data(train_Z, train_y)
        gp_optim.zero_grad()
        loss_gp = loss_fn(self.gp_model.model, self.gp_model.guide)
        loss_gp.backward()
        gp_optim.step()
        return loss_fixed.item() + loss_gp.item()

    def define_new_GP(self):
        # define kernel function
        kernel = RBF(self.random_effects_dim, lengthscale=torch.ones(self.random_effects_dim))

        # define gaussian process regression model
        gp_model = GPRegression(X=torch.ones(1, self.random_effects_dim),
                                y=None,
                                kernel=kernel,
                                mean_function=_zero_mean_function)
        return gp_model