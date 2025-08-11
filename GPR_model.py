import torch
import gpytorch
import os
from tqdm import tqdm
import numpy as np


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        if isinstance(train_x, np.ndarray):
            train_x = torch.from_numpy(train_x).float()
        if isinstance(train_y, np.ndarray):
            train_y = torch.from_numpy(train_y).float()
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.likelihood=likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_hypers(model, likelihood, lr=0.05, maxiter=2000, 
                 save_path='params.pth', tol=1e-8):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses = []
    for i in tqdm(range(maxiter),desc='Training hyper-params for GPR using MarLogLike'):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(model.train_inputs[0])
        # Calc loss and backprop gradients
        loss = -mll(output, model.train_targets)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        losses.append(current_loss)
        #if change in loss is less than tol
        if i>5: 
            if np.abs(current_loss-losses[-2])<tol:
                break
            else:
                pass

    if type(save_path) != type(None):
        torch.save(model.state_dict(),save_path)
    
    losses = np.asarray(losses)
    return losses
    


