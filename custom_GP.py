
class MultifidelityGPModel(SingleTaskGP, GPyTorchModel):
        _num_outputs = 1  # to inform the BoTorch api
        def _init_(self, train_x, train_y, likelihood, composition='product'):
            super(ExactGPModel, self)._init_(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.Rbfx_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]-1))
            # self.Rbfx_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[-1]-1))
            self.Rbfs_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            # self.Rbfs_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=1))
            # self.Rbfs_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
            self.comp = composition
        def forward(self, x):
            assert(x.shape[-1] > 1)
            d_plus_one = x.shape[-1]
            if x.ndim > 2: # necessary to separate out fidelity and nonfidelity part of x
                X = x[..., -d_plus_one:-1]
                s = x[..., -1:]
            else:
                X = x[:, :-1]
                s = x[:, -1:]
            mean_x = self.mean_module(x)
            if self.comp.lower() == 'product':
                covar_x = self.Rbfx_module(X) * self.Rbfs_module(s)
            elif self.comp.lower() == 'sum':
                covar_x = self.Rbfx_module(X) + self.Rbfs_module(s)
            else:
                raise NotImplementedError('The requested composition is not implemented yet.')
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        
        