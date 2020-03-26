import torch
from torch.nn.functional import binary_cross_entropy
from utils.general import write_to_csv

"""
Variational lower bound (the objective to be maximized)
more in https://arxiv.org/abs/1312.6114

cross entropy between the data and reconstructed data
+
KL divergence of distributions

NOTE: use binary_cross_entropy_with_logits if last activation in network is not a sigmoid
"""


def variational_ELBO(recon_x, x, log_var, mu):
    cross_entropy = binary_cross_entropy(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return cross_entropy + kl_divergence


"""
Weighted version of the Variational lower bound (the objective to be maximized)

cross entropy between the data and reconstructed data
+
(weighting term) * KL divergence of distributions
"""


def weighted_variational_ELBO(weight, recon_x, x, log_var, mu):
    cross_entropy = binary_cross_entropy(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return cross_entropy + (weight * kl_divergence)


"""
Function to log data on a given network and the optimizer used
for training the model. Logs the 
    - model name
    - number of trainable parameters in the model
    - optimizer used
"""


def log(model=None, optimizer=None):
    assert model is not None and optimizer is not None
    print("using optimizer: " + optimizer.__class__.__name__)
    print("using model: " + model.__class__.__name__)

    net_parameters = sum(p.numel() for p in model.parameters())
    print("number of model params: " + str(net_parameters))


"""
Function to perform the re-parameterization of the layers as described in 
section 2.4 of https://arxiv.org/pdf/1312.6114v10.pdf
"""


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


"""
Flattens an input (tensor) to a C x 1 x 1 vector where C is the number of channels
in the tensor.

Example:
256 x 2 x 2 tensor flattened -> 1024 x 1 x 1 tensor
"""


def flatten(x):
    return x.view(x.size(0), -1)


"""
Un-flattens an input (tensor) from a C x 1 x 1 vector to a (C, H, W) tensor.
Useful in the decoder architecture when doing Conv2dTranspose

Example:
1024 x 1 x 1 tensor un-flattened to a -> 256 x 2 x 2 (C=256, h=2, w=2) tensor that is then fed into a 
(C_i=256 -> C_o=128) decoder Conv2dTranspose layer
"""


def un_flatten(x, channels, h, w):
    return x.view(x.size(0), channels, h, w)


"""
Function to save data to disk from the training run of the model such as
    - historical training loss data
    - historical test loss data
    - model weights
"""


def save_training_data(training_loss_data=None, test_loss_data=None, output_dir='results'):
    assert training_loss_data is not None and test_loss_data is not None

    write_to_csv(training_loss_data, output_dir + '/historic_training_loss.csv')
    write_to_csv(test_loss_data, output_dir + '/historic_testing_loss.csv')
