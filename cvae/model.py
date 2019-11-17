import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model_constants import *


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim, conditional, num_labels):

        super().__init__()

        self.conditional = conditional

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_dim)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_dim)
  
    def idx2onehot(self, idx, n):
        assert torch.max(idx).item() < n
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)

        onehot = torch.zeros(idx.size(0), n)
        onehot.scatter_(1, idx, 1)

        return onehot

    def forward(self, x, c=None):

        if self.conditional:
            # c = self.idx2onehot(c, n=C_DIM)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_dim + num_labels
        else:
            input_size = latent_dim

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            # c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x

class CVAE(nn.Module):

    def __init__(self,
                 x_dim=X_DIM,
                 c_dim=C_DIM,
                 latent_dim=LATENT_DIM,
                 run_id=1,
                 experiment_path_prefix=EXPERIMENT_PATH_PREFIX,
                 hiddens=HIDDEN_LAYERS,
                 initial_learning_rate=INITIAL_LEARNING_RATE,
                 lr_update_cycles=LEARNING_RATE_UPDATE_CYCLES,
                 weight_decay=WEIGHT_DECAY,
                 cuda_available=CUDA_AVAILABLE,
                 use_tboard=USE_TENSORBOARD,
                 tf_sub_path=TF_PATH):
        """
        :param x_dim: x dimension
        :param c_dim: c dimension
        :param learning_rate:
        """
        super(CVAE, self).__init__()
        self.x_dim = x_dim
        self.c_dim = c_dim
        self.latent_dim = latent_dim

        self.cuda_available = cuda_available
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

        self.hiddens = hiddens
        self.encoder_layer_sizes = [self.x_dim + self.c_dim] + self.hiddens
        self.decoder_layer_sizes = self.hiddens + [self.x_dim] 
        self.encoder = Encoder(
            self.encoder_layer_sizes, self.latent_dim, True, self.c_dim)
        self.decoder = Decoder(
            self.decoder_layer_sizes, self.latent_dim, True, self.c_dim)

        print("Encoder : {}".format(self.encoder))
        print("Decoder : {}".format(self.decoder))
        self.initial_learning_rate = initial_learning_rate
        self.current_lr = self.initial_learning_rate
        self.weight_decay = weight_decay

        self.use_tboard = use_tboard
        self.tf_sub_path = tf_sub_path
        self.run_id = run_id
        self.experiment_path_prefix = experiment_path_prefix
        if self.cuda_available:
            self.models = [self.models[i].cuda() for i in range(self.num_nets)]
        if self.use_tboard:
            from tensorboardX import SummaryWriter
            self.tf_path = '{}/{}/{}/'.format(self.experiment_path_prefix, self.run_id, self.tf_sub_path)
            if not os.path.exists(self.tf_path):
                os.makedirs(self.tf_path)
            self.tboard = SummaryWriter(self.tf_path)

    def update_learning_rate(self, lr):
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def get_current_lr(self, episode_num):
        return self.initial_learning_rate / pow(2.0, (episode_num // self.lr_update_cycles))

    def reset_current_lr(self, episode_num):
        this_lr = self.get_current_lr(episode_num)
        if ((self.current_lr > 1.1 * this_lr) or (self.current_lr < 0.9 * this_lr)):
            self.current_lr = this_lr
            self.update_learning_rate(self.current_lr)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path)['state_dict'])

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        model_path = os.path.join(self.experiment_path_prefix, self.run_id, 'model-{}.pkl'.format(suffix))
        torch.save({'state_dict': self.model.state_dict()}, model_path)

    def forward(self, x, c=None):
        if x.dim() > 2:
            x = x.view(-1, 28*28)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_dim])

        # Reparametrize 
        z = eps * std + means

        recon_x = self.decoder(z, c)
        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_dim])

        recon_x = self.decoder(z, c)
        return recon_x

    def loss_fn(self, recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)
