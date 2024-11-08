import torch.nn as nn
from deltaconv.nn import MLP
import torch
from torch_geometric.nn import global_max_pool, global_mean_pool
import torch.nn.functional as F
from deltaconv.models.deltanet_base import DeltaNetBase

class DeltaNetAE(torch.nn.Module):
    def __init__(self, in_channels, point_size, latent_size, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width, centralize_first=True):
        super().__init__()
        self.deltanet_base = DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width)
        self.channels = in_channels
        self.out_dim = in_channels
        self.self_condition = None
        print(sum(p.numel() for p in self.deltanet_base.parameters()))
        
        self.lin_embedding = MLP([sum(conv_channels), latent_size])
        self.classification_head = nn.Sequential(
            MLP([1024 * 2, 512]), nn.Dropout(0.5), MLP([512, 256]), nn.Dropout(0.5),
            nn.Linear(256, latent_size))
        

        
        self.point_size = point_size
        self.dec1 = nn.Linear(latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,3*self.point_size)

    def encoder(self, data): 
        conv_out = self.deltanet_base(data)

        x = torch.cat(conv_out, dim=1)
        x = self.lin_embedding(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)

        x = torch.cat([x_max, x_mean], dim=1)

        return self.classification_head(x)

    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)

    def forward(self, data):
        data = self.encoder(data)
        data = self.decoder(data)
        return data
    
class DeltaNetVAE(torch.nn.Module):
    def __init__(self, in_channels, point_size, latent_size, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width, centralize_first=True):
        super().__init__()
        self.deltanet_base = DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width)
        self.channels = in_channels
        self.out_dim = in_channels
        self.self_condition = None
        print(sum(p.numel() for p in self.deltanet_base.parameters()))
        
        self.lin_embedding = MLP([sum(conv_channels), 1024])
        self.fc_mu = nn.Linear(1024 * 2, latent_size)
        self.fc_logvar = nn.Linear(1024 * 2, latent_size)
        
        self.point_size = point_size
        self.dec1 = nn.Linear(latent_size, 256)
        self.dec2 = nn.Linear(256, 256)
        self.dec3 = nn.Linear(256, 3 * self.point_size)

    def encoder(self, data): 
        conv_out = self.deltanet_base(data)

        x = torch.cat(conv_out, dim=1)
        x = self.lin_embedding(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)

        x = torch.cat([x_max, x_mean], dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        z = F.relu(self.dec1(z))
        z = F.relu(self.dec2(z))
        z = self.dec3(z)
        return z.view(-1, self.point_size, 3)

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        recon_data = self.decoder(z)
        return recon_data