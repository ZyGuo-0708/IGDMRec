# @email: zyguo0166@gmail.com
r"""
CD-Net
# Update: 2026/02/05
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class CDNet(nn.Module):

    def __init__(self, config, n_items, norm=False):
        super(CDNet, self).__init__()
        self.time_emb_dim = config['time_emb']
        self.norm = norm
        self.n_items = n_items
        self.hid = config['n_hid']
        self.emb_layer = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(self.time_emb_dim*2, self.time_emb_dim),
        )
        self.encode = nn.Sequential(
            nn.Linear(self.n_items, self.hid),
        )
        self.decode = nn.Sequential(
            nn.Linear(self.hid, self.n_items),
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(self.hid*2+10, self.hid),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.hid*2+10, self.hid + 1500),
            nn.GELU(),
            nn.Linear(self.hid + 1500, self.hid)
        )

        self.drop1 = nn.Dropout(0.5)

        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.n_items
        )
        nn.init.normal_(self.none_embedding.weight,0,0)

        self.init_weights()

    def init_weights(self):
        for layer in self.emb_layer:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
        for layer in self.mlp1:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
        for layer in self.encode:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
        for layer in self.decode:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                fan_out = size[0]
                fan_in = size[1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, con, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        x = self.encode(x)
        con = self.encode(con)
        if self.norm:
            x = F.normalize(x)
            con = F.normalize(con)
        x = self.drop1(x)
        con = self.drop1(con)
        h = torch.cat([x, con.to_dense(), emb], dim=-1)
        h = self.mlp1(h)
        h = self.decode(h)
        return h


    def forward_unco(self, x, timesteps):
        device = self.none_embedding.weight.device
        con = self.none_embedding(torch.tensor([0], device = device))#.to(self.device)
        con = torch.cat([con.view(1,self.n_items)]*x.shape[0], dim=0) #self.n_items
        con = self.encode(con)
        con.zero_()
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        x = self.encode(x)
        if self.norm:
            x = F.normalize(x)
            con = F.normalize(con)
        x = self.drop1(x)
        con = self.drop1(con)
        h = torch.cat([x, con, emb], dim=-1)
        h = self.mlp1(h)
        h = self.decode(h)
        return h

def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SinEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim//2
        embeddings = math.log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim,device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim =-1)
        return  embeddings

