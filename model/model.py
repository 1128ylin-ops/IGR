import torch
import torch.nn as nn
import torch.nn.functional as F
from .Resnet import resnet18

class SimpleMINE(nn.Module):

    def __init__(self, feat_dim, prob_dim, hidden_dim=256):
        super(SimpleMINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + prob_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z, probs):
        B, D = z.size()
        joint = torch.cat([z, probs], dim=1)  # [B, D+C]
        t_pos = self.net(joint)               # [B,1]
        idx = torch.randperm(B)
        neg_joint = torch.cat([z, probs[idx]], dim=1)
        t_neg = self.net(neg_joint)           # [B,1]
        E_pos = t_pos.mean()
        E_neg = torch.log(torch.exp(t_neg).mean() + 1e-8)
        mi_lb = E_pos - E_neg
        return mi_lb, t_pos, t_neg

class InformationBottleneck(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(InformationBottleneck, self).__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar