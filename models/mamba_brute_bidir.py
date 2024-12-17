from mamba_ssm import Mamba
from dataset import sign_to_bin
from configuration import Config
import torch.nn.functional as F
from torch.nn import ModuleList, LayerNorm
import torch
import copy

device = "cuda"

def clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def build_mask(code):
    mask_size = code.n + code.pc_matrix.size(0)
    mask = torch.eye(mask_size, mask_size)
    for ii in range(code.pc_matrix.size(0)):
        idx = torch.where(code.pc_matrix[ii] > 0)[0]
        for jj in idx:
            for kk in idx:
                if jj != kk:
                    mask[jj, kk] += 1
                    mask[kk, jj] += 1
                    mask[code.n + ii, jj] += 1
                    mask[jj, code.n + ii] += 1
    src_mask = (mask > 0)
    return src_mask

class EncoderLayer(torch.nn.Module):
    def __init__(self, config: Config, length) -> None:
        super().__init__()
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state
        )
        self.norm = LayerNorm((length, config.d_model))
        self.n = config.code.n
        self.pc_checks = config.code.pc_matrix.size(0)
        self.register_buffer('pc_mask', build_mask(config.code))
        self.in_resize = torch.nn.Linear(config.d_model, self.n+self.pc_checks)
        self.in_reset_size = torch.nn.Linear(self.n + self.pc_checks, config.d_model)
        self.out_resize = torch.nn.Linear(config.d_model, self.n+self.pc_checks)
        self.out_reset_size = torch.nn.Linear(self.n + self.pc_checks, config.d_model)
    
    def forward(self, x):
        h = self.in_resize(x)
        h = h * self.pc_mask[0,0]
        h = self.in_reset_size(h)
        h *= x
        o1 = self.mamba.forward(h)
        o2 = torch.flip(self.mamba.forward(torch.flip(h,[1])),[1])
        o = o1+o2
        o = self.out_resize(o)
        o = o * self.pc_mask
        o = self.out_reset_size(o)
        o *= x
        return self.norm(F.tanh(x))

class ECCM(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n = config.code.n
        self.pc_checks = config.code.pc_matrix.size(0)
        self.src_embed = torch.nn.Parameter(torch.ones(
            (self.n + self.pc_checks, config.d_model)))
        self.resize_output_dim = torch.nn.Linear(config.d_model, 1)
        self.resize_output_length = torch.nn.Linear(self.n + self.pc_checks, self.n)
        self.norm_output = LayerNorm((self.n,))
        
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        
        self.mamba: ModuleList = clones(EncoderLayer(config, (self.n + self.pc_checks)), config.N_dec)
    
    def forward(self, magnitude, syndrome):
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        out: torch.Tensor = self.src_embed.unsqueeze(0) * emb
        for sublayer in self.mamba:
            out: torch.Tensor = sublayer.forward(out) # self.n+self.syndrom_length, d_model
        
        out: torch.Tensor = self.resize_output_length(out.swapaxes(-2,-1))
        out: torch.Tensor = self.resize_output_dim(out.swapaxes(-2,-1))
        out: torch.Tensor = out.squeeze(-1)
        return self.norm_output(F.tanh(out))

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy_with_logits(
            z_pred, sign_to_bin(torch.sign(z2)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred