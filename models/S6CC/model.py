
from dataset import sign_to_bin, bin_to_sign
import torch.nn.functional as F
from torch.nn import ModuleList
import torch
from configuration import Config
from models.S6CC.block import EncoderLayer

class ErrToNoise(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = torch.nn.Linear(2, 2)
        self.b = torch.nn.Linear(2, 1)
    
    def forward(self, x, out):
        y = torch.concatenate([x, out], -1)
        y = F.relu(self.a(y))
        return F.tanh(self.b(y))

class ECCM(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n = config.code.n
        self.seq_len = self.n
        self.d_model = config.code.pc_matrix.size(0)
        # self.src_embed = torch.nn.Parameter(torch.ones(
        #     (self.seq_len, self.d_model)))
        self.register_buffer("pc_matrix", config.code.pc_matrix)
        self.err_to_noise = ErrToNoise()
        self.resize_output_dim = torch.nn.Linear(self.d_model, 1)
        
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.mamba: ModuleList = ModuleList([EncoderLayer(config,) for _ in range(config.N_dec)])
        self.activation = F.tanh
    
    def forward(self, x, syndrome):
        """
        x: (B,L)
        syndrome: (B,P)
        """
        x = x.unsqueeze(-1)
        out = torch.zeros_like(x)
        for sublayer in self.mamba:
            xi = x*torch.sign(bin_to_sign(self.activation(out)))
            diff = sublayer.forward(xi)
            out = diff + out
        return self.activation(5*out)

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy(z_pred, z2)
        x_pred = sign_to_bin(torch.sign(bin_to_sign(z_pred)) * torch.sign(y))
        return loss, x_pred