import torch.nn.functional as F
from torch.nn import LayerNorm
from configuration import Config
import torch
from .utils import get_syndrome

class EncoderLayer(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.seq_len = config.code.n
        self.d_model = config.code.pc_matrix.shape[0]
        self.Sa = torch.nn.Linear(self.d_model, 1, bias=False)
        self.Sb = torch.nn.Linear(self.d_model, 1, bias=False)
        self.Sc = torch.nn.Linear(self.d_model, 1, bias=False)
        self.resize_output = torch.nn.Linear(self.d_model,1)
        self.norm_output = LayerNorm((self.seq_len,1))
        self.activation = F.softmax
        self.register_buffer("pc_matrix", config.code.pc_matrix.float())
        self.register_buffer("syndrome_connection", torch.eye(self.d_model))
        
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        
        # # For tests:
        # Azer = torch.zeros_like(self.Sa.weight)
        # Bzer = torch.zeros_like(self.Sb.weight)
        # Czer = torch.zeros_like(self.Sc.weight)
        # rw_zer = torch.ones_like(self.resize_output.weight)
        # rb_zer = torch.zeros_like(self.resize_output.bias)
        # with torch.no_grad():
        #     self.Sa.weight.copy_(Azer)
        #     self.Sb.weight.copy_(Bzer)
        #     self.Sc.weight.copy_(Czer)
        #     self.resize_output.weight.copy_(rw_zer)
        #     self.resize_output.bias.copy_(rb_zer)
        
        # Azer = torch.randn_like(self.Sa.weight)
        # Bzer = torch.randn_like(self.Sb.weight)
        # Czer = torch.randn_like(self.Sc.weight)
        # with torch.no_grad():
        #     self.Sa.weight.copy_(Azer)
        #     self.Sb.weight.copy_(Bzer)
        #     self.Sc.weight.copy_(Czer)

        

    def _retrieve(self, y, states):
        out = torch.zeros_like(y)
        out[...,0,:] = y[...,0,:]
        for i in range(1,self.seq_len):
            out[...,i,:] = y[...,i,:] * states[...,i-1,:]
        out = out.sum(-1,keepdim=True)
        return -out
    
    def _ssm_iteration(self, X,A,B,C,h,y,states, pc_matrix):
        pi = B * pc_matrix
        A_ij = - A * pc_matrix * h
        h = h + A_ij + pi # The default assumption is no error therefore we should stay at the same state.
        y = C * (h @ pc_matrix).unsqueeze(-1)
        return h,h,y
    
    def _ssm_calc(self, X, pc_matrix, h):
        """
        X: (B,L,p)
        pc_matrix: (p,L)
        h: (B,p)

        where:
        B - batch
        L - Length
        p - PC check count
        """
        y = torch.zeros_like(X)
        states = torch.zeros_like(X)
        has_errors = torch.any(h != 1,dim=-1,keepdim=True).unsqueeze(-1) # (B,1,1)
        # Taking a viterbi-like approach
        # A: Transition matrix proportional to <s_i, h>, if h and s_i don't belong to the same lane this would be negative.
        # B: Initial probabilty matrix, p(e_i = true) = f(|x|,syn@pc[:,i])
        # C: Emmision matrix, p(e_i=true| s_i).
        A = 1.0 # self.Sa(X)
        Bbias = (h @ pc_matrix).unsqueeze(-1)
        Binf = self.Sb(torch.abs(X))
        B = Bbias/torch.abs(Bbias).max() + \
            F.tanh(Binf)
            # h0 = syndrome
        C = 0.5 + F.tanh(self.Sc(torch.abs(X)))
        
        for i in range(self.seq_len):
            h, states[..., i, :], y[..., i, :] = self._ssm_iteration(
                X[..., i, :],
                A,
                B[...,i,:],
                C[...,i,:],
                h,
                y[..., i, :],
                states[..., i, :],
                self.pc_matrix[:,i]
            )
        
        return y* has_errors, B, C , states

    def _forward_pass(self, X, h=None):
        return self._ssm_calc(X,self.pc_matrix, h)
    
    def _reverse_pass(self,X,h=None):# -> tuple[Tensor, Any, Any, float | Any]:
        r = self._ssm_calc(torch.flip(X,[1]),torch.flip(self.pc_matrix,[1]), h)
        return tuple(torch.flip(M,[1]) for M in r)
    
    def _bider_pass(self, X, h=None):
        y1,B1,C1,h1 = self._forward_pass(X,h)
        y2,B2,C2,h2 = self._reverse_pass(X,h)
        return y1, y2, B1, C1, h1
    
    def embed(self, X):
        X = X * self.pc_matrix.T
        return X
        
    def forward(self, X, h=None):
        if h is None:
            h = get_syndrome(X, self.pc_matrix).squeeze(-1).detach()
        X = self.embed(X)
        y1,y2,*rest = self._bider_pass(X,h)
        out = y1+y2
        out_resized = self.resize_output(out)
        out_activated = self.activation(out_resized,dim=-2)
        return out_activated