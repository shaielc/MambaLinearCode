from dataset import sign_to_bin, bin_to_sign
from torch.nn import ModuleList
import copy
import torch

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
    src_mask = ~(mask > 0).unsqueeze(0).unsqueeze(0)
    return src_mask

def get_syndrome(x, pc_matrix):
    return bin_to_sign(torch.matmul(
        (pc_matrix).float(),
        sign_to_bin(torch.sign(x))
    ) % 2)
