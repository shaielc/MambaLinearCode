def test_torch():
    import torch    

    print("Test Conv", end=" ")
    batch, length, dim = 2, 64, 16
    x = torch.randn(batch, length, dim).to("cuda")
    m  =torch.nn.Conv1d(length, dim, dim).to("cuda")

    m(x)
    print("Passed!")

def test_causal_conv1d():
    import torch
    import numpy as np
    try:
        from causal_conv1d import causal_conv1d_fn
    except:
        print("Causal Conv1d not installed.")
        return

    print("Test Causal", end=" ")

    x = torch.from_numpy(np.random.normal(0,1,size=(1,3,100)).astype(np.float32)).to("cuda")
    w = torch.from_numpy(np.random.normal(0,1,size=(3,2)).astype(np.float32)).to("cuda")
    causal_conv1d_fn(x, w)
    print("Passed!")

def test_mamba():
    import torch
    from mamba_ssm import Mamba
    
    print("Test Mamba", end=" ")
    batch, length, dim = 2, 64, 16
    x = torch.randn(batch, length, dim).to("cuda")
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")
    y = model(x)
    assert y.shape == x.shape
    print("Passed!")

test_torch()
test_causal_conv1d()
test_mamba()
