# Stage 1: Build wheels for causal-conv1d and mamba
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder
ENV TORCH_CUDA_ARCH_LIST="6.1"

COPY causal-conv1d /opt/causal-conv1d
WORKDIR /opt/causal-conv1d
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE python setup.py bdist_wheel

COPY mamba /opt/mamba
WORKDIR /opt/mamba
RUN MAMBA_FORCE_BUILD=TRUE python setup.py bdist_wheel

RUN mkdir /out && cp /opt/causal-conv1d/dist/*.whl /opt/causal-conv1d/dist/*.whl /out
WORKDIR /out

# Stage 2: install wheels
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
ENV TORCH_CUDA_ARCH_LIST="6.1"

# Copy the wheels from the builder stage
COPY --from=builder /opt/causal-conv1d/dist/*.whl /wheels/
COPY --from=builder /opt/mamba/dist/*.whl /wheels/

# Install the wheels
RUN pip install /wheels/causal_conv1d-*.whl
RUN pip install /wheels/mamba_ssm-*.whl

# Clean up
RUN rm -rf /wheels

# Set the working directory
WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
