ARG PYTORCH_VERSION=2.6.0
ARG CUDA_VERSION=12.4 # should match CUDA driver version

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    git

WORKDIR /workspace
RUN git clone -b main https://github.com/google-deepmind/mujoco_menagerie

COPY environment.yml environment.yml
RUN conda env create


FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn9-devel

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    sudo \
    git \
    wget \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libxrender1 \
    libx11-6 \
    mesa-vulkan-drivers \
    vulkan-tools

RUN useradd -m -u 1000 user

RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN groupadd -g 105 render

RUN usermod -aG video user
RUN usermod -aG render user

USER user

WORKDIR /app
COPY --chown=user sim_gradio.py .
COPY --chown=user slobot ./slobot

COPY --chown=user --from=builder /workspace/mujoco_menagerie/trs_so_arm100 /app/trs_so_arm100

ARG CONDA_ENV_NAME=slobot # should match the env name in environment.yml

ENV CONDA_ENV_PATH=/opt/conda/envs/${CONDA_ENV_NAME}

# Copy the Conda environment from the builder stage
COPY --from=builder ${CONDA_ENV_PATH} ${CONDA_ENV_PATH}

RUN echo "source activate ${CONDA_ENV_NAME}" > ~/.bashrc

ENV PATH=${CONDA_ENV_PATH}/bin:$PATH

EXPOSE 7860

# ENTRYPOINT ["/bin/bash"]

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

CMD ["python", "sim_gradio.py"]