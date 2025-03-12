# Select Debian version
FROM debian:bookworm AS builder

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# install packages
RUN apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN git clone -b main https://github.com/google-deepmind/mujoco_menagerie

RUN python3 -m venv /venv

# Copy the list of dependencies
WORKDIR /app
COPY requirements.txt .

RUN /venv/bin/pip install --no-cache-dir -r requirements.txt


FROM debian:bookworm

# avoid language keyboard configuration to prompt a confirmation from the user
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN echo "deb http://deb.debian.org/debian/ bookworm main contrib non-free non-free-firmware" >> /etc/apt/sources.list

RUN apt-get update

RUN apt-get install -y \
    sudo \
    git \
    wget \
    linux-headers-cloud-amd64 \
    nvidia-driver \
    firmware-misc-nonfree \
    vulkan-tools \
    mesa-utils \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libxrender1 \
    libgl1 \
    libegl1 \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN groupadd -g 105 render

RUN usermod -aG video user
RUN usermod -aG render user

COPY --chown=user --from=builder /venv /venv

# Copy the robot configuration
COPY --chown=user --from=builder /mujoco_menagerie/trs_so_arm100 /app/trs_so_arm100

# Copy the application code
WORKDIR /app
COPY --chown=user sim_gradio.py .
COPY --chown=user slobot ./slobot

USER user

ENV HOME=/home/user \
    PATH="/venv/bin:$PATH"

EXPOSE 7860

#ENTRYPOINT ["/bin/bash"]
CMD ["python", "sim_gradio.py"]
