FROM debian:trixie

# add extra debian repos for proprietary packages
COPY debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt update

# install packages
RUN --mount=target=/tmp/packages.txt,source=packages.txt \
    xargs -r -a /tmp/packages.txt apt install -y

RUN rm -rf /var/lib/apt/lists/*
RUN apt clean

# merge platform-specific and common headers from the kernel directories. Run uname -r to get the version
ARG kernel=6.12.20

RUN --mount=target=/tmp/merge_headers.sh,source=merge_headers.sh \
    /tmp/merge_headers.sh /usr/src/linux-headers-$kernel-cloud-amd64 /usr/src/linux-headers-$kernel-common /usr/src/linux-headers-$kernel


# nvidia driver version. This should match the host version showed in nvidia-smi output
ARG version=550.144.03

ENV script=NVIDIA-Linux-x86_64-$version.run
RUN curl -o $script "https://us.download.nvidia.com/XFree86/Linux-x86_64/$version/$script" 
RUN chmod +x $script

# Skip loading kernel modules to avoid "Kernel module load error: Operation not permitted" error
RUN ./$script --silent --kernel-source-path /usr/src/linux-headers-$kernel --skip-module-load \
    || cat /var/log/nvidia-installer.log

# create user
RUN useradd -m -s /bin/bash user

RUN usermod -aG video user
RUN usermod -aG render user

# next commands will be executed as the user
USER user

ENV HOME=/home/user
WORKDIR $HOME

# python version
ARG python_version=3.12.9

RUN curl https://pyenv.run | bash
ENV PATH="$HOME/.pyenv/bin:$PATH"
RUN pyenv install $python_version
ENV PATH="$HOME/.pyenv/versions/$python_version/bin:$PATH"

RUN python -m venv /home/user/venv
RUN echo "cd $HOME" >> /home/user/.bashrc
RUN echo ". /home/user/venv/bin/activate" >> /home/user/.bashrc

ENV PATH="/home/user/venv/bin:$PATH"

RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/user/app

# deploy robot configuration
RUN git clone -b main https://github.com/google-deepmind/mujoco_menagerie
RUN cp -r mujoco_menagerie/trs_so_arm100 trs_so_arm100
RUN rm -rf mujoco_menagerie

# deploy application into target directory
COPY --chown=user . /home/user/app

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD [ "python", "sim_gradio_video.py" ]