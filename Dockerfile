FROM debian:bookworm

# add extra debian repos for proprietary packages
COPY debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt update

# install packages
RUN --mount=target=/tmp/packages.txt,source=packages.txt \
    xargs -r -a /tmp/packages.txt apt install -y

RUN rm -rf /var/lib/apt/lists/*
RUN apt clean

# merge platform-specific and common headers from the kernel directories. Run uname -r to get the version
ARG kernel=6.1.0-32

RUN --mount=target=/tmp/merge_headers.sh,source=merge_headers.sh \
    /tmp/merge_headers.sh /usr/src/linux-headers-$kernel-amd64 /usr/src/linux-headers-$kernel-common /usr/src/linux-headers-$kernel


# nvidia driver version. This should match the host version showed in nvidia-smi output
ARG version=550.144.03

ENV script=NVIDIA-Linux-x86_64-$version.run
RUN curl -o $script "https://us.download.nvidia.com/XFree86/Linux-x86_64/$version/$script" 
RUN chmod +x $script
RUN ./$script --silent --kernel-source-path /usr/src/linux-headers-$kernel


# create user
RUN useradd -m -s /bin/bash user

RUN groupadd -g 105 render

RUN usermod -aG video user
RUN usermod -aG render user

# next commands will be executed as the ucleanser
USER user
RUN python3 -m venv /home/user/venv
RUN echo "cd $HOME" >> /home/user/.bashrc
RUN echo ". /home/user/venv/bin/activate" >> /home/user/.bashrc

ENV PATH="/home/user/venv/bin:$PATH"

RUN --mount=target=/tmp/requirements.txt,source=requirements.txt \
    pip install --no-cache-dir -r /tmp/requirements.txt

# deploy application into target directory
COPY --chown=user . /home/user/app

WORKDIR /home/user/app
CMD [ "python", "sim_gradio.py" ]