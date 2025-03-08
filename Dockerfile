# Use the latest Debian Bookworm image as the base image
FROM debian:bookworm AS builder

# Set environment variables to ensure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3.11-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN git clone -b main https://github.com/google-deepmind/mujoco_menagerie

RUN python3 -m venv /venv

# Copy the list of dependencies
WORKDIR /app
COPY requirements.txt .

RUN /venv/bin/pip install --no-cache-dir -r requirements.txt


FROM debian:bookworm

RUN apt-get update && apt-get install -y \
    python3 \
    libglib2.0-0 \
    libxrender1 \
    libgl1 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv

# Copy the robot configuration
COPY --from=builder /mujoco_menagerie/trs_so_arm100 /app/trs_so_arm100

# Copy the application code
WORKDIR /app
COPY sim_gradio.py .
COPY slobot ./slobot

EXPOSE 7860

ENV PATH="/venv/bin:$PATH"

#ENTRYPOINT ["/bin/bash"]
CMD ["python", "sim_gradio.py"]
