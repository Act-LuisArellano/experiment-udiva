FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG USER_NAME=jramirez
ARG USER_ID=11056
ARG GROUP_NAME=guest
ARG GROUP_ID=11000

ARG DEBIAN_FRONTEND=noninteractive TZ=Etc/Madrid

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git htop curl wget ffmpeg libsm6 libxext6 \
    software-properties-common numactl pciutils \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
       python3.13 python3.13-venv python3.13-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.13 /usr/bin/python \
 && ln -sf /usr/bin/python3.13 /usr/bin/python3

# Install uv system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/ \
 && mv /root/.local/bin/uvx /usr/local/bin/

RUN groupadd --gid $GROUP_ID $GROUP_NAME \
 && useradd --uid $USER_ID --gid $GROUP_ID --shell /bin/bash --create-home $USER_NAME

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN usermod -aG sudo $USER_NAME

# Install deps into /opt/venv as root (so /opt is writable), then hand off to user
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python python3.13 \
 && UV_LINK_MODE=copy uv pip install \
    --python /opt/venv/bin/python \
    -r /tmp/requirements.txt \
 && chown -R $USER_ID:$GROUP_ID /opt/venv \
 && rm -rf /tmp/requirements.txt /root/.cache/uv

USER $USER_NAME

ENV PATH="/home/$USER_NAME/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# /workspace is the mount point for the experiment-udiva repo root
# Relative paths in configs (../data/, ../data-slow/) all resolve correctly
# from /workspace/code.
WORKDIR /workspace/code

# Activate venv explicitly on every shell entry so PATH is always correct
RUN echo 'source /opt/venv/bin/activate' >> /home/$USER_NAME/.bashrc

CMD ["/bin/bash"]
