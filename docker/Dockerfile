FROM mambaorg/micromamba:1.5.10-jammy

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libgbm1 \
    libglib2.0-0 \
    libgl1 \
    libgtk-3-0 \
    libice6 \
    libnss3 \
    libsm6 \
    libvulkan1 \
    libx11-6 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxkbcommon0 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    mesa-utils \
    procps \
    unzip \
    wget \
    x11-utils \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yml /tmp/environment.yml

RUN micromamba create -y -f /tmp/environment.yml \
    && micromamba clean --all --yes

ENV CONDA_DEFAULT_ENV=ai2thor-objectnav
ENV PATH=/opt/conda/envs/ai2thor-objectnav/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AI2THOR_CACHE_DIR=/workspaces/CS175-AI2-THOR/.ai2thor-cache

USER $MAMBA_USER
WORKDIR /workspaces/CS175-AI2-THOR

CMD ["/bin/bash"]
