FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=vscode

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y build-essential --no-install-recommends make \
    ca-certificates \
    bash-completion \
    git \
    just \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Python and poetry installation
USER $USER
ARG HOME="/home/$USER"
ARG PYTHON_VERSION=3.10

ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:$PATH"

EXPOSE 8800

RUN echo "Configuring Python environment" \
    && curl https://pyenv.run | bash \
    && pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && pip install ruff semver toml
