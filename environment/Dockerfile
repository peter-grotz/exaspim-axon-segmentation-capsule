# hash:sha256:07f4d66346144ee74c111cd5ee6e3458a49a39fc18d67f87d55122a8831421e6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/cuda12-base-uv:main-5692e50

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
