ARG BASE_IMAGE_TAG=${BASE_IMAGE_TAG}

FROM nvidia/cuda:${BASE_IMAGE_TAG}

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends wget git libxrender1 libxext-dev && rm -rf /var/lib/apt/lists/*

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# create user
RUN useradd -m drugex-test-runner

ENV PATH /opt/conda/bin:$PATH

# copy entrypoint
COPY entrypoint.sh /entrypoint.sh

# copy test files

COPY --chown=drugex-test-runner:drugex-test-runner tests.sh /home/drugex-test-runner/tests.sh

WORKDIR /home/drugex-test-runner
