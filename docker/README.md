# DrugEx Jupyter Container

This Docker container can be used to quickly run a DrugEx-enabled Jupyter Notebook server with GPU support (if available).

## Prerequisites
- Docker (with `docker-compose`) installed on Linux.
- NVIDIA Container Toolkit installed if you plan to use GPUs (so the container can access GPUs).

## Environment Setup

Create a `.env` file next to `docker-compose-app.yml`. Here are example environment variables you might want to in this file    :

```bash
NVIDIA_VISIBLE_DEVICES=0
BASE_IMAGE_TAG="13.1.0-runtime-ubuntu24.04"
PYTHON_VERSION="3.12"
DRUGEX_REPO="https://github.com/CDDLeiden/DrugEx.git"
DRUGEX_REVISION="dev"
QSPRPRED_REPO="https://github.com/CDDLeiden/QSPRpred.git"
QSPRPRED_REVISION="main"
CONTAINER_NAME="drugex-app-runner"
APP_PORT=8080
JUPYTER_PASSWORD="your_secure_password"
JUPYTER_TOKEN="your_secure_token"
```

## Running the Container

The image is built and the container is started using `docker-compose`. Run the following command in the directory containing `docker-compose-app.yml`:

```bash
docker-compose -f docker-compose-app.yml up # assumes environment variables are set in a .env file
```
