# DrugEx Test Runner

Run unit and other tests for DrugEx in GPU-enabled Docker containers.

```bash
# define important variables
export NVIDIA_VISIBLE_DEVICES=0 # ids of the GPUs to use
export BASE_IMAGE_TAG="12.0.1-cudnn8-runtime-ubuntu22.04" # cuda base image tag, translated to nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
export PYTHON_VERSION="3.10" # python version for the conda base environment
export DRUGEX_REPO="https://<username>:<access-token>@your_hosting_service.com/DrugEx.git"
export DRUGEX_REVISION="master" # can be branch, commit ID or a tag
export QSPRPRED_REPO="https://<username>:<access-token>@your_hosting_service.com/QSPRPred.git"
export QSPRPRED_REVISION="main" # can be branch, commit ID or a tag

# spawn a runner with the given settings
./runner.sh # make sure you are in the 'docker' group so that you have correct permissions
```

A few tips:

- Use any repo URLs format that git will understand, but remember the docker image will not have access to your ssh keys by default.
- You can customize the [`tests.sh`](./tests.sh) script to change what tests to run or do other stuff inside the runner container.
- When you run the `runner.sh` script logs are saved to the `logs` folder in the current directory.
- It is possible to permanently set the environment variables in a `.env` file that you can save in this directory. In this case, it would have the following contents:
 
    ```bash
    NVIDIA_VISIBLE_DEVICES=0
    DRUGEX_REPO=https://<gitlab-username>:<access-token>@<gitlab-repo-url>
    DRUGEX_REVISION=master
    QSPRPRED_REPO=https://<gitlab-username>:<access-token>@<gitlab-repo-url>
    QSPRPRED_REVISION=main
    ```



