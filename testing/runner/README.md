# DrugEx Test Runner

Run unit and other tests for DrugEx in GPU-enabled Docker containers.

```bash
# define important variables
export NVIDIA_VISIBLE_DEVICES=0
export DRUGEX_REPO="https://<username>:<access-token>@your_hosting_service.com/DrugEx.git"
export DRUGEX_REVISION="master" # can be branch, commit ID or a tag
export QSPRPRED_REPO="https://<username>:<access-token>@your_hosting_service.com/QSPRPred.git"
export QSPRPRED_REVISION="main" # can be branch, commit ID or a tag

# spawn a runner with the given settings
./runner.sh
```

A few tips:

- Use any repo URLs format that git will understand, but remember the docker image will not have access to your ssh keys by default.
- You can customize the [`tests.sh`](./tests.sh) script to change what tests to run or do other stuff inside the runner container.
- When you run the `runner.sh` script logs are saved to the `logs` folder in the current directory.
- It is possible to permanently set the environment variables in a [`.env`](./.env) file that you can save in this directory. In this case, it would have the following contents:
 
    ```bash
    NVIDIA_VISIBLE_DEVICES=0
    DRUGEX_REPO=https://<gitlab-username>:<access-token>@<gitlab-repo-url>
    DRUGEX_REVISION=master
    QSPRPRED_REPO=https://<gitlab-username>:<access-token>@<gitlab-repo-url>
    QSPRPRED_REVISION=main
    ```



