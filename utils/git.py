import os
import git

def commit_hash(GIT_PATH):
    repo = git.Repo.init(GIT_PATH)
    return '#' + repo.head.object.hexsha[:8] 

