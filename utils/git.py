import git

GIT_PATH = '/zfsdata/data/sohvi/DrugEx'

def commit_hash():
    repo = git.Repo.init(GIT_PATH)
    return '#' + repo.head.object.hexsha[:8] 

