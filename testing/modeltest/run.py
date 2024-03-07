import importlib
import modeltest.settings as settings
from modeltest.settings import *
from modeltest.plot import plot

def print_info():
    print("Using DrugEx version: ", DRUGEX_VERSION)
    print("Experiment ID: ", EXPERIMENT_ID)
    print("Settings:")
    # print variables in this module
    for var in [x for x in dir(settings) if not x.startswith('__') and x.isupper()]:
        print(f"\t{var}: {eval(var)}")

if __name__ == '__main__':
    print_info()
    model = importlib.import_module(f"modeltest.{MODEL}.run")
    model.test()
    plot()
