import os
from datetime import datetime

profile_path = None


def dls_profiler_enabled():
    if "DLS_PROFILER" in os.environ and os.environ["DLS_PROFILER"] == "1":
        return True
    else:
        return False


def set_profile_path(model_name):
    global profile_path
    profile_path = os.path.join(os.getcwd(), "{}_{:%Y_%m_%d_%H_%M_%S}".format(model_name, datetime.now()))


def get_profile_path():
    return profile_path


def summarize_tf_profiling():
    print(f"\nTo visualize TF profiler output run locally:\n tensorboard --logdir={get_profile_path()}")
