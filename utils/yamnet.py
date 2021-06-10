import utils.misc as utils
import utils.dataset as utils_ds


class Yamnet(utils_ds.AudioData):

    def __init__(self, batch_size: int, sounds_path=None, pre_processing=None):

        if sounds_path is None:
            env_var = "SOUNDS_PATH"
            sounds_path = utils.get_env_variable(
                env_var, f"Path to ImageNet images directory has not been specified with {env_var} flag")

        self.batch_size = batch_size
        self.pre_processing = pre_processing

    def __get_path_to_audio(self):
        # TODO: implement this
        pass

    def __get_input_array(self):
        # TODO: implement this
        pass
