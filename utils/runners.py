import utils.tf as utils_rn


class UnetRunner(utils_rn.TFSavedModelRunner):

    def __init__(self, path_to_model: str):
        super().__init__(path_to_model)
        self.output_name = list(self.model.structured_outputs)[0]

    def get_output_name(self):

        return list(self.model.structured_outputs)[0]
