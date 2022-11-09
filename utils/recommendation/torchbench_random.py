from utils.recommendation.criteo import append_dlrm_to_pypath

class RandomDataset:
    def __init__(self, opt) -> None:
        append_dlrm_to_pypath()
        from utils.recommendation.dlrm.dlrm_data_pytorch import make_random_data_and_loader
        self.train_data, self.train_ld, _, _ = make_random_data_and_loader(opt, opt.ln_emb, opt.m_den)
        self.__predictions = []
        self.num_available_instances = 1
    
    def reset(self):
        return False

    def get_inputs(self):
        """
        A function returning input arrays for DLRM network.
        """
        pass

    def submit_predictions(self, prediction):
        self.__predictions.append(prediction)

    def summarize_accuracy(self):
        print("Random dataset used. Accuracy value is meaningless.")
        return self.__predictions
