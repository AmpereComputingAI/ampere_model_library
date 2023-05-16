
def run_tf(model_path, batch_size, num_runs, timeout):
    import tensorflow as tf
    from tensorflow.python.saved_model import tag_constants
    from utils.tf import TFSavedModelRunner
    from utils.benchmark import run_model
    from utils.recommendation.census_income import CensusIncome
    from utils.recommendation.DeepCTR.deepctr.models import MMOE

    def run_single_pass(tf_runner, dataset):
        output = tf_runner.run(dataset.get_inputs())
        dfg
        print(output)
        sfd

    runner = TFSavedModelRunner()
    ds = CensusIncome(batch_size)
    #model = MMOE(ds.get_dnn_feature_columns(),
    #             tower_dnn_hidden_units=[],
    #             task_types=['binary', 'binary'],
    #             task_names=['label_income', 'label_marital'])
    #model.compile()
    #model.save("test")
    model = tf.keras.models.load_model("test")
    model = tf.function(model)
    print(model(CensusIncome(batch_size).get_inputs()))
    fds
    # print(runner.model)
    # dfs
    return run_model(run_single_pass, runner, ds, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["tf"])
    parser.require_model_path()
    run_tf(**vars(parser.parse()))
