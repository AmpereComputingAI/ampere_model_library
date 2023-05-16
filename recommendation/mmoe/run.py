def single_pass_tf(tf_runner, dataset):
    tf_runner.run(dataset.get_inputs())


def run_tf(batch_size, num_runs, timeout):
    import tensorflow as tf
    from utils.tf import TFSavedModelRunner
    from utils.benchmark import run_model
    from utils.recommendation.census_income import CensusIncome
    from utils.recommendation.DeepCTR.deepctr.models import MMOE

    runner = TFSavedModelRunner()
    ds = CensusIncome(batch_size)
    model = MMOE(ds.get_dnn_feature_columns(),
                 tower_dnn_hidden_units=[],
                 task_types=['binary', 'binary'],
                 task_names=['label_income', 'label_marital'])
    model.compile()
    runner.model = tf.function(model)
    return run_model(single_pass_tf, runner, ds, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["tf"], default_batch_size=256)
    run_tf(**vars(parser.parse()))
