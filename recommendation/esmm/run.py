def single_pass_tf(tf_runner, dataset):
    dataset.submit_results(tf_runner.run(**dataset.get_inputs()))


def run_tf(model_path, batch_size, num_runs, timeout, dataset_path):
    import tensorflow as tf
    from utils.tf import TFSavedModelRunner
    from utils.benchmark import run_model
    from utils.recommendation.census_income import CensusIncome

    runner = TFSavedModelRunner()
    ds = CensusIncome(batch_size, dataset_path)
    runner.model = tf.saved_model.load(model_path).signatures["serving_default"]
    return run_model(single_pass_tf, runner, ds, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["tf"])
    parser.ask_for_batch_size(default_batch_size=2048)
    parser.require_model_path()
    parser.add_argument("--dataset_path",
                        type=str, required=True, help="path to csv file with 'Adult Census Income' data")
    run_tf(**vars(parser.parse()))
