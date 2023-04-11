from utils.helpers import DefaultArgParser

if __name__ == "__main__":
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_path()
    print(parser.parse())


def run_pytorch(path_to_model, batch_size, num_runs, timeout):
    from utils.pytorch import PyTorchRunnerV2
    from segment_anything.segment_anything import build_sam, SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=path_to_model))
    runner = PyTorchRunnerV2(mask_generator)

    # def run_single_pass(pytorch_runner, imagenet):
    #     masks = mask_generator.generate()
    #
    #     shape = (224, 224)
    #     output = pytorch_runner.run(torch.from_numpy(imagenet.get_input_array(shape)))
    #
    #     for i in range(batch_size):
    #         imagenet.submit_predictions(
    #             i,
    #             imagenet.extract_top1(output[i]),
    #             imagenet.extract_top5(output[i])
    #         )

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)
