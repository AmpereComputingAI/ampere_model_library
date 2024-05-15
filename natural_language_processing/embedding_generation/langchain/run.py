# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
try:
    from utils import misc  # noqa
except ModuleNotFoundError:
    import os
    import sys
    filename = "set_env_variables.sh"
    directory = os.path.realpath(__file__).split("/")[:-1]
    for idx in range(1, len(directory) - 1):
        subdir = "/".join(directory[:-idx])
        if filename in os.listdir(subdir):
            print(f"\nPlease run \033[91m'source {os.path.join(subdir, filename)}'\033[0m first.")
            break
    else:
        print(f"\n\033[91mFAIL: Couldn't find {filename}, are you running this script as part of Ampere Model Library?"
              f"\033[0m")
    sys.exit(1)

def run_pytorch_fp32(model_name, num_runs, timeout, filepath, **kwargs):
    import os
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2, apply_compile
    from utils.helpers import DummyDataset

    embedding_model = HuggingFaceEmbeddings(model_name=model_name, show_progress=False)
    embedding_model.client.eval()
    embedding_model.client.forward = apply_compile(embedding_model.client.forward)

    documents = TextLoader(filepath).load_and_split(CharacterTextSplitter())

    def single_pass_pytorch(_runner, _):
        _runner.run(os.path.getsize(filepath), documents)

    def embeddings_gen(_documents):
        return Chroma.from_documents(_documents, embedding_model)

    runner = PyTorchRunnerV2(embeddings_gen, throughput_only=True)
    return run_model(single_pass_pytorch, runner, DummyDataset(), 1, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name([
        "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"])
    parser.add_argument("--filepath", type=str, required=True, help="path to a .txt file")
    run_pytorch_fp32(**vars(parser.parse()))
