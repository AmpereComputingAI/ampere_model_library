import numpy as np
import json
import pathlib
import utils.misc as utils


class OutOfInstances(Exception):
    """
    An exception class being raised as an error in case of lack of further images to process by the pipeline.
    """
    pass


class Squad_v1_1:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size: int, sequence_size: int, tokenizer, target_seq_size,
                 dataset_path=None, labels_path=None):

        if dataset_path is None:
            env_var = "SQUAD_V1.1_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to Squad v1.1 .json file has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__seq_size = sequence_size
        self.__tokenizer = tokenizer
        self.__target_seq_size = target_seq_size
        self.__dataset = self.__verify_and_load(dataset_path)
        self.__example_iterator = self.__examples()

        self.__current_question = 0
        self.available_instances = self.__get_num_questions()
        self.__current_inputs = None
        self.__answers_submitted = False

    def __get_num_questions(self):
        total_questions = 0
        for section in self.__dataset:
            for paragraph in section["paragraphs"]:
                total_questions += len(paragraph["qas"])
        return total_questions

    def __verify_and_load(self, dataset_path, expected_version="1.1"):
        """
        A function parsing validation file for ImageNet 2012 validation dataset.

        .txt file consists of 50000 lines each holding data on a single image: its file name and 1 label with class best
        describing image's content

        :param labels_path: str, path to file containing image file names and labels
        :param is1001classes: bool, parameter setting whether the tested model has 1001 classes (+ background) or
        original 1000 classes
        :return: list of strings, list of ints
        """
        with open(dataset_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            encountered_version = dataset_json["version"]
            if encountered_version != expected_version:
                print_goodbye_message_and_die(
                    f"Expected SQUAD version {expected_version} but encountered version {encountered_version}")
            dataset = dataset_json["data"]
        return dataset

    def __get_path_to_img(self):
        """
        A function providing path to the ImageNet image.

        :return: pathlib.PurePath object containing path to the image
        """
        try:
            file_name = self.__file_names[self.__current_img]
        except IndexError:
            raise utils_ds.OutOfInstances("No more ImageNet images to process in the directory provided")
        self.__current_img += 1
        return pathlib.PurePath(self.__images_path, file_name)

    def __examples(self):
        for section in self.__dataset:
            for paragraph in section["paragraphs"]:
                for qas in paragraph["qas"]:
                    print("$$$$$$$$$$$$$$$$$$$$4")
                    print(paragraph["context"], qas["question"], qas["answers"])
                    yield paragraph["context"], qas["question"], qas["answers"]


    def __load_next_inputs_maybe(self):
        if self.__answers_submitted or self.__current_question == 0:
            contextes = list()
            questions = list()
            self.__valid_answers = list()
            for _ in range(self.__batch_size):
                print("a")
                context, question, correct_answers = next(self.__example_iterator)
                print("b")
                contextes.append(context)
                questions.append(question)
                self.__valid_answers.append(correct_answers)
            self.__current_inputs = self.__tokenizer(questions, contextes)

    def __get_input_array(self, input_name):
        self.__load_next_inputs_maybe()

        input_ids = self.__current_inputs[input_name]
        input_ids_padded = np.empty([self.__batch_size, self.__target_seq_size])

        for i in range(self.__batch_size):
            space_to_pad = self.__target_seq_size - input_ids[i].shape[0]
            input_ids_padded[i] = np.pad(input_ids[i], (0, space_to_pad), "constant", constant_values=0)

        return input_ids_padded

    def get_input_ids_array(self):
        return self.__get_input_array("input_ids")

    def get_attention_mask_array(self):
        return self.__get_input_array("attention_mask")

    def get_token_type_ids_array(self):
        return self.__get_input_array("token_type_ids")

    def extract_top1(self, output_array):
        """
        A helper function for extracting top-1 prediction from an output array holding soft-maxed data on 1 image.

        :param output_array: 1-D numpy array containing soft-maxed logits referring to 1 image
        :return: int, index of highest value in the supplied array
        """
        top_1_index = np.argmax(output_array)
        return top_1_index

    def extract_top5(self, output_array):
        """
        A helper function for extracting top-5 predictions from an output array holding soft-maxed data on 1 image.

        :param output_array: 1-D numpy array containing soft-maxed logits referring to 1 image
        :return: list of ints, list containing indices of 5 highest values in the supplied array
        """
        top_5_indices = np.argpartition(output_array, -5)[-5:]
        return top_5_indices

    def submit_predictions(self, id_in_batch: int, top_1_index: int, top_5_indices: list):
        """
        A function meant for submitting a class predictions for a given image.

        :param id_in_batch: int, id of an image in the currently processed batch that the provided predictions relate to
        :param top_1_index: int, index of a prediction with highest confidence
        :param top_5_indices: list of ints, indices of 5 predictions with highest confidence
        :return:
        """
        ground_truth = self.__labels[self.__current_img - self.__batch_size + id_in_batch]
        self.__top_1_count += int(ground_truth == top_1_index)
        self.__top_5_count += int(ground_truth in top_5_indices)

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
        """
        top_1_accuracy = self.__top_1_count / self.__current_img
        print("\n Top-1 accuracy = {:.3f}".format(top_1_accuracy))

        top_5_accuracy = self.__top_5_count / self.__current_img
        print(" Top-5 accuracy = {:.3f}".format(top_5_accuracy))

        print(f"\nAccuracy figures above calculated on the basis of {self.__current_img} images.")
        return {"top_1_acc": top_1_accuracy, "top_5_acc": top_5_accuracy}
