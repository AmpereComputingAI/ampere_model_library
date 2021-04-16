import os.path
import numpy as np
import cv2
import utils.misc as utils


def get_path(path, path_env_variable, error_message):
    """
    A function returning the paths to file with validation dataset labels or directory with ImageNet images.

    :return: labels_path: str
    # """
    if path is None:
        try:
            path = os.environ[path_env_variable]
        except KeyError:
            utils.print_goodbye_message_and_die(error_message)

    return path


class ImageNet:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size, is1001classes, channels, images_path, labels_path):

        """
        A function for initialization of the class

        :param batch_size: int, a batch size on which a model will run an inference,
        usual sizes are 1, 4, 16, 32, 64
        :param is1001classes: boolean, True if model has 10001 classes ( one extra class for background ) &
        False if it doesn't
        :param channels: str, specifies the order of color channels a model accepts, eg. "BGR", "RGB"
        :param images_path: str, specify a path of images folder
        :param labels_path: str, specify a path of file with validation labels
        """

        # paths
        self.labels_path = get_path(labels_path, 'LABELS_PATH', 'The path to labels was not defined!\n' 
                                                                'You can specify the path when running the script: '
                                                                "'-l path/to/labels' \n"
                                                                'alternatively you can set the environment variable: '
                                                                "'export IMAGES_PATH=/path/to/images'")
        self.images_path = get_path(images_path, 'IMAGES_PATH', 'The path to images was not defined!\n'
                                                                'You can specify the path when running the script: '
                                                                "'-i path/to/images' \n"
                                                                'alternatively you can set the environment variable: '
                                                                "'export IMAGES_PATH=/path/to/images'")

        # images
        self.channels = channels
        self.batch_size = batch_size

        # labels
        self.is1001classes = is1001classes
        self.labels, self.file_names = self.get_labels_iterator()

        self.labels_iterator = utils.batch(self.labels, batch_size)
        self.file_names_iterator = utils.batch(self.file_names, batch_size)

        self.isNotDone = True

        # Accuracy
        self.image_count = 0
        self.top_1 = 0
        self.top_5 = 0

    def get_input_tensor(self, input_shape, preprocess):
        """
        A function providing preprocess images in batches.

        :param input_shape: tuple, a shape of input image for the model, eg. (224, 224)
        :param preprocess: a function performing preprocessing
        :return: numpy array of images, eg. (1, 224, 224, 3)
        """
        final_batch = np.empty((0, 224, 224, 3))

        try:
            batch = self.file_names_iterator.__next__()
        except StopIteration:
            print('you have reached the end of the dataset')
            raise self.OutOfImageNetImages("you have reached the end of the dataset")

        if len(batch) == self.batch_size:
            for i in batch:
                # note: cv2 returns by default BGR
                img = cv2.imread(os.path.join(self.images_path, i))
                assert img is not None, 'looks like the image in the provided path does not exist!'

                img = img[:, :, [2, 1, 0]]

                resized_img = cv2.resize(img, input_shape)
                img_array_expanded_dims = np.expand_dims(resized_img, axis=0)
                preprocessed_img = preprocess(img_array_expanded_dims.astype("float32"))

                final_batch = np.append(final_batch, preprocessed_img, axis=0)

        else:
            print("can't form a batch from remaining images ... skipping the last images")
            raise self.OutOfImageNetImages()

        return final_batch

    def extract_top1(self, result):
        top_1_indices = np.argmax(result, axis=1)
        return top_1_indices

    def extract_top5(self, result):
        top_5_indices = np.argpartition(result, -5)[:, -5:]
        return top_5_indices

    def record_measurement(self, result, top1_func=None, top5_func=None):
        """
        A function recording measurements of each run inference.

        :param result: numpy array, containing the results of inference
        :param top1_func:
        :param top5_func:
        """

        label_array = np.array(next(self.labels_iterator))
        self.image_count += self.batch_size

        if top1_func is not None:
            top1 = top1_func(result)

            self.top_1 += np.count_nonzero(top1 == label_array)

        if top5_func is not None:
            top5 = top5_func(result)
            n = 0
            for i in label_array:
                if i in top5[n]:
                    self.top_5 += 1
                n += 1

    def print_accuracy(self):
        """
        A function printing accuracy obtained after running all batches of images.
        """
        top_1_accuracy = ((self.top_1 / self.image_count) * 100)
        print("top-1 accuracy: %.2f" % top_1_accuracy, "%")

        top_5_accuracy = ((self.top_5 / self.image_count) * 100)
        print("top-5 accuracy: %.2f" % top_5_accuracy, "%")

    def get_labels_iterator(self):
        """
        A function which creates an iterator of ground truth labels corresponding to each image.

        :return: iterator, returns labels iterator
        """
        try:
            file = open(self.labels_path, 'r')
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(e)
        else:
            lines = file.readlines()
            labels = []
            file_names = []
            for line in lines:
                label = int(line[28:])
                if self.is1001classes:
                    labels.append(label + 1)
                else:
                    labels.append(label)

                file_name = line[:28]
                file_names.append(file_name)

        return labels, file_names

    class OutOfImageNetImages(Exception):
        """
        An exception class being raised as an error in case of lack of further images to process by the pipeline.
        """
        pass
