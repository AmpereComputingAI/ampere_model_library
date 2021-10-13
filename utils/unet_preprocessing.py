import numpy as np




def prepare_one_hot(my_array, num_classes):
    """
    Reinterprets my_array into one-hot encoded, for classes as many as num_classes
    """
    res = np.eye(num_classes)[np.array(my_array).reshape(-1)]
    return res.reshape(list(my_array.shape) + [num_classes])


def to_one_hot(my_array, channel_axis):
    """
    Changes class information into one-hot encoded information
    Number of classes in KiTS19 is 3: background, kidney segmentation, tumor segmentation
    As a result, 1 channel of class info turns into 3 channels of one-hot info
    """
    print(my_array)
    my_array = prepare_one_hot(my_array, num_classes=3)
    print(my_array)
    my_array = np.transpose(my_array, (0, 4, 1, 2, 3)).astype(np.float64)
    return my_array


def get_dice_score(case, prediction, target):
    """
    Calculates DICE score of prediction against target, for classes as many as case
    One-hot encoded form of case/prediction used for easier handling
    Background case is not important and hence removed
    """
    # constants
    channel_axis = 1
    reduce_axis = (2, 3, 4)
    smooth_nr = 1e-6
    smooth_dr = 1e-6

    # apply one-hot
    prediction = to_one_hot(prediction, channel_axis)
    target = to_one_hot(target, channel_axis)

    # remove background
    target = target[:, 1:]
    prediction = prediction[:, 1:]

    # calculate dice score
    assert target.shape == prediction.shape, \
        f"Different shape -- target: {target.shape}, prediction: {prediction.shape}"
    assert target.dtype == np.float64 and prediction.dtype == np.float64, \
        f"Unexpected dtype -- target: {target.dtype}, prediction: {prediction.dtype}"

    # intersection for numerator; target/prediction sum for denominator
    # easy b/c one-hot encoded format
    intersection = np.sum(target * prediction, axis=reduce_axis)
    target_sum = np.sum(target, axis=reduce_axis)
    prediction_sum = np.sum(prediction, axis=reduce_axis)

    # get DICE score for each class
    dice_val = (2.0 * intersection + smooth_nr) / \
               (target_sum + prediction_sum + smooth_dr)

    print(dice_val)

    # return after removing batch dim
    return (case, dice_val[0])