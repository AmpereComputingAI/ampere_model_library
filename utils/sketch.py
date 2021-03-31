import numpy as np


truth = [{'bbox': [376.09, 326.5339, 394.67, 387.84268], 'category_id': 44},
         {'bbox': [344.44, 249.50629, 353.54, 308.33807], 'category_id': 44},
         {'bbox': [315.67, 271.16986, 328.43, 315.595], 'category_id': 44},
         {'bbox': [489.5, 367.9866, 561.34, 424.20752], 'category_id': 51},
         {'bbox': [504.3, 301.81757, 524.01, 315.32718], 'category_id': 51},
         {'bbox': [485.03, 290.28955, 504.81, 304.6025], 'category_id': 51},
         {'bbox': [488.4, 263.1364, 510.51, 281.2251], 'category_id': 51},
         {'bbox': [351.21, 389.74393, 495.86, 443.98328], 'category_id': 81},
         {'bbox': [326.96, 267.86276, 340.53, 310.23935], 'category_id': 44},
         {'bbox': [352.29, 246.74811, 359.92, 306.91882], 'category_id': 44},
         {'bbox': [504.86, 316.53223, 524.71, 326.69455], 'category_id': 51},
         {'bbox': [329.8, 352.6025, 347.95, 418.5975], 'category_id': 44},
         {'bbox': [0.0, 470.57404, 66.72, 629.436], 'category_id': 70},
         {'bbox': [356.18, 265.14478, 374.52, 310.18576], 'category_id': 44},
         {'bbox': [543.39, 316.74646, 562.59, 350.08536], 'category_id': 44},
         {'bbox': [415.02, 277.0477, 434.14, 306.91882], 'category_id': 44},
         {'bbox': [391.51, 356.4452, 405.48, 389.2753], 'category_id': 44},
         {'bbox': [295.0, 262.4268, 557.0, 443.1799], 'category_id': 44}]

pred = [{'bbox': [376.09, 326.5339, 394.67, 387.84268], 'category_id': 44},
        {'bbox': [344.44, 249.50629, 353.54, 308.33807], 'category_id': 44},
        {'bbox': [315.67, 271.16986, 328.43, 315.595], 'category_id': 44},
        {'bbox': [489.5, 367.9866, 561.34, 424.20752], 'category_id': 51},
        {'bbox': [504.3, 301.81757, 524.01, 315.32718], 'category_id': 51},
        {'bbox': [485.03, 290.28955, 504.81, 304.6025], 'category_id': 51},
        {'bbox': [488.4, 263.1364, 510.51, 281.2251], 'category_id': 51},
        {'bbox': [351.21, 389.74393, 495.86, 443.98328], 'category_id': 81},
        {'bbox': [326.96, 267.86276, 340.53, 310.23935], 'category_id': 44},
        {'bbox': [352.29, 246.74811, 359.92, 306.91882], 'category_id': 44},
        {'bbox': [504.86, 316.53223, 524.71, 326.69455], 'category_id': 51},
        {'bbox': [329.8, 352.6025, 347.95, 418.5975], 'category_id': 44},
        {'bbox': [0.0, 470.57404, 66.72, 629.436], 'category_id': 70},
        {'bbox': [356.18, 265.14478, 374.52, 310.18576], 'category_id': 44},
        {'bbox': [543.39, 316.74646, 562.59, 350.08536], 'category_id': 44},
        {'bbox': [415.02, 277.0477, 434.14, 306.91882], 'category_id': 44},
        {'bbox': [391.51, 356.4452, 405.48, 389.2753], 'category_id': 44},
        {'bbox': [295.0, 262.4268, 557.0, 443.1799], 'category_id': 44}]

truth = [{'bbox': [1, 1, 3, 3], 'category_id': 11},
         {'bbox': [5, 5, 9, 9], 'category_id': 11},
         {'bbox': [6, 3, 10, 6], 'category_id': 11},
         {'bbox': [8, 1, 11, 4], 'category_id': 11},
         {'bbox': [8, 8, 11, 11], 'category_id': 11}]

pred = [{'bbox': [1, 6, 4, 9], 'category_id': 11},
        {'bbox': [7, 7, 10, 10], 'category_id': 11},
        {'bbox': [6, 2, 10, 7], 'category_id': 11},
        {'bbox': [8, 5, 12, 9], 'category_id': 11}]


def unpack_bbox_into_vars(bbox: list):
    return bbox[0], bbox[1], bbox[2], bbox[3]


def calc_overlapping_part_of_two_line_segments(a_0, b_0, a_1, b_1):
    # a denotes smaller of values a and b marking end points of given (_i) line segment
    assert a_0 <= b_0 and a_1 <= b_1, "a value cannot be bigger than b"
    return min(b_0, b_1) - max(a_0, a_1)


def calc_iou(bbox_0: list, bbox_1: list):
    left_0, top_0, right_0, bottom_0 = unpack_bbox_into_vars(bbox_0)
    left_1, top_1, right_1, bottom_1 = unpack_bbox_into_vars(bbox_1)
    if right_0 <= left_1:
        return 0.0
    if right_1 <= left_0:
        return 0.0
    if bottom_0 <= top_1:
        return 0.0
    if bottom_1 <= top_0:
        return 0.0
    horizontal_common_length = calc_overlapping_part_of_two_line_segments(left_0, right_0, left_1, right_1)
    vertical_common_length = calc_overlapping_part_of_two_line_segments(top_0, bottom_0, top_1, bottom_1)
    intersection_area = horizontal_common_length * vertical_common_length
    area_of_bbox_0 = (right_0 - left_0) * (bottom_0 - top_0)
    area_of_bbox_1 = (right_1 - left_1) * (bottom_1 - top_1)
    union_area = area_of_bbox_0 + area_of_bbox_1 - intersection_area
    assert union_area != 0.0, "area of union cannot be equal to 0"
    return intersection_area / union_area


def calculate_ious(truth, pred):
    # this will contain every iou combination between genuine bboxes (dim 0) and predicted (dim 1)
    #iou_matrix = np.full((len(truth), len(pred)), float(-1))
    iou_matrix = np.zeros((len(truth), len(pred)))
    for true_instance in truth:
        for prediction in pred:
            if true_instance["category_id"] != prediction["category_id"]:
                continue
            iou_matrix[truth.index(true_instance)][pred.index(prediction)] = calc_iou(true_instance["bbox"], prediction["bbox"])
    print(iou_matrix)
    return iou_matrix


class MatchRegistry:
    # registry where reference bboxes are owners of matching predicted bboxes
    def __init__(self, pred_bboxes_num):
        self.__registry = list()
        self.__pred_ownership_array = np.full(pred_bboxes_num, -1)

    def register(self, ref_bbox_item):
        self.registry.append(ref_bbox_item)

    def __resolve_conflict(self, owner, plaintiff):
        

    def summarize(self):
        false_negatives = 0
        for ref_bbox in self.registry:
            if ref_bbox:
                if self.__pred_has_owner(ref_bbox.get_best_pred_id()):
                    owner = self.__registry[self.__get_owner_id(ref_bbox.get_best_pred_id())]
                    self.__resolve_conflict(owner, ref_bbox)
                else:
                    self.__pred_ownership_array[ref_bbox.get_best_pred_id()] = ref_bbox.id
            else:
                false_negatives += 1

    def __get_owner_id(self, pred_id):
        owner_id = self.__pred_ownership_array[pred_id]
        assert owner_id != -1, "this prediction does not have owner"
        return owner_id

    def __pred_has_owner(self, pred_id):
        if self.__pred_ownership_array[pred_id] != -1:
            return True
        return False

    def __try_to_register(self, ref_bbox_id, pred_bbox_id):
        if pred_bbox_id in self.array_with_matches:
            return False
        self.array_with_matches[ref_bbox_id] = pred_bbox_id
        return True

    def ref_bboxes_without_match_num(self):
        return sum(x is None for x in self.registry)


class RefBBox:
    def __init__(self, ref_bbox_id, array_with_IoUs, IoU_threshold):
        self.id = ref_bbox_id
        # sort IoUs in descending order
        self.array_with_IoUs = -np.sort(-array_with_IoUs)
        # create an array with indices in original IoUs matrix for values in array_with_IoUs
        self.array_with_ids = np.argsort(-array_with_IoUs)
        # remove IoUs below the threshold
        ids_to_delete = np.argwhere(self.array_with_IoUs < IoU_threshold).flatten()
        self.array_with_IoUs = np.delete(self.array_with_IoUs, ids_to_delete)
        self.array_with_ids = np.delete(self.array_with_ids, ids_to_delete)

    def get_best_pred_id(self):
        return self.array_with_ids[0]



def match_bboxes(truth, pred, IoU_threshold=0.1):
    false_negatives = 0
    matrix = calculate_ious(truth, pred)
    match_registry = MatchRegistry(len(pred))
    for ref_bbox_id in range(len(truth)):
        array_with_IoUs = matrix[ref_bbox_id]
        if np.max(array_with_IoUs) >= IoU_threshold:
            match_registry.register(RefBBox(ref_bbox_id, array_with_IoUs, IoU_threshold))
        else:
            match_registry.register(None)
    print(match_registry.ref_bboxes_without_match_num())
    for ref_bbox in ref_bboxes_list:
        if ref_bbox:
            if not match_registry.try_to_register(ref_bbox.id, ref_bbox.get_best_pred_id()):
                id_of_owner = match_registry.get_owner_id(ref_bbox.get_best_pred_id())
               # ref_bboxes_list[id_of_owner].
        else:
            false_negatives += 1




match_bboxes(truth, pred)