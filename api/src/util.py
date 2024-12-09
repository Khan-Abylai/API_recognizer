import numpy as np
import cv2
from constants import DETECTION_IMAGE_H, DETECTION_IMAGE_W
import constants


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for
                      im in im_list]
    return cv2.hconcat(im_list_resize)


def prepare_for_detector(image_orig):
    h, w, _ = image_orig.shape

    if h > w:
        s = max(image_orig.shape[0:2])
        f = np.zeros((s, s, 3), np.uint8)
        ax, ay = (s - image_orig.shape[1]) // 2, (s - image_orig.shape[0]) // 2
        f[ay:image_orig.shape[0] + ay, ax:ax + image_orig.shape[1]] = image_orig
        resized_image = cv2.resize(f.copy(), (DETECTION_IMAGE_W, DETECTION_IMAGE_H))
    else:
        ax, ay = 0, 0
        resized_image = cv2.resize(image_orig, (DETECTION_IMAGE_W, DETECTION_IMAGE_H))

    resized_image = resized_image.transpose((2, 0, 1))
    resized_image = 2 * (resized_image / 255.0 - 0.5)
    x = resized_image.astype(np.float32)
    x = np.ascontiguousarray(x)

    return image_orig, x, ax, ay


def bbox_iou_np(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0,
                                                                               None)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def nms_np(predictions, conf_thres=0.2, nms_thres=0.2, include_conf=False):
    filter_mask = (predictions[:, -1] >= conf_thres)
    predictions = predictions[filter_mask]

    if len(predictions) == 0:
        return np.array([])

    output = []

    while len(predictions) > 0:
        max_index = np.argmax(predictions[:, -1])

        if include_conf:
            output.append(predictions[max_index])
        else:
            output.append(predictions[max_index, :-1])

        ious = bbox_iou_np(np.array([predictions[max_index, :-1]]), predictions[:, :-1], x1y1x2y2=False)

        predictions = predictions[ious < nms_thres]

    return np.stack(output)


def preprocess_image_recognizer(img, box):
    lt = box[2]
    lb = box[3]
    rt = box[4]
    rb = box[5]

    w = ((rt[0] - lt[0]) + (rb[0] - lb[0])) / 2
    h = ((lb[1] - lt[1]) + (rb[1] - rt[1])) / 2

    ratio = w / h
    print(f'plate size ratio: {w} / {h} = {ratio}')

    if ratio <= 2.6:
        plate_img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box[2:], constants.PLATE_SQUARE),
                                        (constants.RECOGNIZER_IMAGE_W // 2, constants.RECOGNIZER_IMAGE_H * 2))
        top = plate_img[:32, :]
        bottom = plate_img[32:, :]
        plate_img = hconcat_resize_min([top, bottom])
    else:
        plate_img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(box[2:], constants.PLATE_RECT),
                                        (constants.RECOGNIZER_IMAGE_W, constants.RECOGNIZER_IMAGE_H))

    # cv2.imwrite("./image.jpg", plate_img)
    return np.ascontiguousarray(np.stack([plate_img]).astype(np.float32).transpose(
        constants.RECOGNIZER_IMG_CONFIGURATION) / constants.PIXEL_MAX_VALUE)
