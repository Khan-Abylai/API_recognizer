import cv2
import numpy as np
from fastapi import FastAPI, Request
import time
from detection_service import DetectionEngine
from recognition_service import RecognitionEngine
from util import prepare_for_detector, nms_np, preprocess_image_recognizer
from constants import DETECTION_IMAGE_H, DETECTION_IMAGE_W

app = FastAPI()
detector = DetectionEngine()
recognizer = RecognitionEngine()


def getPlates(img_orig, img_model, ax, ay):
    # Проверка на RGB формат
    if img_orig.shape[2] != 3:
        raise ValueError("Ожидаются изображения в формате RGB с 3 каналами.")

    original_image_h, original_image_w, _ = img_orig.shape
    plate_output = detector.predict(img_model)
    plates = nms_np(plate_output[0], conf_thres=0.7, include_conf=True)

    results = []
    if len(plates) > 0:
        plates[..., [4, 6, 8, 10]] += plates[..., [0]]
        plates[..., [5, 7, 9, 11]] += plates[..., [1]]
        ind = np.argsort(plates[..., -1])
        plates = plates[ind]

        for plate in plates:
            box = np.copy(plate[:12]).reshape(6, 2)  # 6 точек для описания рамки
            box[:, ::2] *= (original_image_w + ax * 2) / DETECTION_IMAGE_W
            box[:, 1::2] *= (original_image_h + ay * 2) / DETECTION_IMAGE_H

            box[:, ::2] -= ax
            box[:, 1::2] -= ay

            # Обработка номера
            plate_img = preprocess_image_recognizer(img_orig, box)
            plate_labels, probs = recognizer.predict(plate_img)

            # Сохраняем результат
            results.append({
                "label": plate_labels,
                "prob": probs,
                "lp_coords": {
                    "center_x": str(box[0][0]), "center_y": str(box[0][1]), "plate_w": str(box[1][0]),
                        "plate_h": str(box[1][1]), "left_top_x": str(box[2][0]), "left_top_y": str(box[2][1]),
                        "left_bottom_x": str(box[3][0]), "left_bottom_y": str(box[3][1]), "right_top_x": str(box[4][0]),
                        "right_top_y": str(box[4][1]), "right_bottom_x": str(box[5][0]),
                        "right_bottom_y": str(box[5][1])
                }
            })

    return results


def readb64(uri):
    nparr = np.frombuffer(uri, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.post("/")
async def main():
    return {"success": True}


@app.post("/api/image")
async def analyze_route(request: Request):
    form = await request.form()
    if "image" in form:
        t1 = time.time()
        upload_file = form["image"]
        filename = form["image"].filename  # str
        image_base64 = await form["image"].read()  # bytes
        content_type = form["image"].content_type  # str
        image = readb64(image_base64)

        img_orig, img_model, ax, ay = prepare_for_detector(image)
        results = getPlates(img_orig, img_model, ax, ay)
        t2 = time.time()
        process_time = t2 - t1

        if not results:
            return {"status": False}
        else:
            return {
                "status": True,
                "results": results,
                "exec_time": process_time
            }
    else:
        return {"status": False, "message": "No image provided"}
