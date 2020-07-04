from pre_processing import *
import cv2
from tensorflow_core.python.keras.models import load_model
import time


def start():
    cap = cv2.VideoCapture(0)
    cap.set(3, 650)
    cap.set(4, 400)
    time.sleep(2)

    model_saved = load_model("sdr_model.h5")
    results = ['bkl', 'mel', 'nv', 'working']

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # save and load again to process
        if frame is not None:
            cv2.imwrite('live.jpg', frame)
            live = cv2.imread('live.jpg')
        # extract features and predict label of image
        img_sample, img_px = etl_one_img(live)
        # print(img_sample.shape)
        label = model_saved.predict_classes(img_sample)
        cv2.putText(frame, results[label[0]], (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        # time.sleep(10)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
