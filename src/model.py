import tensorflow as tf
import numpy as np
import cv2

classes = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
    "ship", "truck"
]

classes_lung = ["NORMAL", "BENIGN", "MALIGNANT"]


def main():
    # use the model that is in models/cifar10
    model = tf.keras.models.load_model("../models/lung/")
    # load example image
    # img = load_img("src/assets/plane.jpeg")
    # # FIXME: Image dimensions are 185x275, however, the model expects 32, 32 images.
    # img = np.reshape(img, (32, 32, 3))
    # img = img_to_array(img)[:, :, :1]  # grayscale
    img = cv2.imread("assets/benignlung.jpeg")
    if img is None:
        print("Couldn't read the image.")
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imshow("zaza", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img2 = cv2.imread(
        "assets/datasets/lungIQ/malignant/Malignant case (2).jpg")
    img2 = cv2.resize(img2, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imshow("zaza", img2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    img = np.expand_dims(img, axis=0)  # 32x32x3 = ?x
    Y_pred = model.predict(img)
    print(Y_pred)
    Y_pred = Y_pred[0]
    idx = 0
    max_val = 0
    for i in range(0, len(Y_pred)):
        if (Y_pred[i] > max_val):
            idx = i
            max_val = Y_pred[i]

    print(classes_lung[idx])


if __name__ == "__main__":
    main()
