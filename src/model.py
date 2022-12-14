import tensorflow as tf
import numpy as np
import cv2

classes = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
    "ship", "truck"
]


def main():
    # use the model that is in models/cifar10
    model = tf.keras.models.load_model("models/cifar10/")
    # load example image
    # img = load_img("src/assets/plane.jpeg")
    # # FIXME: Image dimensions are 185x275, however, the model expects 32, 32 images.
    # img = np.reshape(img, (32, 32, 3))
    # img = img_to_array(img)[:, :, :1]  # grayscale
    img = cv2.imread("src/assets/bird.jpeg")
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    img = np.expand_dims(img, axis=0)  # 32x32x3 = ?x
    Y_pred = model.predict(img)[0]
    idx = 0
    max_val = 0
    for i in range(0, len(Y_pred)):
        if (Y_pred[i] > max_val):
            idx = i
            max_val = Y_pred[i]

    print(classes[idx])


if __name__ == "__main__":
    main()
