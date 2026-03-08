import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def load_images(folder, limit=500):
    features = []
    labels = []

    for label in ['cats', 'dogs']:
        path = os.path.join(folder, label)
        count = 0

        for file in os.listdir(path):

            if count >= limit:
                break

            img_path = os.path.join(path, file)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hog_features = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )

            features.append(hog_features)

            if label == 'cats':
                labels.append(0)
            else:
                labels.append(1)

            count += 1

    return np.array(features), np.array(labels)


train_path = "dataset/training_set"
test_path = "dataset/test_set"

print("Loading training data...")
X_train, y_train = load_images(train_path, 500)

print("Loading test data...")
X_test, y_test = load_images(test_path, 200)


print("Training SVM model...")
model = LinearSVC()

model.fit(X_train, y_train)


print("Predicting...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


def predict_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found")
        return

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    hog_features = hog_features.reshape(1, -1)

    prediction = model.predict(hog_features)

    if prediction == 0:
        print("Prediction: Cat")
    else:
        print("Prediction: Dog")


predict_image("dataset/test_set/dogs/dog.4001.jpg")