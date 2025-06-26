import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ==== Configuration ====
TRAIN_DIR = r'C:\Users\apani\Desktop\data\train'
TEST_DIR = r'C:\Users\apani\Desktop\data\test1'
IMAGE_SIZE = 64
NUM_IMAGES_PER_CLASS = 1000


def load_images_from_folder(folder, label_name=None, label_val=None, count=None):
    images = []
    labels = []
    files = [f for f in os.listdir(folder) if label_name is None or f.startswith(label_name)]
    if count:
        files = files[:count]

    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        images.append(img)
        if label_val is not None:
            labels.append(label_val)
    return images, labels if label_val is not None else images


def prepare_training_data():
    print("Loading training data...")
    cat_imgs, cat_labels = load_images_from_folder(TRAIN_DIR, 'cat', 0, NUM_IMAGES_PER_CLASS)
    dog_imgs, dog_labels = load_images_from_folder(TRAIN_DIR, 'dog', 1, NUM_IMAGES_PER_CLASS)

    X = np.array(cat_imgs + dog_imgs)
    y = np.array(cat_labels + dog_labels)

    print("Preprocessing...")
    X = X / 255.0
    X_flat = X.reshape(len(X), -1)

    return train_test_split(X_flat, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    print("Training SVM model...")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    print("Evaluating...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

    cm = confusion_matrix(y_test, y_pred)

    # Ensure visuals directory exists
    os.makedirs("visuals", exist_ok=True)

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ['Cat', 'Dog'])
    plt.yticks([0, 1], ['Cat', 'Dog'])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("visuals/confusion_matrix.png")
    print("Saved confusion_matrix.png")
    plt.close()


def predict_test_images(model):
    print("Predicting on test data...")
    test_files = sorted(os.listdir(TEST_DIR), key=lambda x: int(x.split('.')[0]))
    X_test = []

    for file in test_files:
        img = cv2.imread(os.path.join(TEST_DIR, file))
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        X_test.append(img)

    X_test = np.array(X_test) / 255.0
    X_test_flat = X_test.reshape(len(X_test), -1)

    predictions = model.predict(X_test_flat)

    submission = pd.DataFrame({
        "id": [int(file.split('.')[0]) for file in test_files],
        "label": predictions
    })

    submission.sort_values("id", inplace=True)
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


def visualize_with_pca(X, y, model_full):
    print("Reducing to 2D with PCA for visualization...")
    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(X)

    # Train SVM on PCA-reduced data
    svm_2d = SVC(kernel='linear')
    svm_2d.fit(X_2D, y)
    support_vectors = svm_2d.support_

    # Create meshgrid
    h = .02
    x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
    y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Ensure visuals directory exists
    os.makedirs("visuals", exist_ok=True)

    # Plot and save PCA decision boundary
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.2)
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
    plt.scatter(X_2D[support_vectors, 0], X_2D[support_vectors, 1], facecolors='none',
                edgecolors='black', s=100, label='Support Vectors')

    plt.title("SVM Decision Boundary with PCA Projection")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visuals/pca_decision_boundary.png")
    print("Saved pca_decision_boundary.png")
    plt.close()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_training_data()
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)

    visualize_with_pca(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), model)

    predict_test_images(model)
