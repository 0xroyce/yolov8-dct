# auto_classifier.py

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class AutoClassifier:
    def __init__(self, photo_dir):
        self.photo_dir = photo_dir
        self.features = []
        self.file_names = []
        self.labels = {}

    def extract_features(self, image):
        # Simple feature extraction (you might want to use a pre-trained CNN for better results)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return hist.flatten()

    def load_images(self):
        for obj_dir in os.listdir(self.photo_dir):
            obj_path = os.path.join(self.photo_dir, obj_dir)
            if os.path.isdir(obj_path):
                for filename in os.listdir(obj_path):
                    if filename.endswith(".jpg"):
                        image_path = os.path.join(obj_path, filename)
                        image = cv2.imread(image_path)
                        features = self.extract_features(image)
                        self.features.append(features)
                        self.file_names.append(os.path.join(obj_dir, filename))

    def cluster_images(self):
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(self.features)
        kmeans = KMeans(n_clusters=min(len(self.features), 5))  # Adjust number of clusters as needed
        kmeans.fit(normalized_features)
        return kmeans.labels_

    def get_user_labels(self, cluster_labels):
        unique_clusters = set(cluster_labels)
        for cluster in unique_clusters:
            sample_image = self.file_names[cluster_labels.tolist().index(cluster)]
            print(f"Cluster {cluster} - Sample image: {sample_image}")
            label = input("What is this object? ")
            self.labels[cluster] = label

    def apply_labels(self, cluster_labels):
        for filename, cluster in zip(self.file_names, cluster_labels):
            obj_dir, old_name = os.path.split(filename)
            new_name = f"{self.labels[cluster]}_{old_name}"
            old_path = os.path.join(self.photo_dir, filename)
            new_path = os.path.join(self.photo_dir, obj_dir, new_name)
            os.rename(old_path, new_path)

    def run(self):
        self.load_images()
        cluster_labels = self.cluster_images()
        self.get_user_labels(cluster_labels)
        self.apply_labels(cluster_labels)

if __name__ == "__main__":
    classifier = AutoClassifier("cropped_unknown_objects")
    classifier.run()