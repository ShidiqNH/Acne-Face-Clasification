import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
import dlib
from imutils import face_utils
import os
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.uic import loadUi

class SkinClassification(QMainWindow):
    def __init__(self):
        super(SkinClassification, self).__init__()
        loadUi('interface.ui', self)
        
        self.predictionLabel.setText("Prediction : None")
        self.loadButton.clicked.connect(self.loadClicked)
        self.predictButton.clicked.connect(self.showDetails)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.load_and_train_model()
        
        self.loadedImage = None
        self.face_only_cropped = None
        self.input_pca = None
        self.prediction = None
        self.lbp_image = None
        self.lbp_hist = None
    
    def loadClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Select an Image File", "", "Image Files (*.png *.jpg *.bmp *.jpeg *.JPG);;All Files (*)", options=options)
        
        if fileName:
            self.loadedImage = fileName
            pixmap = QtGui.QPixmap(fileName)
            self.outputWindow.setPixmap(pixmap)
            self.outputWindow.setScaledContents(True)
            self.inputImageProcess()
    
    def str_to_list(self, s):
        return list(map(float, s.strip('[]').split(',')))
    
    def load_and_preprocess_data(self, file_path, label):
        df = pd.read_csv(file_path)
        df['label'] = label
        df['LBP Histogram Value'] = df['LBP Histogram Value'].apply(self.str_to_list)
        return df
    
    def extract_lbp_from_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        self.lbp_image = lbp
        self.lbp_hist = hist
        return hist
    
    def load_and_train_model(self):
        lbp_acne_df = self.load_and_preprocess_data('Dataset LBP/lbpValueJerawat.csv', 'jerawat')
        lbp_normal_df = self.load_and_preprocess_data('Dataset LBP/lbpValueNormal.csv', 'normal')
        
        combined_df = pd.concat([lbp_acne_df, lbp_normal_df], ignore_index=True)
        X = np.array(combined_df['LBP Histogram Value'].tolist())
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X_scaled)
        
        y = combined_df['label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn.fit(X_train, y_train)
        
        self.X_pca = X_pca
        self.y = y
    
    def inputImageProcess(self):
        input_image_path = self.loadedImage
        if input_image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.heic')):
            image = cv2.imread(input_image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)

            if len(rects) == 0:
                self.predictionLabel.setText("No face detected in the input image.")
                self.face_only_cropped = None
                self.input_pca = None
                self.prediction = None
            else:
                for (i, rect) in enumerate(rects):
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    jaw = shape[0:17]
                    left_eyebrow = shape[17:22]
                    right_eyebrow = shape[22:27]
                    nose = shape[27:36]
                    left_eye = shape[36:42]
                    right_eye = shape[42:48]
                    mouth = shape[48:68]

                    points = np.concatenate([jaw, left_eyebrow, right_eyebrow, nose, left_eye, right_eye, mouth])
                    mask = np.zeros_like(gray)
                    hull = cv2.convexHull(points)
                    cv2.drawContours(mask, [hull], -1, (255), -1)

                    (x, y, w, h) = cv2.boundingRect(hull)
                    forehead_height = int(0.2 * h)
                    y_start = max(0, y - forehead_height)
                    y_end = min(y + h, image.shape[0])
                    x_end = min(x + w, image.shape[1])

                    if y_start >= y_end or x >= x_end:
                        self.predictionLabel.setText("Cropping region is invalid for the input image.")
                        self.face_only_cropped = None
                        self.input_pca = None
                        self.prediction = None
                    else:
                        mask_expanded = np.zeros_like(gray)
                        hull_expanded = np.concatenate([shape, np.array([[x, y_start], [x + w, y_start]])])
                        hull_expanded = cv2.convexHull(hull_expanded)
                        cv2.drawContours(mask_expanded, [hull_expanded], -1, (255), -1)

                        face_only = cv2.bitwise_and(image, image, mask=mask_expanded)
                        self.face_only_cropped = face_only[y_start:y_end, x:x_end]

                        if self.face_only_cropped.size == 0:
                            self.predictionLabel.setText("Cropped face image is empty for the input image.")
                            self.face_only_cropped = None
                            self.input_pca = None
                            self.prediction = None
                        else:
                            lbp_input = self.extract_lbp_from_image(self.face_only_cropped)
                            input_scaled = self.scaler.transform([lbp_input])
                            self.input_pca = self.pca.transform(input_scaled)
                            self.prediction = self.knn.predict(input_scaled)[0]
                            self.predictionLabel.setText('Prediction: ' + self.prediction)
        else:
            self.predictionLabel.setText("Input file format is not supported.")

    def showDetails(self):
        if self.face_only_cropped is not None and self.input_pca is not None:
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))

            # KNN Classification Plot
            for label in np.unique(self.y):
                axes[0].scatter(self.X_pca[self.y == label, 0], self.X_pca[self.y == label, 1], label=label)
            axes[0].scatter(self.input_pca[:, 0], self.input_pca[:, 1], marker='o', color='red', label='Input Image')
            axes[0].legend()
            axes[0].set_title('KNN Classification')
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')

            # Cropped Face Image
            axes[1].imshow(cv2.cvtColor(self.face_only_cropped, cv2.COLOR_BGR2RGB))
            axes[1].set_title('Kategori: ' + self.prediction)
            axes[1].axis('off')

            # LBP Image
            axes[2].imshow(self.lbp_image, cmap='gray')
            axes[2].set_title('LBP Image')
            axes[2].axis('off')

            # LBP Histogram
            axes[3].bar(np.arange(len(self.lbp_hist)), self.lbp_hist, color='black')
            axes[3].set_title('LBP Histogram')
            axes[3].set_xlabel('Bins')
            axes[3].set_ylabel('Frequency')

            plt.show()
        else:
            self.predictionLabel.setText("No face image to show details.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkinClassification()
    window.setWindowTitle('Skin Classification')
    window.show()
    sys.exit(app.exec_())
