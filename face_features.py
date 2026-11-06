import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(img, target_size=(64, 64)):
    if img.dtype != np.uint8:
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    else:
        img_uint8 = img

    if len(img_uint8.shape) == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_uint8

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        gray = gray[y:y+h, x:x+w]

    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    return resized, norm

def extract_hog(norm_img):
    return hog(norm_img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

def extract_lbp(uint8_img, P=8, R=1):
    lbp = local_binary_pattern(uint8_img, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
    return hist

class FaceFeatureExtractor:
    def __init__(self, n_pca_components=50):
        self.n_pca_components = n_pca_components
        self.pca = PCA(n_components=n_pca_components, random_state=42)
        self.is_fitted = False

    def fit(self, norm_images):
        flat_images = np.array([img.ravel() for img in norm_images])
        self.pca.fit(flat_images)
        self.is_fitted = True
        return self

    def extract(self, uint8_img, norm_img):
        f_hog = extract_hog(norm_img)
        f_lbp = extract_lbp(uint8_img)
        if self.is_fitted:
            f_pca = self.pca.transform(norm_img.ravel().reshape(1, -1))[0]
            return np.hstack([f_hog, f_lbp, f_pca])
        return np.hstack([f_hog, f_lbp])
