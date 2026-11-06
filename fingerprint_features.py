import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# Pre-build Gabor filters
GABOR_WAVELENGTHS = [4, 8, 16]
GABOR_ORIENTATIONS = [0, 45, 90, 135]

def build_gabor_bank():
    bank = []
    for lam in GABOR_WAVELENGTHS:
        for deg in GABOR_ORIENTATIONS:
            theta = np.deg2rad(deg)
            sigma = lam / np.pi * np.sqrt(np.log(2) / 2) * 3.0
            ksize = int(6 * sigma + 1) | 1
            kr = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma=0.5, psi=0, ktype=cv2.CV_32F)
            ki = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma=0.5, psi=np.pi / 2, ktype=cv2.CV_32F)
            bank.append((kr, ki))
    return bank

_GABOR_BANK = build_gabor_bank()
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_fingerprint(img, target_size=(96, 96)):
    """Preprocess fingerprint: uint8 grayscale, resize, CLAHE enhancement, normalize to [0, 1]."""
    if not isinstance(img, np.ndarray):
        img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img}")

    if img.dtype != np.uint8:
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    else:
        img_uint8 = img

    if len(img_uint8.shape) == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

    h, w = target_size
    resized = cv2.resize(img_uint8, (w, h), interpolation=cv2.INTER_CUBIC)
    enhanced = _CLAHE.apply(resized)
    norm = enhanced.astype(np.float32) / 255.0
    return enhanced, norm

def extract_gabor(enhanced_uint8):
    feats = []
    for kr, ki in _GABOR_BANK:
        rr = cv2.filter2D(enhanced_uint8, cv2.CV_32F, kr)
        ri = cv2.filter2D(enhanced_uint8, cv2.CV_32F, ki)
        energy = np.sqrt(rr**2 + ri**2)
        feats.extend([float(energy.mean()), float(energy.std())])
    return np.array(feats, dtype=np.float32)

def extract_fingerprint_hog(norm_img):
    return hog(norm_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').astype(np.float32)

def extract_fingerprint_lbp(enhanced_uint8, P=24, R=3):
    lbp = local_binary_pattern(enhanced_uint8, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
    return hist.astype(np.float32)

def extract_fingerprint_features(enhanced_uint8, norm_img):
    """Extract fused Gabor + HOG + LBP features."""
    f_gabor = extract_gabor(enhanced_uint8)
    f_hog = extract_fingerprint_hog(norm_img)
    f_lbp = extract_fingerprint_lbp(enhanced_uint8)
    return np.concatenate([f_gabor, f_hog, f_lbp]).astype(np.float32)
