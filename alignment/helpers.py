import numba
import cv2
import numpy as np
from skimage import feature
import random
import math

def detect_keypoints_lbp(image, num_points=24, radius=3, threshold=0.1, max_keypoints=10, min_distance=3, mask=None):
    # Apply the mask to the image using bitwise AND operation
    if mask is not None:
        image = cv2.bitwise_and(image, mask)

    # Compute LBP features
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")

    # Find keypoints
    keypoints = []
    lbp_diffs = []
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            lbp_diff = max(abs(lbp[y, x] - lbp[y - 1:y + 2, x - 1:x + 2]).max(), 0)
            if lbp_diff > threshold:
                keypoints.append(cv2.KeyPoint(x, y, 1))
                lbp_diffs.append(lbp_diff)

    # Shuffle the keypoints to avoid bias towards top-left corner
    combined = list(zip(keypoints, lbp_diffs))
    random.shuffle(combined)
    keypoints, lbp_diffs = zip(*combined)

    # Apply non-maximum suppression and select top N keypoints with minimum distance constraint
    keypoints, lbp_diffs = zip(*sorted(zip(keypoints, lbp_diffs), key=lambda x: x[1], reverse=True))
    filtered_keypoints = []
    for keypoint in keypoints:
        if len(filtered_keypoints) >= max_keypoints:
            break

        min_dist = min([cv2.norm(keypoint.pt, kp.pt) for kp in filtered_keypoints], default=min_distance)
        if min_dist >= min_distance:
            filtered_keypoints.append(keypoint)

    return filtered_keypoints

def detect_keypoints_lbp_vectorized(image, num_points=24, radius=3, threshold=0.1, max_keypoints=10, min_distance=3, mask=None):
    # Apply the mask to the image using bitwise AND operation
    if mask is not None:
        image = cv2.bitwise_and(image, mask)

    # Compute LBP features
    lbp = feature.local_binary_pattern(image, num_points, radius, method="uniform")

    # Find keypoints
    keypoints = []
    lbp_diffs = []
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            lbp_diff = max(abs(lbp[y, x] - lbp[y - 1:y + 2, x - 1:x + 2]).max(), 0)
            if lbp_diff > threshold:
                keypoints.append(cv2.KeyPoint(x, y, 1))
                lbp_diffs.append(lbp_diff)

    # Shuffle the keypoints to avoid bias towards top-left corner
    combined = list(zip(keypoints, lbp_diffs))
    random.shuffle(combined)
    keypoints, lbp_diffs = zip(*combined)

    # Apply non-maximum suppression and select top N keypoints with minimum distance constraint
    keypoints_arr = np.array([kp.pt for kp in keypoints])
    filtered_keypoints = []
    for keypoint in keypoints:
        if len(filtered_keypoints) >= max_keypoints:
            break

        distances = np.linalg.norm(keypoints_arr - keypoint.pt, axis=1)
        min_dist = np.min(distances, initial=min_distance)
        if min_dist >= min_distance:
            filtered_keypoints.append(keypoint)

    return filtered_keypoints


def match_histograms(image1, image2):
    # Convert images to grayscale
    gray1 = image1.copy()
    gray2 = image2.copy()

    # Match histograms of the two images
    hist1, _ = np.histogram(gray1.flatten(), bins=256, range=[0, 256])
    hist2, _ = np.histogram(gray2.flatten(), bins=256, range=[0, 256])

    cdf1 = hist1.cumsum()
    cdf2 = hist2.cumsum()

    cdf1_normalized = cdf1.astype('float64') / cdf1.max()
    cdf2_normalized = cdf2.astype('float64') / cdf2.max()

    lut = np.interp(cdf1_normalized, cdf2_normalized, np.arange(256))

    # Apply the lookup table to the second image
    matched = cv2.LUT(gray2, lut.astype('uint8'))

    return matched

def match_keypoints(descriptors1, descriptors2, max_distance=50, max_angle_diff=15):
    # Match keypoints using a k-NN matcher with cross-check
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    # Filter matches by distance and angle between lines
    filtered_matches = []
    prev_pt1 = None
    prev_pt2 = None
    for match in matches:
        if len(match) < 2:
            continue
        
        pt1 = match[0].queryIdx
        pt2 = match[0].trainIdx
        pt3 = match[1].trainIdx
        
        # Check distance between keypoints
        if match[0].distance > max_distance:
            continue
        
        # Check angle between lines
        if prev_pt1 is not None and prev_pt2 is not None:
            angle1 = math.atan2(descriptors1[pt1][1], descriptors1[pt1][0])
            angle2 = math.atan2(descriptors2[pt2][1] - descriptors2[pt3][1], descriptors2[pt2][0] - descriptors2[pt3][0])
            angle_diff = abs(math.degrees(angle2 - angle1))
            
            if angle_diff > max_angle_diff:
                continue
        
        filtered_matches.append(match[0])
        prev_pt1 = pt1
        prev_pt2 = pt2
    
    return filtered_matches

def preprocess_image(image, gamma=1.5):

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    # Apply gamma correction
    image = np.power(image/float(np.max(image)), gamma)
    image = np.uint8(image*255)

    # Apply a soft sharpening filter
    # np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) is too strong
    # kernel = np.array([[-0.1, -0.1, -0.1], [-0.1, 1.8, -0.1], [-0.1, -0.1, -0.1]])
    # image = cv2.filter2D(image, -1, kernel)
    
    return image

