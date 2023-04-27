# stitch.py

from helpers import *
import cv2
import numpy as np
import argparse
import tqdm
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', type=str, default='input1.jpg', help='path to the input image')
    parser.add_argument('--image2', type=str, default='input2.jpg', help='path to the input image')
    args = parser.parse_args()
    image1 = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(args.image2, cv2.IMREAD_GRAYSCALE)

    # Preprocess images
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    image1 = match_histograms(image2, image1)

    # Convert images to grayscale
    gray1 = image1.copy()
    gray2 = image2.copy()

    mask_left = np.zeros(gray1.shape, dtype=np.uint8)
    # rightmost 10% of the image is set to 255
    mask_left[:, int(0.95 * gray1.shape[1]):] = 255

    mask_right = np.zeros(gray2.shape, dtype=np.uint8)
    # leftmost 10% of the image is set to 255
    mask_right[:, :int(0.05 * gray2.shape[1])] = 255

    # Compute LBP features
    keypoints1 = detect_keypoints_lbp_vectorized(gray1, mask=mask_left, max_keypoints=200, threshold=0.05)
    keypoints2 = detect_keypoints_lbp_vectorized(gray2, mask=mask_right, max_keypoints=200, threshold=0.05)
    print(len(keypoints1))

    # Draw keypoints on the images
    image1_show = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2_show = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image1', image1_show)
    cv2.imshow('image2', image2_show)
    cv2.waitKey()

    # Detect keypoints and compute SIFT descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.compute(gray1, keypoints1)
    keypoints2, descriptors2 = sift.compute(gray2, keypoints2)

    # Match keypoints 
    matches = match_keypoints(descriptors1, descriptors2, max_distance=300, max_angle_diff=15)

    # Draw top matches
    image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:20], None)
    cv2.imshow('matches', image_matches)
    cv2.waitKey()

    # Compute homography using RANSAC
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    # Warp and stitch images
    result = cv2.warpPerspective(image2, H, (image1.shape[1] + image2.shape[1], image2.shape[0]))
    result[0:image1.shape[0], 0:image1.shape[1]] = image1

    # Save the stitched image
    cv2.imwrite('stitched_image.jpg', result)

if __name__ == '__main__':
    main()
