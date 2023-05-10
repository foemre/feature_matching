import cv2 
import numpy as np
from skimage.util.shape import view_as_blocks
import math

descriptor_type = 'ORB'

def get_keypoints_descriptors(img_left, img_right, block_size=(12, 8)):

    # Detect and compute SIFT features only on the blocks.
    # To do this, provide a mask to detectAndCompute() that only contains the block.
    # The mask is a binary image of the same size as the original image.
    # The mask is 0 everywhere except in the block, where it is 255.

    # Enhance contrast using CLAHE

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2LAB)
    img_left[:,:,0] = clahe.apply(img_left[:,:,0])
    img_left = cv2.cvtColor(img_left, cv2.COLOR_LAB2BGR)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2LAB)
    img_right[:,:,0] = clahe.apply(img_right[:,:,0])
    img_right = cv2.cvtColor(img_right, cv2.COLOR_LAB2BGR)

    # convert to grayscale
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    scale = 1.5
    res = (int(640*scale), int(480*scale))
    block_size = (int(block_size[0]*scale), int(block_size[1]*scale))

    left_blur = cv2.GaussianBlur(img_left, (3,3), 0)
    right_blur = cv2.GaussianBlur(img_right, (3,3), 0)
    img_left = cv2.addWeighted(img_left, 1.2, left_blur, -0.2, 0)
    img_right = cv2.addWeighted(img_right, 1.2, right_blur, -0.2, 0)

    # Upscale using Lanczos interpolation
    img_left = cv2.resize(img_left, res, interpolation=cv2.INTER_LANCZOS4)
    img_right = cv2.resize(img_right, res, interpolation=cv2.INTER_LANCZOS4)

    block_x, block_y = block_size
    left_kps = ()
    right_kps = ()
    matches = ()
    left_descs, right_descs = None, None

    if descriptor_type == 'SIFT':
        sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.001, edgeThreshold=6, sigma=1.2)
    elif descriptor_type == 'SURF':
        sift = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=4, nOctaveLayers=3, extended=True, upright=False)
    elif descriptor_type == 'ORB':
        sift = cv2.ORB_create(nfeatures=1000, scaleFactor=1.1, nlevels=24, edgeThreshold=4, firstLevel=4, WTA_K=4, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=block_x//4, fastThreshold=10)
    else:
        raise Exception('Unknown descriptor type')
    
    # Loop over the blocks, and for each block, set the corresponding region in the mask to 255
    for i in range(img_left.shape[0] // block_y):
        mask_left = np.zeros(img_left.shape, dtype=np.uint8)
        mask_left[i*block_y:(i+1)*block_y, -block_x:] = 255

        mask_right = np.zeros(img_right.shape, dtype=np.uint8)
        mask_right[i*block_y:(i+1)*block_y, :block_x] = 255

        # Detect and compute SIFT features in the block
        kp_left, desc_left = sift.detectAndCompute(img_left, mask_left)
        kp_right, desc_right = sift.detectAndCompute(img_right, mask_right)

        # Add the keypoints and descriptors to the lists

        left_kps += kp_left
        right_kps += kp_right
        
        # concatenate the descriptors if they are not empty (None)
        if desc_left is not None:
            if left_descs is None:
                left_descs = desc_left
            else:
                left_descs = np.concatenate((left_descs, desc_left), axis=0)
        if desc_right is not None:
            if right_descs is None:
                right_descs = desc_right
            else:
                right_descs = np.concatenate((right_descs, desc_right), axis=0)

        # Match the features
        if desc_left is not None and desc_right is not None:
            if descriptor_type == 'ORB':
                bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
            else:
                bf = cv2.BFMatcher(crossCheck=True)
            matches_block = bf.match(desc_left, desc_right)
            # Which descriptors are matched?
            for match in matches_block:
                # Offset the queryIdx and trainIdx by the number of descriptors in the previous blocks
                match.queryIdx += len(left_kps) - len(kp_left)
                match.trainIdx += len(right_kps) - len(kp_right)
                print(match.queryIdx, match.trainIdx)
                # get the keypoints from the match queryidx and trainidx
                kp1 = left_kps[match.queryIdx]
                kp2 = right_kps[match.trainIdx]
                # get the coordinates of the keypoints
                x1, y1 = kp1.pt
                x2, y2 = kp2.pt

            # Add the matches to the list
            matches += matches_block  

    # If the Euclidean distance between two keypoints is larger than 
    # the distance between the first pair of keypoints, remove the match
    # First match is the match where the y coordinate of the left keypoint is smallest
    # if the angle is not within the range, remove the match. The range is -2 to -20 degrees. Angle is
    # calculated from the left keypoint to the right keypoint

    first_match = matches[0]
    first_keypoint_left = left_kps[first_match.queryIdx]
    first_keypoint_right = right_kps[first_match.trainIdx]
    first_x_diff = first_keypoint_right.pt[0] - first_keypoint_left.pt[0] + img_left.shape[1]
    first_y_diff = first_keypoint_left.pt[1] - first_keypoint_right.pt[1]
    first_distance = np.sqrt(first_x_diff**2 + first_y_diff**2)
    first_angle = np.arctan2(first_y_diff, first_x_diff) * 180 / np.pi
    for match in matches:
        keypoint_left = left_kps[match.queryIdx]
        keypoint_right = right_kps[match.trainIdx]
        x_diff = keypoint_right.pt[0] - keypoint_left.pt[0] +  img_left.shape[1]
        y_diff = keypoint_left.pt[1] - keypoint_right.pt[1]
        distance = np.sqrt(x_diff**2 + y_diff**2)
        if abs(distance - first_distance) > 2 or abs(np.arctan2(y_diff, x_diff) * 180 / np.pi - first_angle) > 5:
            matches = tuple([m for m in matches if m != match])        

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Show the matches
    matched_img = cv2.drawMatches(img_left, left_kps, img_right, 
                                  right_kps, matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches', matched_img)
    cv2.waitKey()

    # save the matches
    cv2.imwrite('matches.png', matched_img)

    # Estimate the homography
    # Convert keypoints to an array
    left_kps = np.float32([kp.pt for kp in left_kps])
    right_kps = np.float32([kp.pt for kp in right_kps])

    max_length = min(len(left_kps), len(right_kps))
    # Find the homography
    H, mask = cv2.findHomography(left_kps[:max_length-1], right_kps[:max_length-1], cv2.RANSAC, 5.0)

    # Warp the left image to the right image plane
    img_left_warped = cv2.warpPerspective(img_left, H, (img_left.shape[1], img_left.shape[0]))

    # Show the warped image
    cv2.imshow('warped', img_left_warped)
    cv2.waitKey()

if __name__ == '__main__':
    img_left = cv2.imread('sol_undistorted.png')
    img_right = cv2.imread('orta_undistorted.png')
    get_keypoints_descriptors(img_left, img_right)