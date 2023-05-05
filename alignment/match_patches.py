import cv2 
import numpy as np
from skimage.util.shape import view_as_blocks

descriptor_type = 'ORB'

def get_keypoints_descriptors(img_left, img_right, block_size=(30,24), overlap_ratio=0.1):

    # Detect and compute SIFT features only on the blocks.
    # To do this, provide a mask to detectAndCompute() that only contains the block.
    # The mask is a binary image of the same size as the original image.
    # The mask is 0 everywhere except in the block, where it is 255.

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_left = clahe.apply(img_left)
    img_right = clahe.apply(img_right)

    left_blur = cv2.GaussianBlur(img_left, (3,3), 0)
    right_blur = cv2.GaussianBlur(img_right, (3,3), 0)
    img_left = cv2.addWeighted(img_left, 1.2, left_blur, -0.2, 0)
    img_right = cv2.addWeighted(img_right, 1.2, right_blur, -0.2, 0)

    # Upscale using Lanczos interpolation
    img_left = cv2.resize(img_left, (1280,960), interpolation=cv2.INTER_LANCZOS4)
    img_right = cv2.resize(img_right, (1280,960), interpolation=cv2.INTER_LANCZOS4)

    block_x, block_y = block_size
    left_kps = ()
    right_kps = ()
    matches = ()
    left_descs, right_descs = None, None
    # Create a numpy array to store the descriptors. Make sure we can concatenate descriptors of length 128
    #left_descs = np.empty((0, 128), dtype=np.float32)
    #right_descs = np.empty((0, 128), dtype=np.float32)
    if descriptor_type == 'SIFT':
        sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.01, edgeThreshold=6, sigma=1.2)
    elif descriptor_type == 'SURF':
        sift = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=4, nOctaveLayers=3, extended=True, upright=False)
    elif descriptor_type == 'ORB':
        sift = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=6, firstLevel=2, WTA_K=2, scoreType=cv2.ORB_FAST_SCORE, patchSize=32, fastThreshold=10)
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
            bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
            matches_block = bf.match(desc_left, desc_right)
            # Which descriptors are matched?
            for match in matches_block:
                # Offset the queryIdx and trainIdx by the number of descriptors in the previous blocks
                match.queryIdx += len(left_kps) - len(kp_left)
                match.trainIdx += len(right_kps) - len(kp_right)
                print(match.queryIdx, match.trainIdx)


        # Add the matches to the list
        matches += matches_block        

        # # draw matches and make sure kps and descs are not empty
        # if desc_left is not None and desc_right is not None and len(kp_left) > 0 and len(kp_right) > 0:
        #     img = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        #     img = cv2.drawMatches(img, kp_left, img_right, kp_right, matches_block, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #     cv2.imshow('matches', img)
        #     cv2.waitKey()
    
    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Show the matches
    matched_img = cv2.drawMatches(img_left, left_kps, img_right, right_kps, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches', matched_img)
    cv2.waitKey()

    # Estimate the homography
    # Convert keypoints to an array
    left_kps = np.float32([kp.pt for kp in left_kps])
    right_kps = np.float32([kp.pt for kp in right_kps])

    max_length = min(len(left_kps), len(right_kps))
    # Find the homography
    H, mask = cv2.findHomography(left_kps[:max_length-1], right_kps[:max_length-1], cv2.USAC_ACCURATE, 3.0)
    # Filter the matches using the mask
    # matches2 = [m for m, msk in zip(matches, mask) if msk == 1]

    # transform right image to left image
    img_right_transformed = cv2.warpPerspective(img_right, H, (img_left.shape[1], img_left.shape[0]))
    cv2.imshow('transformed', img_right_transformed)
    cv2.waitKey()

    # matched_img = cv2.drawMatches(img_left, left_kps, img_right, right_kps, matches, None) 
    # cv2.imshow('matches', matched_img)
    # cv2.waitKey()

if __name__ == '__main__':
    img_left = cv2.imread('sol_undistorted.png', cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread('orta_undistorted.png', cv2.IMREAD_GRAYSCALE)
    get_keypoints_descriptors(img_left, img_right)