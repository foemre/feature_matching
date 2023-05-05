'''
Logic: 
1. Provide a ratio of overlap between the left and the right image.
2. Divide the overlapping region into Nx1 blocks. The blocks should be square; block height should divide image height. If it does not, 
    then adjust the overlap ratio to make it so.
3. For each block, perform SIFT feature matching between the left and the right block.
'''

import numpy as np
import cv2
from skimage.util.shape import view_as_blocks

def get_keypoints_descriptors(img_left, img_right, block_size=32, overlap_ratio=0.1):

    left_blocks = view_as_blocks(img_left, block_shape=(block_size, block_size))[:,-1,:,:]
    right_blocks = view_as_blocks(img_right, block_shape=(block_size, block_size))[:,0,:,:]

    # compute SIFT keypoints and descriptors in each block. Make it sensitive ie. make sure it returns a lot of keypoints
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.01, edgeThreshold=10, sigma=0.5)

    # compute keypoints and descriptors for the left blocks. Each element of the list is a list of keypoints and descriptors for a block
    left_kp_desc = [sift.detectAndCompute(block, None) for block in left_blocks]

    # compute keypoints and descriptors for the right blocks. Each element of the list is a list of keypoints and descriptors for a block
    right_kp_desc = [sift.detectAndCompute(block, None) for block in right_blocks]
    
    # Unpack the keypoints and descriptors from the list of lists
    left_kp, left_desc = zip(*left_kp_desc)
    right_kp, right_desc = zip(*right_kp_desc)

    # Compute the actual positions of the keypoints in the original image.
    # The keypoints are computed in the block, so we need to add the block offset to the keypoints
    for i, block in enumerate(left_kp):
        for kp in block:
            kp.pt = (kp.pt[0] + img_left.shape[1] - block_size, kp.pt[1] + block_size * i)
    for i, block in enumerate(right_kp):
        for kp in block:
            kp.pt = (kp.pt[0], kp.pt[1] + block_size * i)

    # Compute the matches between the left and the right keypoints for each block
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    matches = [bf.knnMatch(left_desc[i], right_desc[i], k=2) for i in range(len(left_desc))]
    # Flatten the list of lists
    matches = [match for block in matches for match in block]
    # length of the list is the number of matches
    print(len(matches))

    # Draw the matches
    img = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    img = cv2.drawMatches(img, left_kp, img, right_kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == '__main__':
    img_left = cv2.imread('sol.png', cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread('orta.png', cv2.IMREAD_GRAYSCALE)
    get_keypoints_descriptors(img_left, img_right)

