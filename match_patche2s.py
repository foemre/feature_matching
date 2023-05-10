import cv2 
import numpy as np
from skimage.util.shape import view_as_blocks
import argparse

def get_keypoints_descriptors(img_left, img_right, block_size=(48,20), descriptor_type='SIFT'):

    # Detect and compute SIFT features only on the blocks.
    # To do this, provide a mask to detectAndCompute() that only contains the block.
    # The mask is a binary image of the same size as the original image.
    # The mask is 0 everywhere except in the block, where it is 255.

    img_left = cv2.imread(img_left, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(img_right, cv2.IMREAD_GRAYSCALE)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_left = clahe.apply(img_left)
    img_right = clahe.apply(img_right)

    left_blur = cv2.GaussianBlur(img_left, (3,3), 0)
    right_blur = cv2.GaussianBlur(img_right, (3,3), 0)
    img_left = cv2.addWeighted(img_left, 1.2, left_blur, -0.2, 0)
    img_right = cv2.addWeighted(img_right, 1.2, right_blur, -0.2, 0)

    # Upscale using Lanczos interpolation
    img_left = cv2.resize(img_left, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    img_right = cv2.resize(img_right, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)

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
        sift = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=6, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_FAST_SCORE, patchSize=31, fastThreshold=20)
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
        # Add the matches to the list
        matches += matches_block        

        # # draw matches and make sure kps and descs are not empty
        # if desc_left is not None and desc_right is not None and len(kp_left) > 0 and len(kp_right) > 0:
        #     img = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
        #     img = cv2.drawMatches(img, kp_left, img_right, kp_right, matches_block, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #     cv2.imshow('matches', img)
        #     cv2.waitKey()
    # sort matches
    matches = sorted(matches, key=lambda x: x.distance)

    # img3 = cv2.drawMatches(img_left, left_kps, img_right, right_kps, matches, None)
    # cv2.imshow('matches', img3)
    # cv2.waitKey()

    # Estimate the fundamental matrix between the two images

    # Get the keypoints from the matches
    left_pts = np.float32([left_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    right_pts = np.float32([right_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate the essential matrix
    E, mask = cv2.findEssentialMat(right_pts, left_pts, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
    F, mask = cv2.findFundamentalMat(left_pts, right_pts, method=cv2.RANSAC, ransacReprojThreshold=1.0)

    # Recover the pose from the essential matrix
    points, R, t, mask = cv2.recoverPose(E, left_pts, right_pts, focal=1.0, pp=(0., 0.))

    # Get the rotation matrix from the rotation vector
    R = cv2.Rodrigues(R)[0]

    # Get the camera projection matrices
    P1 = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                        [0., 0., 1., 0.]])
    P2 = np.hstack((R, t))

    # Triangulate the points
    points_4d = cv2.triangulatePoints(P1, P2, left_pts, right_pts)

    # Convert the points to 3D
    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_left', type=str, default='sol.png', help='Left image')
    parser.add_argument('--image_right', type=str, default='orta.png', help='Right image')
    parser.add_argument('--descriptor_type', type=str, default='SIFT', help='Descriptor type (SIFT, SURF, ORB)')
    parser.add_argument('--block_size', default='48x20', help='Block size (width x height)')
    args = parser.parse_args()

    # block size is a tuple of (width, height)
    block_size = args.block_size.split('x')
    block_size = (int(block_size[0]), int(block_size[1]))


    get_keypoints_descriptors(img_left=args.image_left, 
                              img_right=args.image_right, 
                              descriptor_type=args.descriptor_type, 
                              block_size=block_size)