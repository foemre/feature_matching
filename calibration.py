# Camera Calibration Tool
# Author: F.O.Emre Erdem
# Estimates intrinsic parameters, distortion coefficients and reprojection error
# using a set of chessboard images.
# Saves the calibration results to a file.

import numpy as np
import cv2
import glob
import argparse
import os
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

def calibrate(w, h, folder, save_undistorted_images=False):
    '''
    w: Number of corners in width
    h: Number of corners in height
    folder: Folder containing the images
    '''

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(w-1,h-1,0)
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane

    # Get the images
    images = glob.glob(folder + '/*.jpg')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # invert the image
        gray = cv2.bitwise_not(gray)

        # Apply unsharp mask and adaptive histogram equalization
        blur = cv2.GaussianBlur(gray, (0,0), 3)
        gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        gray = clahe.apply(gray)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (w,h), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print('Found corners in ' + fname)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # Measure the reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    # print reprojection error in pixels
    print('Reprojection error: {} pixels'.format(mean_error/len(objpoints)))

    # Save the calibration results as a yaml file. Keep the matrix structures
    data = {'camera_matrix': np.asarray(mtx).tolist(),
            'dist_model': 'plumb_bob',
            'dist_coeff': np.asarray(dist).tolist(),
            'reprojection_error': mean_error/len(objpoints),
            'image_width': gray.shape[1],
            'image_height': gray.shape[0]}
    with open(folder + '/calibration.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    if save_undistorted_images:
        undistorted_folder = folder + '/undistorted'
        # Undistort the images
        for fname in images:
            img = cv2.imread(fname)
            h,  w = img.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]

            # resize the image to 640x480
            dst = cv2.resize(dst, (640, 480))

            # save the undistorted image to "undistorted" folder under the image folder. check if the folder exists
            if not os.path.exists(undistorted_folder):
                os.makedirs(undistorted_folder)
            cv2.imwrite(undistorted_folder + '/' + fname.split('/')[-1], dst)


def main():
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--width', type=int, default=9, help='Number of corners in width')
    parser.add_argument('--height', type=int, default=6, help='Number of corners in height')
    parser.add_argument('--folder', type=str, default='images', help='Folder containing the images')
    parser.add_argument('--undistort', type=bool, default=False, help='Save undistorted images')

    args = parser.parse_args()

    calibrate(args.width, args.height, args.folder, args.undistort)

if __name__ == '__main__':
    main()
    