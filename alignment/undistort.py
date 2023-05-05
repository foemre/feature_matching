'''
Undistort the images in a folder using the calibration results

Example YAML file:

camera_matrix:
- - 1496.0279704487211
  - 0.0
  - 449.94721154701205
- - 0.0
  - 1163.5100956885287
  - 253.92834104463262
- - 0.0
  - 0.0
  - 1.0
dist_coeff:
- - -0.22636219772836919
  - -0.861380112534953
  - 0.007187378300494048
  - 0.0033653302956500398
  - 1.277063284309825
dist_model: plumb_bob
reprojection_error: 0.06605188958720562
image_width: 960
image_height: 576
'''
import cv2
import numpy as np
import glob
import argparse
import os
import yaml
ACCEPTED_IMAGE_FORMATS = ['jpg', 'png', 'jpeg']
def undistort(folder, calib_file):
    # Load YAML file
    with open(calib_file) as f:
        loadeddict = yaml.load(f, Loader=yaml.FullLoader)
        camera_matrix = loadeddict.get('camera_matrix')
        dist_coeff = loadeddict.get('dist_coeff')
        dist_model = loadeddict.get('dist_model')
        reprojection_error = loadeddict.get('reprojection_error')

    # Load images
    images =  glob.glob(folder + '/*.' + ACCEPTED_IMAGE_FORMATS[0]) + glob.glob(folder + '/*.' + ACCEPTED_IMAGE_FORMATS[1]) + glob.glob(folder + '/*.' + ACCEPTED_IMAGE_FORMATS[2])

    # Undistort images
    for fname in images:
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        image_camera_matrix = np.array(camera_matrix).reshape((3,3))
        image_dist_coeff = np.array(dist_coeff).reshape((1,5))
        
        # Modify the camera matrix based on the values obtained from calibration and the size of the image
        image_camera_matrix[0][0] = image_camera_matrix[0][0] * w / loadeddict.get('image_width')
        image_camera_matrix[0][2] = image_camera_matrix[0][2] * w / loadeddict.get('image_width')
        image_camera_matrix[1][1] = image_camera_matrix[1][1] * h / loadeddict.get('image_height')
        image_camera_matrix[1][2] = image_camera_matrix[1][2] * h / loadeddict.get('image_height')
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(image_camera_matrix, image_dist_coeff, (w,h), 0, (w,h))

        # undistort
        dst = cv2.undistort(img, image_camera_matrix, image_dist_coeff, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        cv2.imwrite(fname.split('.')[0] + '_undistorted.png', dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Undistort images using calibration results')
    parser.add_argument('--folder', help='Folder with images to undistort')
    parser.add_argument('--dest', help='Destination folder')
    parser.add_argument('--calib_file', help='Calibration file')
    args = parser.parse_args()
    undistort(args.folder, args.calib_file)