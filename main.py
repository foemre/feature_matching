# main.py
# Author : F.O.Emre ERDEM
# Implementation of "Alignment and Mosaicing of Non-Overlapping Images" by Y.Poleg and S. Peleg
# https://www.cs.huji.ac.il/~peleg/papers/iccp12-no-overlap.pdf

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm
import numba
from numba import njit

@numba.njit
def compute_distance(p, q):
    return np.sqrt(np.sum((p - q)**2))

def find_most_similar_box(box, img):
    '''
    Patch similarity is inverse to the sum of distances of
    all corresponding pixels. The distance between two pixels p
    and q is is computed in the LAB color space, and is based on
    the difference in each of the L, A, and B color components.
    Distance is defined as the Euclidean distance between the two
    color vectors.
    '''

    h, w = img.shape[:2]
    box_size = box.shape[0]
    # Compute the distance between the box and every box in the image.
    # The distance is computed in the LAB color space.
    distances = np.zeros((h - box_size + 1, w - box_size + 1))
    for i in range(h - box_size + 1):
        for j in range(w - box_size + 1):
            distances[i, j] = np.sum([compute_distance(box[k, l], img[i+k, j+l]) for k in range(box_size) for l in range(box_size)])
    # Find the box with the smallest distance to the input box.
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return img[min_idx[0]:min_idx[0]+box_size, min_idx[1]:min_idx[1]+box_size]
            

def single_scale_pyramid_extrapolate(img, box_size=5):
    '''
    Single scale pyramid extrapolation:
    Take a box of size box_size x box_size. box_size should be odd.
    Place the center of the box at the edge of the image. 
    Slide the box along the boundary pixels of the image. You'll notice that (box_size+1)/2 pixels are inside the image, and (box_size-1)/2 pixels are outside the image.
    Start from the top left corner. Move along the left side of the image. Stride is 1 pixel.
    For each stride, search for a box inside the entire image whose left side is most similar to the left side of the box at the boundary.
    Copy the right side of the box you found to the right side of the box at the boundary.
    Repeat for the other three sides.
    '''
    h, w = img.shape[:2]
    img_out = np.pad(img, ((box_size-1, box_size-1), (box_size-1, box_size-1), (0,0)), 'symmetric')
    # Move the box along the left side of the image. Center of the box is at the edge pixels of the image.
    for i in tqdm.tqdm(range(h - box_size + 1)):
        box = img_out[i:i+box_size, :box_size]
        box_out = find_most_similar_box(box, img_out)
        img_out[i:i+box_size, :box_size] = box_out
    # Move the box along the right side of the image. Center of the box is at the edge pixels of the image.
    for i in tqdm.tqdm(range(h - box_size + 1)):
        box = img_out[i:i+box_size, -box_size:]
        box_out = find_most_similar_box(box, img_out)
        img_out[i:i+box_size, -box_size:] = box_out
    # Move the box along the top side of the image. Center of the box is at the edge pixels of the image.
    for i in tqdm.tqdm(range(w - box_size + 1)):
        box = img_out[:box_size, i:i+box_size]
        box_out = find_most_similar_box(box, img_out)
        img_out[:box_size, i:i+box_size] = box_out
    # Move the box along the bottom side of the image. Center of the box is at the edge pixels of the image.
    for i in tqdm.tqdm(range(w - box_size + 1)):
        box = img_out[-box_size:, i:i+box_size]
        box_out = find_most_similar_box(box, img_out)
        img_out[-box_size:, i:i+box_size] = box_out
    
    return img_out

def multi_scale_pyramid_extrapolate(img, box_size=5, pyr_level=5, scale=1.0):
    '''
    Perform single scale pyramid extrapolation at every level of the image pyramid.
    Create an image pyramid with 5 levels.
    Perform single scale pyramid extrapolation at the smallest scale first.
    Then, the output of the smallest scale is used as input for the next scale.
    '''
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
    # Pad the image with symmetric padding, padding size should be proportional to the box size and the number of levels in the pyramid
    pad_size = int((box_size - 1) / 2)**pyr_level

    # Pad the image with symmetric padding. Convert the image to float32.
    img_out = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0,0)), 'symmetric').astype(np.float32)

    # Create a 5-level gaussian pyramid
    pyramid = [img_out]
    for i in range(pyr_level - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))

    results_pyramid = []

    # Perform single scale pyramid extrapolation at each level of the pyramid
    # Start from the smallest scale and extrapolate the edges. Upsample (pyrUp) the result up to the original size and add the extrapolated edges to the original image.
    for i in range(pyr_level - 1, -1, -1):
        img_out = single_scale_pyramid_extrapolate(pyramid[i], box_size)
        img_out = cv2.pyrUp(img_out)
        results_pyramid.append(img_out)
        cv2.imwrite('output{}.jpg'.format(i), img_out)

    # Add all levels of the pyramid together. Weight the levels by 1/2^i, where i is the level number.
    for i in range(pyr_level):
        img_out += results_pyramid[i] / 2**i

    # convert the image back to uint8
    img_out = img_out.astype(np.uint8)

    # save the image
    cv2.imwrite('output.jpg', img_out)

    return img_out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='input.jpg', help='path to the input image')
    args = parser.parse_args()
    img = cv2.imread(args.image)
    img_out = multi_scale_pyramid_extrapolate(img, box_size=5, pyr_level=3, scale=1)
    cv2.imshow('image', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


    

