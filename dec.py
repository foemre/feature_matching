import cv2 
import numpy as np

'''

Essential matrix:
[[ 3.28330923e-04, -2.23079166e-03, -2.48795548e-01],
[ 6.25281864e-04, -4.30415770e-02,  6.60711605e-01],
[-2.16119671e-02,  7.05461137e-01,  3.95019479e-02]]

Fundamental matrix:
[[-1.51097971e-05, -6.16911686e-05,  1.55952176e-02],
[ 3.46439713e-05, -2.88601180e-05, -2.93517084e-02],
[-1.12285830e-03, -3.58703048e-04,  1.00000000e+00]]

Using this equation:
E = K.T @ F @ K

Estimate the K matrix.
'''

E = np.array([[ 3.28330923e-04, -2.23079166e-03, -2.48795548e-01],
[ 6.25281864e-04, -4.30415770e-02,  6.60711605e-01],
[-2.16119671e-02,  7.05461137e-01,  3.95019479e-02]])

E2 = np.array([[ 3.28330923e-04,  6.25281864e-04, -2.16119671e-02],
       [-2.23079166e-03, -4.30415771e-02,  7.05461137e-01],
       [-2.48795548e-01,  6.60711605e-01,  3.95019479e-02]])

F = np.array([[-1.51097971e-05, -6.16911686e-05,  1.55952176e-02],
[ 3.46439713e-05, -2.88601180e-05, -2.93517084e-02],
[-1.12285830e-03, -3.58703048e-04,  1.00000000e+00]])

# Find the eigenvalues of both matrices
eigvals_E, _ = np.linalg.eig(E)
eigvals_F, _ = np.linalg.eig(F)

print('Eigenvalues of E: {}'.format(eigvals_E))
print('Eigenvalues of F: {}'.format(eigvals_F))




# Decompose the essential matrix
S = cv2.decomposeEssentialMat(E)
S2 = cv2.decomposeEssentialMat(E2)
#print(S)
print()

# The first element of the tuple is the rotation matrix. Determine the yaw, pitch, and roll from this matrix.
R = S[1]
R2 = S2[1]

# Get the yaw, pitch, and roll
yaw = np.arctan2(R[1, 0], R[0, 0])
pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
roll = np.arctan2(R[2, 1], R[2, 2])

yaw2 = np.arctan2(R2[1, 0], R2[0, 0])
pitch2 = np.arctan2(-R2[2, 0], np.sqrt(R2[2, 1]**2 + R2[2, 2]**2))
roll2 = np.arctan2(R2[2, 1], R2[2, 2])

# print the results in degrees
print('yaw: {} degrees'.format(np.rad2deg(yaw)))
print('pitch: {} degrees'.format(np.rad2deg(pitch)))
print('roll: {} degrees'.format(np.rad2deg(roll)))

print('yaw2: {} degrees'.format(np.rad2deg(yaw2)))
print('pitch2: {} degrees'.format(np.rad2deg(pitch2)))
print('roll2: {} degrees'.format(np.rad2deg(roll2)))


# Estimate the calibration matrix from the fundamental matrix
K = np.array([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])





