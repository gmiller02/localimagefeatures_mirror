import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
import math

from skimage.util.dtype import img_as_float32


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_interest_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    xs = np.zeros(1)
    ys = np.zeros(1)

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    g_x = filters.sobel_v(image)
    g_y = filters.sobel_h(image)

    gx = np.square(g_x)
    gy = np.square(g_y)
    xy = np.multiply(gx, gy)

    # STEP 2: Apply Gaussian filter with appropriate sigma
    gx = filters.gaussian(gx, sigma = 1)
    gy = filters.gaussian(gy, sigma = 1)
    gxy = filters.gaussian(xy, sigma=1)

    g2 = np.square(gxy)
    a = 0.05
    # STEP 3: Calculate Harris cornerness score for all pixels.
    cornerness = (np.multiply(g_x, g_y) - g2) - (a * np.square(np.add(gx, gy)))
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    max_m = feature.peak_local_max(cornerness, min_distance= 1, threshold_rel=0.03)
    xs = max_m[:, 1]
    ys = max_m[:, 0]
    
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # STEP 2: Decompose the gradient vectors to magnitude and direction.
    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.
    
    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.

    # This is a placeholder - replace this with your features!

    features = np.zeros((len(x), 128))

    gx = filters.sobel_v(image, mask=None)
    gy = filters.sobel_h(image, mask=None)

    mag = np.sqrt(np.add(np.square(gx), np.square(gy)))

    grad_o = np.add(np.arctan2(gx, gy), np.pi)

    for i in range(0, len(x)):
        des = np.array([])
        if (x[i] + 8 < image.shape[1]) and (y[i] + 8 < image.shape[0]):
            for outerY in range(int(y[i]) - 8, int(y[i]) + 8, 4):
                for outerX in range(int(x[i]) - 8, int(x[i]) + 8, 4):
                    histogram = np.zeros((8, 1))
                    for innerX in range(outerX + 4, outerX):
                        for innerY in range(outerY + 4, outerY):
                            orientation = grad_o[innerX][innerY]
                            mag_help = mag[innerX][innerY]
                            if (orientation >= 0) and (orientation <= 1/4 * np.pi):
                                histogram[0] += mag_help
                            elif (orientation > 1/4 * np.pi) and (orientation < 1/2 * np.pi):
                                histogram[1] += mag_help
                            elif (orientation > 1/2 * np.pi) and (orientation < 3/4 * np.pi):
                                histogram[2] += mag_help
                            elif (orientation > 3/4 * np.pi) and (orientation < np.pi):
                                histogram[3] += mag_help
                            elif(orientation > np.pi) and (orientation < 5/4 * np.pi):
                                histogram[4] += mag_help
                            elif(orientation > 5/4 * np.pi) and (orientation < 3/2 * np.pi):
                                histogram[5] += mag_help
                            elif(orientation > 3/2 * np.pi/2) and (orientation < 7/4 * (np.pi)):
                                histogram[6] += mag_help
                            elif (orientation > 7/4 * np.pi) and (orientation < np.pi):
                                histogram[7] += mag_help 

                    des = np.append(des, histogram) 

            features[i, :] = np.expand_dims(des / np.linalg.norm(des), axis = 0)

    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: https://browncsci1430.github.io/webpage/hw2_featurematching/efficient_sift/
    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.
    
    # BONUS: Using PCA might help the speed (but maybe not the accuracy).

    matches = np.zeros((len(im1_features),2))
    confidences = np.zeros(len(im1_features))

    B = 2 * (np.dot(im1_features, np.matrix.transpose(im2_features)))

    f1_sum = np.sum(np.square(im1_features), axis=1, keepdims=True)
    f2_sum = np.sum(np.square(im2_features), axis=1, keepdims=True)

    f2_sum = np.matrix.transpose(f2_sum)

    A = np.add(f1_sum, f2_sum)
    
    e_dist = np.sqrt(np.subtract(A, B))

    dist_sort = np.sort(e_dist, axis=1)
    d_sort_i = np.argsort(e_dist, axis = 1)

    near_n = dist_sort[:, 0]
    near_n_2 = dist_sort[:, 1]

    confidences = 1 - (near_n/near_n_2)

    for i in range(len(im1_features)):
        matches[i][0] = i
        matches[i][1] = d_sort_i[i][0]

    return matches, confidences
