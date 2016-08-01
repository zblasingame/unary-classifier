# Created By Zander Blasingame
# For CAMEL at Clarkson University
# Collection of helper functions to reduce clutter in main file

import numpy as np


# parses svm data returns tuple of inputs X and labels Y
def parse_svm(filepath):
    X = []
    Y = []
    vec_length = 0
    isFirstLine = True
    with open(filepath) as f:
        for line in f.readlines():
            isFirst = True
            tmpStr = ''
            isValue = False
            x = []
            y = []

            for char in line:
                if isFirst:
                    if char == ' ':
                        isFirst = False
                        y.append(int(tmpStr))
                        tmpStr = ''
                    else:
                        tmpStr += char
                else:
                    if char == ' ' or char == '\n':
                        if tmpStr != '':
                            x.append(float(tmpStr))
                        tmpStr = ''
                        isValue = False
                    elif char == ':':
                        isValue = True
                    elif char != '\n':
                        if isValue:
                            tmpStr += char

            # normalize x
            # length = reduce(lambda a, b: a+b, map(lambda y: y**2, x))**0.5
            # x = map(lambda y: abs(y) / length, x)  # trying to absolute value

            # safety check assumes first input is correct length
            if isFirstLine:
                vec_length = len(x)
                isFirstLine = False

            if vec_length == len(x):
                X.append(x)
                Y.append(y)

    return X, Y


# Finds the bounds of subspace
def find_subspace_bounds(dists):
    sigma = np.std(dists, axis=0)
    mu = np.mean(dists, axis=0)
    norm = np.array([sigma[1], -sigma[0]]) / ((sigma[0]**2 + sigma[1]**2)**0.5)
    xi = np.std(np.asarray(dists).dot(norm)) * norm

    return sigma, xi, mu


# Returns the number of points in bounds
def check_bounds(sigma, xi, mu, points):
    RANGE = 2

    alpha = sigma / ((sigma[0]**2 + sigma[1]**2)**0.5)
    beta = xi / ((xi[0]**2 + xi[1]**2)**0.5)

    beta_bounds = [np.dot(mu - RANGE*xi, beta),
                   np.dot(mu + RANGE*xi, beta)]

    alpha_bounds = [np.dot(mu - RANGE*sigma, alpha),
                    np.dot(mu + RANGE*sigma, alpha)]

    def in_bounds(point):
        pb = np.dot(point, beta)
        pa = np.dot(point, alpha)

        if (
            beta_bounds[0] <= pb <= beta_bounds[1] and
            alpha_bounds[0] <= pa <= alpha_bounds[1]
        ):
            return True
        else:
            return False

    return reduce(lambda a, b: a + (1 if in_bounds(b) else 0), points, 0)
