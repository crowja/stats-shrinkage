#! /usr/bin/env python3

import numpy as np


def stein(x, n, rtol=np.finfo(float).eps):
    """Stein's estimators of sample covariance eigenvalues."""

    """
    Compute adjusted eigenvalues of a sample covariance matrix using Stein's
    method. If S is the sample covariance matrix with spectral decomposition
    Q diag(x) Q^T, where Q is orthogonal and x represents the eigenvalues
    in descending order, the adjusted eigenvalues y are intended to make
    Q diag(y) Q^T an improved estimator of the true covariance.

    Initial weights for adjusting the eigenvalues are then subject to Stein's
    isotonization algorithm to ensure the final eigenvalues are positive and
    descending.
    
    ARGUMENTS
    x     nonnegative eigenvalues sorted in descending order.
    n     number of samples.
    rtol  relative tolerance for setting small eigenvalues to zero.

    RETURNS
    The call y = stein_isonitize(x, n, rtol) returns
    y     adjusted eigenvalues.

    John A. Crow <crowja@gmail.com>
    """

    class EigenvalueCorrection:
        def __init__(self):
            self.val = 0.0
            self.alpha = 0.0
            self.block_start = False
            self.block_end = False

    adjusts = []  # array of adjustments
    for i in range(len(x)):
        evcorr = EigenvalueCorrection()
        evcorr.block_start = i
        evcorr.block_end = i
        adjusts.append(evcorr)

    # Get the effective rank, zero the small eigenvalues.
    rank = 0
    total_mass = x.sum()
    cumul_mass = 0.0
    for i in range(len(x)):
        if np.isclose(cumul_mass, total_mass, rtol=rtol, atol=0.0):
            break
        rank += 1
        cumul_mass += x[i]
        adjusts[i].val = x[i]

    # Compute the initial alphas.
    for i in range(rank):
        s = 0.0
        for j in range(rank):
            # TODO maybe replace the next line with "if np.isclose(x[j], x[i])"
            if i == j:
                continue
            s += 1.0 / (x[i] - x[j])
        adjusts[i].alpha = 0 * (n - rank) + 1 + 2 * x[i] * s  # TODO

    # ISOTONIZATION STEP 1: Coerce the alphas to be positive.

    while True:
        done = True
        for i in range(len(adjusts) - 1, 0, -1):
            if adjusts[i].alpha <= 0.0:
                # Pool i - 1 and i into i.
                adjusts[i - 1].alpha += adjusts[i].alpha
                adjusts[i - 1].val += adjusts[i].val
                adjusts[i - 1].block_end = adjusts[i].block_end
                del adjusts[i]
                done = False
                break
        if done:
            break

    # ISOTONIZATION STEP 2: Ensure the val / alpha list is decreasing.

    # A "violating pair" is one for which val[i]/alpha[i] > val[i+1]/alpha[i+1].
    while True:
        done = True
        for i in range(len(adjusts) - 2, -1, -1):
            if (
                adjusts[i].val * adjusts[i + 1].alpha
                < adjusts[i + 1].val * adjusts[i].alpha
            ):
                # i and i + 1 are a violating pair, pool into i.
                adjusts[i].alpha += adjusts[i + 1].alpha
                adjusts[i].val += adjusts[i + 1].val
                adjusts[i].block_end = adjusts[i + 1].block_end
                del adjusts[i + 1]
                done = False
                break
        if done:
            break

    # ISOTONIZATION STEP 3: Distribute block-pooled values.

    # In Steps 1 and 2, adjustments were made by pooling adjacent values. Now
    # assign these block values to each of the original positions.
    y = np.zeros_like(x)
    for i in range(len(adjusts)):
        for j in range(adjusts[i].block_start, adjusts[i].block_end + 1):
            y[j] = adjusts[i].val / adjusts[i].alpha

    return y


if __name__ == "__main__":

    ##x = np.sort(np.random.rand(9))[::-1]
    ##x = np.array([0.83078802, 0.81182107, 0.75951461, 0.71773099, 0.22039148])
    """
    Example from S. P. Lin and M. D. Perlman, "A monte carlo comparison of four
    estimators of a covariance matrix," Technical Report 44, Department of
    Statistics, University of Washington, April 1984.
    """
    n = 10
    x = np.array([50.91, 39.66, 14.96, 11.88, 8.55, 6.33, 2.40, 1.99, 0.52, 0.01])

    y = stein(x, 10, 0.0)

    mass_x = 0.0
    mass_y = 0.0
    for i in range(len(x)):
        mass_x += x[i]
        mass_y += y[i]
        print(f"{x[i] :0.2f}\t{y[i] :0.2f}")
    print("==")
    print(f"{mass_x :0.2f}\t{mass_y :0.2f}")