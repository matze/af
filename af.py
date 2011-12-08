import numpy as np
import scipy.ndimage
import glob
from PIL import Image


class FocusPoint(object):
    """A point associated with a region that should be focused."""

    def __init__(self, x, y, width, height):
        """TODO: Construct a focus point"""
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    @property
    def region(self):
        """Return a region defined by the rectangle x0, y0,
            x1, y1."""
        return (self._x, self._y, \
                self._x + self._width, self._y + self._height)

    def __str__(self):
        return "[%i:%i, %i:%i]" % self.region

def create_cos_window(w, h):
    ww = np.cos(np.linspace(-np.pi / 2, np.pi / 2, w).reshape((-1, 1)))
    hw = np.cos(np.linspace(-np.pi / 2, np.pi / 2, h).reshape((1, -1)))
    return np.dot(ww, hw)


def cost_stddev(image):
    """Compute the sharpness of the image as the standard deviation of the
    image."""
    return 1.0 / np.std(image) / ((np.max(image) - np.min(image)) / 2)


def cost_sobel(image):
    """ Compute the sharpness as sum of sobel detected edge intensities."""
    sx = scipy.ndimage.filters.sobel(image, axis=0, mode='constant')
    sy = scipy.ndimage.filters.sobel(image, axis=1, mode='constant')
    return np.sum(np.hypot(sx, sy))


def cost_abs_gradient(image):
    """Compute the sharpness as sum of absolute gradient intensities."""
    d = np.roll(image, -1, axis=1)
    return np.sum(np.abs(image[:,:-1] - d[:,:-1]))


def cost_squared_gradient(image):
    """Compute the sharpness as sum of squared gradient intensities."""
    d = np.roll(image, -1, axis=1)
    r = image[:,:-1] - d[:,:-1]
    return np.sum(r * r)


def cost_frequencies(image):
    """ Compute the sharpness of the image as the weighted sum of all
    frequencies."""
    w, h = image.shape
    window = create_cos_window(w, h)
    F_image = np.fft.fft2(image * window)
    F_lower = F_image[w / 2:, :]

    # create weights
    w1 = np.linspace(-w / 2, w / 2, w).reshape(1, w)
    w2 = np.linspace(0, w / 2, w / 2).reshape(w / 2, 1)
    w1 = np.repeat(w1, w / 2, axis=0)
    w2 = np.repeat(w2, w, axis=1)
    weights = np.sqrt(w1**2 + w2**2)
    return np.sum(np.abs(weights * F_lower))


def discriminate(cost_fn, image_stack, focus_point, image1, image2):
    """Compare the sharpness of images indexed by `image1` and `image2` into
        `image_stack` by using `cost_fn` and return the index with higher
        sharpness.
    """
    x0, y0, x1, y1 = focus_point.region
    im1 = image_stack[x0:x1, y0:y1, image1]
    im2 = image_stack[x0:x1, y0:y1, image2]
    return image1 if cost_fn(im1) > cost_fn(im2) else image2


def optimize(image_stack, focus_point, cost_fn):
    """Return the index of the presumably sharpest image of the `image_stack`
    using the region defined by `focus` and the cost function `cost_fn`"""
    x0, y0, x1, y1 = focus_point.region
    window = create_cos_window(y1 - y0, x1 - x0)
    J = np.zeros((image_stack.shape[2]))
    for i in range(J.shape[0]):
        J[i] = cost_fn(window * image_stack[y0:y1,x0:x1,i])
    return J


def load_image_stack(pathname_pattern):
    """Load images from disk using Unix style pathname pattern expansion.

    It returns an 3-dimensional array-like in [row, column, z] format
    """
    filenames = glob.glob(pathname_pattern)
    filenames.sort()
    stack = None
    for filename in filenames:
        img = np.array(Image.open(filename))
        if stack is not None:
            stack = np.dstack((stack, img))
        else:
            stack = img

    return stack


if __name__ == '__main__':
    pass
