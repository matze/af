import af
import scipy.misc
import scipy.ndimage
import numpy as np
import PIL.Image
import uuid
import os


def test_focus_region():
    fp = af.FocusPoint(100., 100., 20., 30.)
    assert fp.region[0] == 100.
    assert fp.region[1] == 100.
    assert fp.region[2] == 100. + 20.
    assert fp.region[3] == 100. + 30.


def test_cost_stddev_range():
    image = np.random.randn(100).reshape(10, 10)
    r = af.cost_stddev(image)
    assert r >= 0.0 and r <= 1.0


def test_cost_stddev():
    x = np.array([-100.0, 100.0])
    assert af.cost_stddev(x) == 1.0


def test_cost_gradient():
    x = np.array([1.0, 2.0, 5.0, 2.0]).reshape(1, 4)
    assert af.cost_gradient(x) == 7.0


def test_discriminator_result():
    fp = af.FocusPoint(0, 0, 10, 10)
    stack = np.random.randn(2 * 100).reshape(10, 10, 2)
    r = af.discriminate(af.cost_stddev, stack, fp, 0, 1)
    assert r == 0 or r == 1


def test_cost_functions():
    lena = scipy.misc.lena()
    blurred = scipy.ndimage.gaussian_filter(lena, 3.0)
    fp = af.FocusPoint(0, 0, 10, 10)
    stack = np.dstack((lena, blurred))
    assert af.discriminate(af.cost_frequencies, stack, fp, 0, 1) == 0
    assert af.discriminate(af.cost_sobel, stack, fp, 0, 1) == 0
    assert af.discriminate(af.cost_gradient, stack, fp, 0, 1) == 0


def test_optimize():
    lena = scipy.misc.lena()
    blurred1 = scipy.ndimage.gaussian_filter(lena, 3.0)
    blurred2 = scipy.ndimage.gaussian_filter(blurred1, 1.5)
    fp = af.FocusPoint(0, 0, 10, 10)
    stack = np.dstack((blurred2, np.dstack((blurred1, lena))))
    assert af.optimize(stack, fp, af.cost_sobel) == 2


def test_load_image_stack_none():
    assert af.load_image_stack("") == None


def test_load_image_stack():
    # setup two images
    ones = np.ones((256, 256), dtype=np.float)
    zeros = np.zeros((256, 256), dtype=np.float)

    random_name = str(uuid.uuid1())
    img = PIL.Image.fromarray(ones)
    img.save("%s_1.tif" % random_name)
    img = PIL.Image.fromarray(zeros)
    img.save("%s_2.tif" % random_name)

    # now test
    stack = af.load_image_stack('./%s_[0-9]*.tif' % random_name)
    assert stack.shape[2] == 2
    assert np.array_equal(stack[:,:,0], ones)
    assert np.array_equal(stack[:,:,1], zeros)

    # cleanup
    os.remove("%s_1.tif" % random_name)
    os.remove("%s_2.tif" % random_name)
