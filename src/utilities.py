from pycromanager import Acquisition
import numpy as np

def read_image(core):
    core.snap_image()
    tagged_image = core.get_tagged_image()
    pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
    return pixels

def get_slm_info(core):
    slm_name = core.get_slm_device()

    height = core.get_slm_height(slm_name)
    width = core.get_slm_width(slm_name)

    return slm_name, height, width


def send_slm_image(core, slm_name, image):
    slm_dims = (core.get_slm_height(slm_name), core.get_slm_width(slm_name))
    assert image.dtype == np.uint8, "expected image data array as uint8, got {}".format(image.dtype)
    assert image.shape == slm_dims, "expected image dimensions to match slm {}, got {}".format(slm_dims, image.shape)

    core.set_slm_image(slm_name, image.flatten())
    core.display_slm_image(slm_name)
