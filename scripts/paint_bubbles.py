import numpy as np
import cv2
from skimage.feature import canny
from numpy.linalg import lstsq
import scipy.ndimage as ndi


def polyval2(p,shape):

    n,m = shape

    x,y = np.meshgrid(np.arange(m),np.arange(n))

    arrfit = p[0]*np.ones(shape) + p[1]*x + p[2]*y +\
        p[3]*x**2 + p[4]*x*y + p[5]*y**2

    return arrfit

def polyfit2(x,y,z):
    N = len(x)

    matrix = np.vstack(\
                    [np.ones(N),x,y,x**2,x*y,y**2]\
                        ).T

    lstsq_result = lstsq(matrix,z,rcond=None)

    p = lstsq_result[0]

    return p

def equalize_image(im, sigma, background=None):
    if background is not None:
        #remove global uneveness in the image
        im = im/background

    im_edges = canny(im,sigma,low_threshold = 0,high_threshold = 0.01)

    good_edges = im_edges
    iedge,jedge = np.where(good_edges)
    p = polyfit2(jedge,iedge, im[good_edges==1])


    return im/polyval2(p,im.shape)

def threshold_single_image(im):
    smooth_im = ndi.gaussian_filter(im,sigma=1)
    thresh=1

    BW = smooth_im>thresh

    return BW

def make_bubble_ramps(im, image_background=None, invert_BW=False, sigma_for_edges=2):
    # divide the image by the background
    # and make the signal even

    im_eq = equalize_image(im,sigma_for_edges, background=image_background)

    BW = threshold_single_image(im_eq)

    # print("Painting bubbles")
    if invert_BW:
        BW = np.invert(BW)
    img_height = BW.shape[1]

    # Image.fromarray(BW).save("BW.tif")
    ret, markers = cv2.connectedComponents(np.uint8(np.asarray(BW)))
    # print(np.unique(markers))

    # Get the markers of regions along the edges of the image
    edge_region_nums = np.unique(np.concatenate([
        markers[0,:], markers[-1,:], markers[:,0], markers[:,-1]]))
    # # Skip these, we should be left with the bubbles within the image
    # region_indices = [i for i in region_indices if i not in edge_bubble_nums]

    bubble_paint = np.zeros(BW.shape, dtype=np.uint8)
    MAX_BRIGHTNESS = np.iinfo(bubble_paint.dtype).max

    # Make ramping brightnesses for each bubble
    for i in range(0, np.max(markers)):
        # print(i)
        # Skip regions bordering image edgee
        if i in edge_region_nums:
            print("Skipping section that touches image border")
            continue

        # Binary image 1 in bubble 0 evereywhere else
        bubble_bin_mask = (markers == i)
        # Coordinates of pixels in bubble
        region_border_coords_xy = np.where(bubble_bin_mask)
        # Only want regions of the correct value in BW - either zero or one
        # depending on whether the region fluid is labeled

        bubble_start_x = np.min(region_border_coords_xy[1])
        bubble_end_x = np.max(region_border_coords_xy[1])
        ramp = np.linspace(0,MAX_BRIGHTNESS, bubble_end_x - bubble_start_x,dtype=bubble_paint.dtype)
        ramp = np.tile(ramp, (1,img_height))
        # print("Ramp shape: ")
        ramp_mask = np.zeros_like(bubble_paint)
        ramp_mask[bubble_start_x:bubble_end_x,:] = ramp

        bubble_paint += ramp_mask * bubble_bin_mask