from pycromanager import Bridge
import numpy as np
from PIL import Image
from scripts.utilities import send_slm_image, get_slm_info, read_image
from scripts.paint_bubbles import make_bubble_ramps


def main():
    camera_offset_rel_to_slm = (0,0)
    camera_slm_zoom_ratio = 1

    #

    with Bridge() as bridge:
        core = bridge.get_core()

        slm_name, slm_height, slm_width = get_slm_info(core)

        microscope_image = read_image(core)
        print("Image dimensions: {}".format(microscope_image.shape))
        Image.fromarray(microscope_image).save("image.tif")


        # bubble_ramp_image = make_bubble_ramps(microscope_image)
        # assert bubble_ramp_image.dtype == np.uint8, "Bubble ramp image should be 8 bit to work right with PIL Image"
        # bubble_ramp_pil_img = Image.fromarray(bubble_ramp_image, 'L')
        # bubble_ramp_pil_img.save("data/image.tif")

        project_image = Image.new('L', (slm_height, slm_width))
        # bubble_ramp_image = bubble_ramp_image.resize([int(camera_slm_zoom_ratio * s) for s in bubble_ramp_image.size])
        # project_image.paste(bubble_ramp_image, camera_offset_rel_to_slm)

        send_slm_image(core,slm_name,project_image)


if __name__ == "__main__":
    main()