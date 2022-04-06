from pycromanager import Bridge
import numpy as np
from scripts.utilities import send_slm_image, get_slm_info, read_image
from scripts.paint_bubbles import make_bubble_ramps



if __name__ == "__main__":
    main()

def main():
    camera_offset_rel_to_slm = (0,0)
    camera_slm_zoom_ratio = 1

    #

    with Bridge() as bridge:
        core = bridge.get_core()

        slm_name, slm_height, slm_width = get_slm_info(core)

        microscope_image = read_image(core)

        bubble_ramp_image = make_bubble_ramps(microscope_image)
        assert bubble_ramp_image.dtype == np.uint8, "Bubble ramp image should be 8 bit to work right with PIL Image"
        bubble_ramp_pil_img = Image.fromarray(bubble_ramp_image, 'L')

        project_image = Image.new('L', (slm_height, slm_width))
        project_image = project_image.resize([int(camera_slm_zoom_ratio * s) for s in im.size])
        project_image.paste(bubble_ramp_image, camera_offset_rel_to_slm)

        send_slm_image(create_slm_mask)
