from celestial_bodies.celestial_body import CelestialBody
from PIL import (
    Image,
    ImageChops,
)
import numpy as np


class Interior(CelestialBody):
    """
    The base class for an interior. Can be used for any interior that has no animated
    elements. Otherwise subclassed for an animate interior
    """

    def __init__(self, vista, file_path, image_name = "observation_windows.png", invert_colors=False):
        super().__init__(vista)
        RNG = np.random.default_rng()
        self.bgr = RNG.permutation(range(3))
        if invert_colors:
            self.invert_color = RNG.integers(2, size=3)
        else:
            self.invert_color = (0, 0, 0)
        self.file_path = file_path
        self.image_name = image_name
        self.activate_camera()

    def image_path(self):
        return self.file_path / self.image_name

    def activate_camera(self):
        """
        A catch-all method for the image processing and setup done to the interior's
        image file(s) before drawing. In this base class it just recolors the image.
        """
        self.im = self.recolor(Image.open(self.image_path()))

    def recolor(self, im):
        """
        This method takes the interior's image and recolors it to match the palette
        of the rest of the Vista.

        Returns recolored PIL.image
        """
        # Convert image to RGB mode so we can quantize the colors to the palette.
        *rgb, a = im.split()
        reorder = []
        for i, c in zip(self.invert_color, self.bgr):
            if i:
                reorder.append(ImageChops.invert(rgb[c]))
            else:
                reorder.append(rgb[c])
        recolor_im = Image.merge("RGB", reorder)
        recolor_im = recolor_im.quantize(
            colors=32,
            palette=self.vista.palette.get_image(),
            method=Image.FASTOCTREE,
            dither=0,
        )
        # Convert back and re-establish the alpha channel through some hoops
        im = Image.merge("RGBA", (*recolor_im.convert("RGB").split(), a))
        return im

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the interior image over everything else in the frame.
        """
        image.alpha_composite(self.im)
