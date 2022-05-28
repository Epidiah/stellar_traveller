from .interior import Interior
from PIL import Image

class AstroGarden(Interior):
    """
    Provides small animation capabilities for the series of astrogarden images that
    include slowly blinking lights along the path.
    """

    def __init__(self, vista, file_path):
        super().__init__(vista, file_path, "astrogarden.png")

    def activate_camera(self):
        """
        The method for assembling the film_strip of interior images and ensuring each
        image matches the Vista's palette.
        """
        file_name, ext = self.image_name.split(".")
        self.film_strip = [
            self.recolor(Image.open(self.file_path / f"{file_name}{n}.{ext}"))
            for n in range(3)
        ]

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the interior image over everything else in the frame, advancing through
        the film_strip at regular intervals.
        """
        blink = frame_n % 96
        if blink < 24:
            im = self.film_strip[0]
        elif 48 <= blink < 72:
            im = self.film_strip[2]
        else:
            im = self.film_strip[1]
        image.alpha_composite(im)

