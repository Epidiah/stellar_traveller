from .interior import Interior
from PIL import Image, ImageChops, ImageFont, ImageDraw, ImageFilter
import numpy as np
from itertools import cycle

RNG = np.random.default_rng()

class ExtraVehicularActivity(Interior):
    """
    The exterior interior, a view from inside a space helmet as our viewer floats
    outside the craft.
    """

    def __init__(self, vista, file_path):
        super().__init__(vista, file_path, "bubble_helmet.png")
        self.fog = fog_like_alpha(self.vista.size, [180, 396, 540, 756])
        self.font_color = vista.palette.get_color(1, 0)
        self.font = ImageFont.truetype(".fonts/HP-15C_Simulator_Font.ttf", 4)
        self.lines = self.readout()
        self.on_screen_text = [next(self.lines) for trash in range(4)]

    def readout(self):
        """
        Provides a vital message to be printed on the space helmet's readout.
        """
        lines = [
            line.ljust(23) + "\n"
            for line in (
                "Swords Without Master",
                "is in",
                "Worlds Without Master,",
                "issue 3",
                " --Epidiah Ravachol",
            )
        ]
        return cycle(lines)

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the interior image over everything else in the frame, including animating
        the fog from the astronaut's breath on the helmet glass and the helment's text
        display.
        """
        fog = ImageChops.multiply(
            Image.new(
                "L",
                image.size,
                color=int(np.ceil(70 + 60 * np.cos(frame_n / 120 * np.pi))),
            ),
            self.fog,
        )
        breath = Image.new("RGBA", image.size, "white")
        breath.putalpha(fog)
        image.alpha_composite(breath)
        image.alpha_composite(self.im)
        # Print text between screen coordinates (168, 372) and (190, 394)
        if (frame_n % 120) == 0:
            self.on_screen_text = self.on_screen_text[1:] + [next(self.lines)]
            self.leading_lines = "".join(self.on_screen_text[:3])
            self.last_line = self.on_screen_text[3]
        drawing_frame.multiline_text(
            (302, 675),
            text=self.leading_lines + self.last_line[: (frame_n % 120) // 5],
            fill=self.font_color,
            font=self.font,
            spacing=2,
            align="left",
        )

def fog_like_alpha(full_size, box, color=None):
    # Make some noise!
    haze = Image.effect_noise(
        [box[2] - box[0], box[3] - box[1]],
        RNG.integers(290, 311),
    )
    haze = ImageChops.multiply(haze, haze)
    rad = Image.radial_gradient("L").resize(haze.size)
    haze = ImageChops.subtract(haze, rad)
    # Now let's shape that noise
    mask = Image.new("L", haze.size, 0)
    drw_mask = ImageDraw.Draw(mask)
    drw_mask.ellipse((0, 0) + haze.size, "white")
    fog = Image.new("L", full_size, color="black")
    fog.paste(haze, box=box, mask=mask)
    fog = fog.effect_spread(60).filter(ImageFilter.GaussianBlur(5))
    return fog