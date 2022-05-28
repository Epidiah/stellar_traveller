from .interior import Interior
from PIL import Image, ImageChops, ImageDraw, ImageFilter
import numpy as np

RNG = np.random.default_rng()

class StellarCafe(Interior):
    """
    A view from inside a on ship cafe. Includes animation for steam rising from mugs.
    """

    def __init__(self, vista, file_path):
        super().__init__(vista, file_path, "stellar_cafe.png")
        self.steam_box_left = [236, 520, 262, 580]
        self.steam_box_right = [423, 533, 450, 594]
        self.left_mug = RNG.choice([True, False], 2)
        self.right_mug = RNG.choice([True, False], 2)
        self.steam = self.steam_cycle()

    def generate_mug_steam(self):
        """
        Creates a snapshot of steam that rising from the mugs.
        """
        steam_left = flame_like_alpha(self.im.size, self.steam_box_left, *self.left_mug)
        steam_right = flame_like_alpha(
            self.im.size, self.steam_box_right, *self.right_mug
        )
        steam = ImageChops.lighter(steam_left, steam_right)
        steaming = Image.new("RGBA", self.vista.size, "white")
        steaming.putalpha(steam)
        return steaming

    def steam_cycle(self):
        first_steaming = next_steaming = self.generate_mug_steam()
        for n in range(self.vista.length - 12):
            if n % 12 == 0:
                prev_steaming, next_steaming = next_steaming, self.generate_mug_steam()
            yield Image.blend(prev_steaming, next_steaming, (n % 12 / 12))
        for n in range(12):
            yield Image.blend(next_steaming, first_steaming, n / 12)

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the interior image over everything else in the frame, including animating
        the steam pouring off of two hot beverages.
        """
        current_im = self.im.copy()
        current_im.alpha_composite(next(self.steam))
        image.alpha_composite(current_im)

def flame_like_alpha(full_size, box, left_curl=False, whisp=False):
    # Make some noise!
    haze = Image.effect_noise(
        [box[2] - box[0], box[3] - box[1]],
        RNG.integers(290, 311),
    )
    x, y = haze.size
    # Now let's shape that noise so it vaguely fits the silhouette of fire
    drw_haze = ImageDraw.Draw(haze)
    drw_haze.ellipse(
        [x * left_curl - x // 2, -y / 8, x * left_curl + x // 2, y * 5 / 8],
        "black",
    )
    if whisp:
        drw_haze.ellipse(
            [
                x * (not left_curl) - x // 2,
                y * 3 / 8,
                x * (not left_curl) + x // 2,
                y * 9 / 8,
            ],
            "black",
        )
    shifts = RNG.integers(-4, 5, 4)
    mask = Image.new("L", haze.size, "black")
    drw_mask = ImageDraw.Draw(mask)
    drw_mask.ellipse(((0, 0) + haze.size + shifts).tolist(), 128)
    flames = Image.new("L", full_size, color="black")
    flames.paste(haze, box=box, mask=mask)
    # Now spread it around and blur it!
    flames = flames.effect_spread(3).filter(ImageFilter.GaussianBlur(3))
    return flames