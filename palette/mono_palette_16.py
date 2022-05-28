from .palette import Palette
import numpy as np
import pandas as pd
from PIL import ImageColor, Image, ImageDraw

class MonoPalette16(Palette):
    def __init__(self, palette=None, hue=None, hue_depth=4, shade_depth=4):
        self.hue = hue % 360
        super().__init__(palette=None, hue_depth=4, shade_depth=4)
        self.n_colors = 16

    def random_palette(self, hue_depth=3):
        if self.hue is not None:
            return self.build_palette(self.hue)
        base_hue = self.vista.coords.integers(0, 360)
        return self.build_palette(base_hue)

    def build_palette(self, base_hue):
        gaps = np.linspace(60, 10, 4)
        palette = pd.DataFrame(
            np.reshape(
                [
                    f"hsl({base_hue}, {s+5*(4-i)}%, {l}%)"
                    for i, s in enumerate(gaps)
                    for l in gaps
                ],
                (4, 4),
            ),
            columns=range(4),
        )
        palette = palette.applymap(ImageColor.getrgb)
        return palette

    def get_image(self):
        h, s = self.palette.shape
        plt_im = Image.new("RGBA", (200 * h, 200 * s), color="black")
        plt_drw = ImageDraw.Draw(plt_im)
        for x, hue in enumerate(self.palette.itertuples(index=False, name=None)):
            for y, (r, g, b) in enumerate(hue):
                plt_drw.rectangle(
                    [200 * x, 200 * y, 200 * (x + 1), 200 * (y + 1)],
                    fill=(r, g, b),
                )
        return plt_im.convert("P")