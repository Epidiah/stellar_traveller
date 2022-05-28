from .palette import Palette
import pandas as pd
from PIL import ImageColor

class ComplementaryPalette16(Palette):
    def __init__(self, palette=None, hue_depth=3, shade_depth=4):
        super().__init__(palette=None, hue_depth=3, shade_depth=4)

    def random_palette(self, hue_depth=3):
        base_hue = self.vista.coords.integers(0, 360)
        palette = pd.Series(
            [ImageColor.getrgb(f"hsl({base_hue}, {sat}%, 50%)") for sat in (20, 50)]
            + [
                ImageColor.getrgb(f"hsl({(base_hue+180)%360}, 50%, 50%)"),
            ]
        )
        palette = self.sort_palette(palette)
        palette = palette.to_frame(name=0)
        return self.fill_shades(palette)