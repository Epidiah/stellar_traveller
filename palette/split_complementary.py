from .palette import Palette
import pandas as pd
from PIL import ImageColor

class SplitComplementaryPalette(Palette):
    def random_palette(self, hue_depth=6):
        base_hue = self.coords.integers(0, 360)
        support_right = (base_hue + 120) % 360
        support_left = (base_hue - 120) % 360
        palette = pd.Series(
            [
                ImageColor.getrgb(f"hsl({hue}, 70%, 50%)")
                for hue in (base_hue, support_left, support_right)
            ]
            + [
                ImageColor.getrgb(f"hsl({hue}, 50%, 60%)")
                for hue in (base_hue, support_left, support_right)
            ]
        )
        palette = self.sort_palette(palette)
        palette = palette.to_frame(name=0)
        return self.fill_shades(palette)



