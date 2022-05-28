from .palette import Palette
from functools import partial

class PastellerPalette(Palette):
    def fill_shades(self, palette):
        """
        Takes a pd.Series of colors in RGB tuple and returns a pd.DataFrame of
        the full palette of colors with the columns representing the shades.

        Generally presents a brighter more diffuse set of colors than the base palette.
        """
        palette.loc[:, 1] = palette.loc[:, 0]

        def color_jump(color):
            return tuple((c + 255) // 2 for c in color)

        palette.loc[:, 0] = palette.loc[:, 0].apply(color_jump)

        def color_dive(color, *, s=0):
            factor = 1 - s * 0.7 / (self.shade_depth)
            return tuple(int(c * factor) for c in color)

        for shade in range(2, self.shade_depth):
            palette.loc[:, shade] = palette.loc[:, (shade - 1)].apply(
                partial(color_dive, s=shade)
            )
        return palette
