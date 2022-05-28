from functools import partial
import numpy as np
import pandas as pd
from PIL import ImageColor, Image, ImageDraw

# COLOR_NAMES are not currently in use, but may return in the future.
COLOR_NAMES = pd.read_csv("palette/colors.csv", squeeze=True, names=["color"])

class Palette:
    """
    A Palette object holds all the colors that should show up in the animation.
    Hue, brightest = 0, darkest = -1
    Shade, brightest = 0, darkest = -1
    """

    def __init__(self, coords, palette=None, hue_depth=6, shade_depth=5):
        self.coords = coords
        self.hue_depth = hue_depth
        self.shade_depth = shade_depth
        if palette is None:
            self.palette = self.random_palette(hue_depth=6)
        elif type(palette) == pd.Series:
            self.palette = self.fill_shades(self.sort_palette(palette).to_frame(name=0))
        elif type(palette) == pd.DataFrame:
            self.palette = palette
        self.n_colors = int(2 ** np.ceil(np.log(self.palette.size + 2) / np.log(2)))

    def random_palette(self, hue_depth=6):
        """
        Returns a pd.DataFrame with colors as the rows, sorted from brightest (hue=0)
        to darkest (hue=hue_depth-1), and shades of the color as the columns,
        sorted from brightest (shade=0) to darkest (shade=shade_depth-1).
        """
        palette = pd.Series(
            self.coords.choice(COLOR_NAMES, hue_depth, replace=False)
        ).apply(ImageColor.getrgb)
        palette = self.sort_palette(palette)
        palette = palette.to_frame(name=0)
        return self.fill_shades(palette)

    def sort_palette(self, palette):
        """
        Sorts the palette hues by some arbitrary definition of brightness.
        """

        def color_sort_key(colors):
            lums = (0.2126, 0.7152, 0.0722)
            return colors.apply(lambda x: sum(c * l for c, l in zip(x, lums)))

        return palette.sort_values(
            ignore_index=True, key=color_sort_key, ascending=False
        )

    def fill_shades(self, palette):
        """
        Takes a pd.Series of colors in RGB tuple and returns a pd.DataFrame of
        the full palette of colors with the columns representing the shades.
        """

        def color_dive(color, *, s=0):
            factor = 1 - s * 0.9 / (self.shade_depth)
            return tuple(int(c * factor) for c in color)

        for shade in range(1, self.shade_depth):
            palette.loc[:, shade] = palette.loc[:, (shade - 1)].apply(
                partial(color_dive, s=shade)
            )
        return palette

    def get_color(self, hue=None, shade=None):
        """
        If an int is provided for hue, returns a list of shades in that hue.
        If an int is provided for shade, returns a list of hues at that shade level.
        If ints are provided for both, returns the color at the intersect of that
        hue and shade.
        """
        if hue is not None and shade is not None:
            return self.palette.iloc[hue, shade]
        elif hue is not None:
            return self.palette.iloc[hue, :].tolist()
        elif shade is not None:
            return self.palette.iloc[:, shade].tolist()
        else:
            raise ValueError("Must have `hue` or `shade` or both, but not neither.")

    def get_image(self):
        """
        Returns an image with all the hues and shades of the palette, useful for
        recoloring outside art.
        """
        h, s = self.palette.shape
        plt_im = Image.new("RGBA", (200 * (h + 1), 200 * s), color="black")
        plt_drw = ImageDraw.Draw(plt_im)
        for x, hue in enumerate(self.palette.itertuples(index=False, name=None)):
            for y, (r, g, b) in enumerate(hue):
                plt_drw.rectangle(
                    [200 * (x + 1), 200 * y, 200 * (x + 2), 200 * (y + 1)],
                    fill=(r, g, b),
                )
        plt_drw.rectangle([0, 0, 200, 100 * s], fill=(255, 255, 255))
        return plt_im.convert("P")
