from .planetary_feature import PlanetaryFeature
from functools import reduce
from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
)
import numpy as np

class Clouds(PlanetaryFeature):
    """
    A static PlanetaryFeature that approximates great bands of clouds like those found
    on gas giants, as well as a few stable elliptical storms.
    """

    def __init__(self, planet):
        super().__init__(planet)
        self.n_clouds = self.planet.vista.coords.integers(
            1, max(1, int(np.log(self.planet.mass))) + 2
        )
        if self.planet.is_planetoid:
            self.planet.vista.status_dict[f"Planet {self.planet.layer}"][
                "cloud bands"
            ] = (self.n_clouds * 2 + 1)
            self.planet.vista.status_dict[f"Planet {self.planet.layer}"][
                "expected storms"
            ] = int(np.log10(self.planet.width))
        self.cloud_size = (
            self.planet.mass,
            self.planet.mass // max(1, (self.n_clouds - 2)),
        )
        if self.planet.colorful:
            self.cloud_colors = self.planet.vista.palette.get_color(
                shade=self.planet.shade
            )
        else:
            self.cloud_colors = self.planet.vista.palette.get_color(hue=self.planet.hue)
        self.cloud_colors = [
            cc + (255,) for cc in self.cloud_colors if cc != self.planet.fill
        ]

    def foreground(self, center, im, draw_im, planet_mask):
        """
        Called once to draw roughly self.n_clouds number of cloud bands across
        the surface of the planet's image.
        """
        surface = im.copy()
        clouds = [
            (
                n,
                nebula_like(
                    self.planet.vista.coords,
                    foreground_color=tuple(
                        self.planet.vista.coords.choice(self.cloud_colors).tolist()
                    ),
                    box=self.cloud_size,
                    background_color=self.planet.fill + (255,),
                    ensure_wrap=False,
                    eddies=True,
                    eddy_colors=self.cloud_colors,
                ),
            )
            for n in range(self.n_clouds)
        ]
        for n, cloud in clouds:
            left_x, top_y = center - self.planet.mass // 2
            surface.alpha_composite(
                cloud, dest=(left_x, top_y + self.cloud_size[1] * n)
            )
        im.paste(surface, (0, 0), mask=planet_mask)
    

def nebula_like(
    coords,
    foreground_color,
    box=(240, 150),
    background_color=(0, 0, 0, 0),
    ensure_wrap=False,
    eddies=False,
    eddy_colors=None,
):
    sketch = Image.new("RGBA", box, (0, 0, 0, 0))
    sketcher = ImageDraw.Draw(sketch)
    width, height = box[0], box[1]
    for level in (height // 3, height // 3 * 2):
        attempts = 7
        while attempts:
            try:
                attempts -= 1
                xs = np.linspace(1, width, 8, endpoint=False)
                # add caps to ensure the nebula behaves at the ends and wraps around
                xs = (
                    np.linspace(-width, 0, 8).tolist()
                    + [x + coords.integers(-width // 24, width // 24) for x in xs]
                    + np.linspace(width, width * 2, 8).tolist()
                )
                ys = (
                    [level] * 8
                    + [coords.normal(level, height // 36) for x in xs[8:-8]]
                    + [level] * 8
                )
                poly = np.linalg.solve(np.vander(xs), ys).tolist()
                break
            except np.linalg.LinAlgError:
                continue
        else:
            return sketch
        points = [
            (x, reduce(lambda a, b: x * (a + b), poly[:-1], 0) + poly[-1])
            for x in range(width)
        ]
        sketcher.line(points, fill=foreground_color, joint="curve", width=2)
    for x in range(1, width + 2, width // 2):
        ImageDraw.floodfill(sketch, (x, int(height / 2)), foreground_color)
    # Adding eddies and thousand-year-old storms
    if eddies:
        if eddies is True:
            eddies = coords.integers(int(np.log10(width)) + 1)
        for ed in range(eddies):
            if eddy_colors is None:
                eddy_color = foreground_color
            else:
                eddy_color = tuple(coords.choice(eddy_colors).tolist())
            zone = width // eddies
            ed_height = coords.integers(height // 6, height // 3)
            ed_width = coords.integers(zone // 6, zone // 3)
            if ed_height > ed_width:
                ed_width, ed_height = ed_height, ed_width
            x = coords.integers(zone * ed, max(2, zone * (1 + ed) - ed_width))
            y = coords.integers(
                height // 3 - ed_height // 2, height // 3 * 2 - ed_height // 2
            )
            sketcher.ellipse(
                [x, y, x + ed_width, y + ed_height],
                fill=eddy_color,
                outline=background_color,
                width=coords.integers(0, int(np.log10(width)) + 2),
            )
    # Let's smooth it out a bit
    sketch = sketch.filter(ImageFilter.SMOOTH)
    return sketch