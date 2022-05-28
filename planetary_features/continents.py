from .planetary_feature import PlanetaryFeature
from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
)

class Continents(PlanetaryFeature):
    """
    A static PlanetaryFeature that resembles coastlines and continents.
    """

    def __init__(self, planet):
        super().__init__(planet)
        if self.planet.colorful:
            self.land_colors = self.planet.vista.palette.get_color(
                shade=self.planet.shade
            )
        else:
            self.land_colors = self.planet.vista.palette.get_color(hue=self.planet.hue)
        self.land_colors = [
            lc + (255,) for lc in self.land_colors if lc != self.planet.fill
        ]
        self.n_continents = self.planet.vista.coords.integers(3, 13)
        if self.planet.is_planetoid:
            self.planet.vista.status_dict[f"Planet {self.planet.layer}"][
                "land masses"
            ] = self.n_continents

    def foreground(self, center, im, draw_im, planet_mask):
        """
        Called once to draw roughly self.n_continents number of continents across
        the surface of the planet's image.
        """
        surface = coastlines(
            self.planet.vista.coords,
            im,
            self.n_continents,
            self.planet.fill,
            self.land_colors,
        )
        im.paste(surface, (0, 0), mask=planet_mask)

def coastlines(coords, im, n_continents, sea_color, land_colors):
    sketch = Image.new("RGBA", im.size, sea_color)
    sketcher = ImageDraw.Draw(sketch)
    for n in range(n_continents):
        x = coords.integers(0, im.size[0])
        y = coords.integers(0, im.size[1])
        path = []
        for n in range(60 * im.size[0]):
            new_x, new_y = x, y
            while new_x == x and new_y == y:
                new_x += coords.integers(-1, 2)
                new_y += coords.integers(-1, 2)
            x, y = new_x, new_y
            path.append(x)
            path.append(y)
        if isinstance(land_colors, list):
            land_color = tuple(coords.choice(land_colors).tolist())
        else:
            land_color = land_colors
        sketcher.line(path, fill=land_color, width=2, joint="curve")
    sketch = (
        sketch.effect_spread(3)
        # .filter(ImageFilter.GaussianBlur(3))
        .filter(ImageFilter.SMOOTH)
    )
    return sketch

