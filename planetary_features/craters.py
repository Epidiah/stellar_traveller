from .planetary_feature import PlanetaryFeature
from PIL import (
    ImageDraw,
)

class Craters(PlanetaryFeature):
    """
    A static PlanetaryFeature that resembles volcanic or impact craters.
    """

    def __init__(self, planet):
        super().__init__(planet)
        self.n_craters = self.planet.vista.coords.integers(1, self.planet.mass // 2 + 2)
        if self.planet.is_planetoid:
            self.planet.vista.status_dict[f"Planet {self.planet.layer}"][
                "craters"
            ] = self.n_craters

    @staticmethod
    def top_left_boundaries(center, mass):
      return (center - mass) // 2

    @staticmethod
    def bottom_right_boundaries(center, mass):
      return (center + mass) // 2

    def foreground(self, center, im, draw_im, planet_mask):
        """
        Called once to draw roughly self.n_ccraters number of craters across
        the surface of the planet's image.
        """
        surface = im.copy()
        scar_surface = ImageDraw.Draw(surface)
        left_x, top_y = self.top_left_boundaries(center, self.planet.mass)
        right_x, bottom_y = self.bottom_right_boundaries(center, self.planet.mass)
        for crater in range(self.n_craters):
            diameter = int(
                self.planet.vista.coords.normal(
                    self.planet.mass / 8, self.planet.mass / 8
                )
            )
            x = self.planet.vista.coords.integers(left_x - diameter, right_x)
            y = self.planet.vista.coords.integers(top_y - diameter, bottom_y)
            hue = self.planet.hue
            shade = self.planet.vista.coords.choice(
                [
                    shade
                    for shade in range(self.planet.vista.palette.shade_depth)
                    if shade != self.planet.shade
                ]
            )
            width = diameter // 10 + self.planet.vista.coords.integers(3)
            outline = self.planet.vista.palette.get_color(hue=hue, shade=shade) + (255,)
            scar_surface.ellipse(
                [x, y, x + diameter, y + diameter],
                fill=self.planet.fill,
                outline=outline,
                width=width,
            )
            ripples = [
                self.planet.vista.coords.integers(0, 360, 2)
                for trash in range(self.planet.vista.coords.integers(3))
            ]
            decay = 0
            while width > decay:
                decay += 1
                for ripple in ripples:
                    scar_surface.arc(
                        [
                            x - width,
                            y - width,
                            x + width + diameter,
                            y + width + diameter,
                        ],
                        start=ripple[0] + 2 * decay,
                        end=ripple[1] - 2 * decay,
                        fill=outline,
                        width=width - decay,
                    )
        im.paste(surface, mask=planet_mask)
