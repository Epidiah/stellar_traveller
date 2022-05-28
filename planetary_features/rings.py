from .planetary_feature import PlanetaryFeature
import numpy as np
from PIL import (
    Image,
    ImageDraw,
)

class Rings(PlanetaryFeature):
    """
    A static PlanetaryFeature that resembles rings around the planet.
    """

    def __init__(self, planet, med_diameter=None, local_height=None):
        super().__init__(planet)
        if self.planet.colorful:
            hue = self.planet.vista.coords.choice(
                [
                    hue
                    for hue in range(self.planet.vista.palette.hue_depth)
                    if hue != self.planet.hue
                ]
            )
            shade = self.planet.shade
        else:
            hue = self.planet.hue
            shade = self.planet.vista.coords.choice(
                [
                    shade
                    for shade in range(self.planet.vista.palette.shade_depth - 2)
                    if shade != self.planet.shade
                ]
            )
        self.fill = self.planet.vista.palette.get_color(hue=hue, shade=shade)
        self.alpha = (255,)
        self.thickness = self.planet.vista.coords.integers(1, self.planet.mass // 4 + 2)
        if med_diameter == None:
            self.med_diameter = self.planet.mass * 2
        else:
            self.med_diameter = med_diameter
        self.diameter = (
            int(
                self.planet.vista.coords.normal(self.med_diameter, self.planet.mass / 6)
            )
            + self.thickness
        )
        if local_height == None:
            self.ring_height = int(
                self.diameter * np.sin(np.radians(self.planet.tilt_from_y))
            )
            if self.ring_height == 0:
                self.ring_height = -1
        else:
            self.ring_height = local_height
        self.sys_width = self.diameter + self.thickness * 2
        self.sys_height = abs(self.ring_height) + self.thickness * 2
        self.center = np.array((self.diameter, abs(self.ring_height))) // 2
        self.planet.vista.status_dict[f"Planet {self.planet.layer}"]["rings"] += 1

    def ring_maker(self, im, center):
        """
        The method used to draw the rings according to the specifications on the static
        image of the planet.
        """
        x, y = center - self.center
        self.ring = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.draw_ring = ImageDraw.Draw(self.ring)
        self.draw_ring.ellipse(
            [x, y, x + self.diameter, y + abs(self.ring_height)],
            fill=(0, 0, 0, 0),
            outline=self.fill + self.alpha,
            width=self.thickness,
        )
        # Cut-out for planet
        clearance = abs(
            int(self.planet.mass * np.sin(np.radians(self.planet.tilt_from_y)))
        )
        co_x = center[0] - self.planet.width // 2
        co_y = center[1] - clearance // 2
        self.draw_ring.ellipse(
            [co_x - 1, co_y - 1, co_x + self.planet.mass + 2, co_y + clearance + 2],
            fill=(0, 0, 0, 0),
        )

    def background(self, center, im, draw_im):
        """
        Called first to draw the full ring behind the planet.
        """
        self.ring_maker(im, center)
        im.alpha_composite(self.ring)

    def foreground(self, center, im, draw_im, planet_mask):
        """
        Called second to redraw the portion of the ring that should be in front
        of the planet.
        """
        x, y = center - self.center
        background = [0, 0, im.size[0], center[1]]
        self.draw_ring.rectangle(background, fill=(0, 0, 0, 0))
        if self.ring_height > 0:
            self.ring.transpose(Image.FLIP_TOP_BOTTOM)
        im.alpha_composite(self.ring)
