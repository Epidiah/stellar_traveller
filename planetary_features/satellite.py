from .planetary_feature import PlanetaryFeature
from .craters import Craters
from .clouds import Clouds
from .continents import Continents
import numpy as np
from PIL import (
    Image,
    ImageDraw,
)

class Satellite(PlanetaryFeature):
    """
    The base class for a satellite. Right now, it's very much only moons. I'll have
    to do some reworking of the code when I want to create artificial satellites.
    """

    def __init__(self, planet, cluster):
        super().__init__(planet)
        self.vista = self.planet.vista
        self.colorful = self.planet.colorful
        self.cluster = cluster

        # It is necessary to set this to False, otherwise certain features will be
        # reported as present in the Vista.status_dict.
        self.is_planetoid = False

        # Setting the appearance of the satellite
        self.mass = max(
            1,
            self.planet.vista.coords.integers(
                self.planet.mass // 10, self.planet.mass // 4
            ),
        )
        if self.colorful:
            self.hue = self.planet.vista.coords.choice(
                [
                    hue
                    for hue in range(self.planet.vista.palette.hue_depth)
                    if hue != self.planet.hue
                ]
            )
            self.shade = self.planet.shade
        else:
            self.hue = self.planet.hue
            self.shade = self.planet.vista.coords.choice(
                [
                    shade
                    for shade in range(self.planet.vista.palette.shade_depth - 1)
                    if shade != self.planet.shade
                ]
            )
        self.fill = self.planet.vista.palette.get_color(hue=self.hue, shade=self.shade)
        self.irregular = self.planet.vista.coords.choice([True, False], p=[0.1, 0.9])
        if self.irregular:
            self.vista.status_dict[f"Planet {self.planet.layer}"]["irregular moon"] = +1
        else:
            self.vista.status_dict[f"Planet {self.planet.layer}"]["moon"] = +1
        if self.mass < 10:
            self.surface = PlanetaryFeature(self)
        elif self.irregular:
            self.surface = Craters(self)
        else:
            self.surface = self.planet.vista.coords.choice(
                [PlanetaryFeature, Clouds, Continents, Craters],
                p=[0.1, 0.05, 0.15, 0.7],
            )(self)
        self.im = self.generate_sat()

        # Setting the orbit's position and velocity.
        self.orbit_velocity = 1  # self.planet.vista.coords.integers(1, 3)
        self.orbit_radius = self.planet.mass * 2 / self.orbit_velocity
        self.orbit_matrix = np.array(
            [
                [
                    np.cos(np.radians(self.planet.tilt_from_x)),
                    -1 * np.sin(np.radians(self.planet.tilt_from_x)),
                ],
                [
                    np.sin(np.radians(self.planet.tilt_from_x)),
                    np.cos(np.radians(self.planet.tilt_from_x)),
                ],
            ]
        )
        self.start_location = self.cluster.get_start_location(self)

    def generate_sat(self):
        """
        A method called when creating the Satellite to generate the Satellite's image.
        As this Satellite is a moon, it will have some randomly generated
        PlanetaryFeatures.

        Returns an PIL.Image of the Satellite with its static features.
        """
        sat_im = Image.new("RGBA", (self.mass + 10, self.mass + 10), (0, 0, 0, 0))
        sat_mask = sat_im.copy()
        draw_sat = ImageDraw.Draw(sat_im)
        draw_mask = ImageDraw.Draw(sat_mask)
        x_shift = self.irregular * self.vista.coords.integers(
            self.mass // -5, self.mass // 5 + 1
        )
        y_shift = self.irregular * self.vista.coords.integers(
            self.mass // -5, self.mass // 5 + 1
        )
        center = np.array((self.mass + 10, self.mass + 10)) // 2
        self.surface.background(center, sat_im, draw_sat)
        draw_sat.ellipse(
            [5, 5, self.mass + x_shift, self.mass + y_shift],
            fill=(*self.fill, 255),
        )
        draw_mask.ellipse(
            [5, 5, self.mass + x_shift, self.mass + y_shift],
            fill=(*self.fill, 255),
        )

        # A small proportion of moons will be irregular in shape.
        if self.irregular:
            print("Irregular moon!")
            style = self.vista.coords.choice(
                [
                    (  # Cut-out Style
                        [
                            self.vista.coords.integers(
                                self.mass // -10, self.mass // 10 + 1
                            ),
                            self.vista.coords.integers(
                                self.mass // -10, self.mass // 10 + 1
                            ),
                            self.mass + 4 + x_shift // 3,
                            self.mass + 4 + y_shift // 3,
                        ],
                        (0, 0, 0, 0),
                    ),
                    (  # Add-on style
                        [
                            self.vista.coords.integers(0, self.mass // 8 + 1),
                            self.vista.coords.integers(0, self.mass // 8 + 1),
                            self.mass + 4 + x_shift // 2,
                            self.mass + 4 + y_shift // 2,
                        ],
                        (*self.fill, 255),
                    ),
                ]
            ).tolist()
            draw_mask.ellipse(style[0], tuple(style[1]))
            if self.vista.coords.integers(2) == 0:
                sat_im.transpose(Image.FLIP_TOP_BOTTOM)
            sat_im.rotate(
                self.vista.coords.integers(360),
                resample=Image.BICUBIC,
                expand=True,
                center=(self.mass // 2 + 2, self.mass // 2 + 2),
                fillcolor=(0, 0, 0, 0),
            )
        self.surface.foreground(center, sat_im, draw_sat, sat_mask)
        sat_im = sat_im.rotate(
            self.planet.tilt_from_x,
            resample=Image.BICUBIC,
            expand=True,
            center=(self.mass // 2 + 2, self.mass // 2 + 2),
            fillcolor=(0, 0, 0, 0),
        )
        return sat_im

    def oribit_shift(self, x, y, image, frame_n, phase):
        """
        A method called once per frame to determine where the Satellite should be
        draw on that frame.

        Returns xy screen coordinates
        """
        # Find where the sat is in its orbit around the origin in Cartesian
        shift_x = self.orbit_radius * np.cos(phase)
        shift_y = (
            self.orbit_radius
            * abs(np.sin(np.radians(self.planet.tilt_from_y)))
            * self.planet.spin
            * np.sin(phase)
        )
        # Rotate around the planet's center to fit the planet's tilt from x
        xy = [shift_x, shift_y] @ self.orbit_matrix
        # Translate origin from 0,0 to the planet's top-right corner
        xy += [x, y]
        # Translate origin from planet's top-right corner to planet's center
        xy += np.array(self.planet.im.size) // 2
        # Offset by the distance from the sat's top-right to its center
        xy -= np.array(self.im.size) // 2
        return xy

    def drifting_background(self, x, y, image, drawing_frame, frame_n):
        """
        Called every frame to draw Satellites that might be occuluded by the planet
        as they move with respect to that planet.
        """
        phase = (
            (self.orbit_velocity * frame_n + self.start_location)
            / self.planet.vista.length
            * 2
            * np.pi
        )
        if (
            self.planet.spin
            * np.sin(np.radians(self.planet.tilt_from_y))
            * np.sin(phase)
            <= 0
        ):
            x, y = self.oribit_shift(x, y, image, frame_n, phase)
            image.paste(self.im, (int(x), int(y)), mask=self.im)

    def drifting_foreground(self, x, y, image, drawing_frame, frame_n):
        """
        Called every frame to draw Satellites that might pass in front of the planet
        as they move with respect to that planet.
        """
        phase = (
            (self.orbit_velocity * frame_n + self.start_location)
            / self.planet.vista.length
            * 2
            * np.pi
        )
        if (
            self.planet.spin
            * np.sin(np.radians(self.planet.tilt_from_y))
            * np.sin(phase)
            > 0
        ):
            x, y = self.oribit_shift(x, y, image, frame_n, phase)
            image.paste(self.im, (int(x), int(y)), mask=self.im)
