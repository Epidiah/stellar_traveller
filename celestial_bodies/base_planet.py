from .celestial_body import CelestialBody
import numpy as np
from PIL import (
    Image,
    ImageDraw,
)
from collections import Counter

class BasePlanet(CelestialBody):
    """
    A basic class to subclass other planets from.
    """

    def __init__(self, vista, distance):
        super().__init__(vista)
        self.distance = distance
        self.is_planetoid = True
        self.velocity *= 1 + self.layer / self.distance
        print(f"velocity={self.velocity} distance={self.distance} layer={self.layer}")
        self.mass = int(
            self.vista.coords.triangular(
                4, self.vista.height / self.distance, self.vista.height / np.sqrt(2)
            )
            + self.layer
        )
        self.height = self.width = self.mass
        self.sys_height = self.sys_width = self.mass + 4

        self.hue = self.vista.coords.integers(self.vista.palette.hue_depth)
        self.shade = self.vista.coords.integers(self.vista.palette.shade_depth)
        self.fill = self.vista.palette.get_color(self.hue, self.shade)
        self.colorful = self.vista.coords.choice([True, False], p=[0.2, 0.8])
        # tilt_from_x rotates the image on the x,y plane
        self.tilt_from_x = self.vista.coords.normal(self.vista.orbital_plane, 30)
        # tilt_from_y rotates the Rings and Satelite features on the y.z plane
        self.tilt_from_y = self.vista.coords.integers(-45, 46)
        # Spin determines which way around the plaent the satellite orbits
        self.spin = self.vista.coords.choice([-1, 1])
        self.vista.status_dict[f"Planet {self.layer}"] = Counter()
        self.features = self.generate_features()
        self.im = self.planetary_formation()
        # Reset sys_dimensions to match reality after rotation
        self.sys_width, self.sys_height = self.im.size
        # Initial position of the planet in the field of stars
        x = self.vista.coords.integers(0, self.vista.length * self.velocity)
        y = self.vista.coords.normal(self.vista.height // 2, self.vista.height // 4)
        self.x, self.y = self.set_course((x + self.sys_width // 2, y)) - [
            self.sys_width // 2,
            self.sys_height // 2,
        ]

    def generate_features(self):
        """
        A placeholder for other planet classes that will have features.

        Returns an empty list.
        """
        return []

    def planetary_formation(self):
        """
        Creates an image of the planet, including all the features that do not move
        with respect to the planet itself.

        Returns an PIL.Image of the planet and its static features
        """
        self.sys_width = int(
            max([feature.sys_width for feature in self.features + [self]])
        )
        self.sys_height = int(
            max([feature.sys_height for feature in self.features + [self]])
        )
        sys_im = Image.new("RGBA", (self.sys_width, self.sys_height), (0, 0, 0, 0))
        # Create the planet_mask here and then...
        planet_mask = sys_im.copy()
        center = np.array((self.sys_width, self.sys_height)) // 2
        draw_sys = ImageDraw.Draw(sys_im)
        draw_mask = ImageDraw.Draw(planet_mask)
        for feature in self.features:
            feature.background(center, sys_im, draw_sys)
        draw_sys.ellipse(
            (center - self.mass // 2).tolist() + (center + self.mass // 2).tolist(),
            fill=(*self.fill, 255),
        )
        # ...draw on it here so you're only masking the planet.
        draw_mask.ellipse(
            (center - self.mass // 2).tolist() + (center + self.mass // 2).tolist(),
            fill=(*self.fill, 255),
        )
        for feature in self.features:
            feature.foreground(center, sys_im, draw_sys, planet_mask)
        sys_im = sys_im.rotate(
            self.tilt_from_x,
            resample=Image.BICUBIC,
            expand=True,
            center=(self.sys_width // 2, self.sys_height // 2),
            fillcolor=(0, 0, 0, 0),
        )
        return sys_im

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the planet and its features on a frame of the animated gif.
        """
        planetary_systems = self.drift(
            np.array([self.x, self.y]),
            velocity=self.velocity,
            frame_n=frame_n,
        )
        # Goes through all the dynamic features of the planet and draws them so that
        # those that drift behind the planet are occluded and those that drift in front
        # are not.
        for x, y in planetary_systems:
            for feature in self.features:
                feature.drifting_background(x, y, image, drawing_frame, frame_n)
            image.paste(self.im, (int(x), int(y)), mask=self.im)
            for feature in self.features:
                feature.drifting_foreground(x, y, image, drawing_frame, frame_n)
