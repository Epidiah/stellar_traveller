# space_vista.py
#
# Here we assemble a randomly generate gif that is a view of some astronomical
# wonders from various rooms inside our space craft.


from functools import partial, reduce
from itertools import cycle, product
from PIL import (
    Image,
    ImageChops,
    ImageColor,
    ImageDraw,
    ImageFilter,
    ImageFont,
    ImageMath,
)
from tqdm import tqdm
import numpy as np
import pandas as pd
import subprocess

# Constants

COLOR_NAMES = pd.read_csv("colors.csv", squeeze=True, names=["color"])
FLIP_Y = np.array([[1, 0], [0, -1]])
RNG = np.random.default_rng()
COORD_GERUNDS = [
    "Approaching",
    "Passing",
    "Orbiting",
    "Transiting",
    "Arriving at",
    "Departing from",
    "Visiting",
    "Surveying",
    "Slingshotting through",
    "Navigating",
    "De-warping at",
    "Hyper-exiting near",
]
DETECTOR = [
    "Sensors",
    "Survey",
    "Telemetry",
    "Probes",
    "Databanks",
    "Observations",
    "Spectrum analyses",
    "Guides",
]
DETECTION = [
    "reveal",
    "report",
    "indicate",
    "suggest",
    "confirm",
    "pinpoint",
]
EVA_GERUNDS = [
    "Performing ",
    "Initiating ",
    "Actuating ",
    "Commencing ",
    "Enjoying ",
]
EVAS = [
    "extra vehicular activity.",
    "a space stroll.",
    "routine hull maintenance.",
    "a recreational float.",
    "observational astronomy.",
    "collection of rare samples.",
    "game of zero-g hide and seek.",
]

# Classes


class Coordinates(np.random.Generator):
    """
    The random number generator seed for your location in the cosmos.
    """

    def __init__(self, coords=None):
        if coords is None:
            coords = RNG.integers(-2_147_483_648, 2_147_483_647, 3)
        if type(coords) is str:
            coords = np.array(
                [
                    int(sc[:3] + sc[4:7] + sc[8:11] + sc[12:])
                    for sc in coords.split(" : ")
                ]
            )
        self.coords = coords
        bg = np.random.default_rng(coords + 2_147_483_648).bit_generator
        super().__init__(bg)

    def __str__(self):
        strcoords = [f"{coord:+011}" for coord in self.coords]
        strcoords = [
            sc[:3] + "*" + sc[3:6] + "°" + sc[6:9] + "." + sc[9:] for sc in strcoords
        ]
        return " : ".join(strcoords)


class Vista:
    """
    The container of all the layers in view.
    Call this first, it'll be used as an argument to all your layers.
    Then call your backdrop.
    Followed by all the layers in order of furthest to nearest.
    End with an interior.
    """

    def __init__(
        self,
        coords,
        field_width=400,
        field_height=None,
        palette=None,
        bearing=None,
        length=1200,
        fps=24,
    ):
        self.RNG = coords
        print(f"Approaching coordinates {coords}…")
        self.width = field_width
        if field_height is not None:
            self.height = field_height
        else:
            self.height = field_width
        self.size = (self.width, self.height)
        self.center = np.array(self.size) // 2
        if palette is not None:
            self.palette = palette
        else:
            self.palette = Palette()
        if bearing is not None:
            self.bearing = np.radians(bearing)
        else:
            self.bearing = np.radians(self.RNG.integers(0, 360))
        self.rot_matrix = np.array(
            [
                [np.cos(self.bearing), -1 * np.sin(self.bearing)],
                [np.sin(self.bearing), np.cos(self.bearing)],
            ]
        )
        self.bodies = []
        self.length = length
        self.fps = fps
        self.orbital_plane = self.RNG.integers(0, 360)

    def add_celestial_body(self, body):
        """
        Called by the celestial body to add itself to the Vista.
        """
        self.bodies.append(body)
        layer = len(self.bodies) - 1
        return layer

    def draw_bodies(self):
        """
        Called after all the bodies have been added to Vista, before saving the gif.
        Draws all the bodies in the void, advancing them after each frame.
        """
        voids = (
            Image.new("RGBA", (self.width, self.height), color="Black")
            for dummy in range(self.length)
        )
        drawing_frames = tqdm(
            ((im, ImageDraw.Draw(im, mode="RGBA")) for im in voids),
            total=self.length,
        )
        for frame_n, (im, df) in enumerate(drawing_frames):
            for body in self.bodies:
                body.draw(im, df, frame_n)
            yield im

    def total_pixels(self):
        return self.width * self.height

    def save(self, file_name="starry.gif"):
        db = self.draw_bodies()
        first = next(db)
        first.save(
            file_name,
            save_all=True,
            include_color_table=True,
            append_images=db,
            optimize=False,
            duration=1000 / self.fps,
            loop=0,
        )
        # Optimizing with gifsicle
        subprocess.call(
            [
                "gifsicle",
                file_name,
                "-O2",
                "--no-extensions",
                "--optimize",
                "--colors",
                str(self.palette.n_colors),
                "--output",
                file_name,
            ]
        )
        first.close()


class CelestialBody:
    """
    The parent class of all layers to your vista.
    """

    def __init__(self, vista):
        self.vista = vista
        self.layer = self.vista.add_celestial_body(self)

    def screen_to_cart(self, xy):
        """
        Assumes `xy` is an array-like object representing the vector [x, y] in
        screen coordinates where the origin is at the top-left corner of the screen
        and positive y goes down.
        Returns a numpy array representing the vector [x, y] in Cartesian coordinates
        where the origin is at the middle of the screen and positive y goes up.
        """
        return (xy - self.vista.center) @ FLIP_Y

    def cart_to_screen(self, xy):
        """
        Assumes `xy` is an array-like object representing the vector [x, y] in
        Cartesian coordinates where the origin is at the middle of the screen
        and positive y goes up.
        Returns a numpy array representing the vector [x, y] in screen coordinates
        where the origin is at the top-left corner of the screen and positive y
        goes down.
        """
        return xy @ FLIP_Y + self.vista.center

    def draw(self, image, drawing_frame, frame_n):
        pass

    def drift(self, xy, velocity, frame_n):
        """
        Takes in a vector of screen coordinates `xy`.
        Returns new screen coordinates that assure an Atari wrap-around situation.
        """
        # Transform screen coordinates to Cartesian because we're not barbarians here.
        xy = self.screen_to_cart(xy)
        # Our rocket runs on that sweet, sweet linear algebra.
        nudge_matrix = self.vista.rot_matrix @ (velocity * frame_n * np.array([1, 0]))
        xy = xy - nudge_matrix
        # Mirroring body so it is at both ends of the journey
        return self.parallel_worlds(xy, velocity)  # self.atari(xy, velocity)

    def parallel_worlds(self, xy, velocity):
        mirror_vector = self.vista.length * velocity * np.array([1, 0])
        test = xy @ self.vista.rot_matrix
        mirror_xy = xy - self.vista.rot_matrix @ mirror_vector
        parallel_xy = xy + self.vista.rot_matrix @ mirror_vector
        # if test[0] > mirror_vector[0]:
        #     mirror_xy = xy - self.vista.rot_matrix @ mirror_vector
        # else:
        #     mirror_xy = xy + self.vista.rot_matrix @ mirror_vector
        return (
            self.cart_to_screen(xy),
            self.cart_to_screen(mirror_xy),
            self.cart_to_screen(parallel_xy),
        )

    def set_course(self, xy):
        """
        This will take bodies placed on the horizontal field of
        """
        return self.cart_to_screen(
            self.vista.rot_matrix @ self.screen_to_cart(np.array(xy))
        )


class StarField(CelestialBody):
    def __init__(self, vista, n_stars=500, velocity=None):
        super().__init__(vista)
        if velocity is not None:
            self.velocity = velocity / 3
        else:
            self.velocity = self.vista.RNG.integers(1, 4, endpoint=True) / 3
        hypotenuse = np.ceil(np.sqrt(self.vista.width ** 2 + self.vista.height ** 2))
        self.leeway = (hypotenuse - self.vista.height) // 2
        self.traverse = self.vista.length * self.velocity
        self.fieldsize = self.traverse * hypotenuse
        self.star_map = self.let_there_be_light(n_stars)
        self.n_stars = len(self.star_map)
        print(f"Star density = {self.n_stars/(self.fieldsize)}")

    def let_there_be_light(self, stars):
        """
        Generates a field of stars along the starship's course given either the number
        of stars or the density.
        Returns a pd.DataFrame with the starting xy in screen coordinates & rgb colors.
        """
        if stars >= 1:
            n_stars = int(stars)
        elif 0 <= stars < 1:
            n_stars = int(self.fieldsize * stars)
        else:
            raise ValueError("`stars` must be an integer >= 1 or a float [0,1)")
        print(f"Generating {n_stars} stars…")
        # Create a DataFrame of all the star locations
        star_map = pd.DataFrame(
            self.vista.RNG.integers(
                0, self.vista.length * self.velocity, n_stars, endpoint=True
            ),
            columns=["x"],
        )
        star_map["y"] = self.vista.RNG.integers(
            -1 * self.leeway, self.vista.height + self.leeway, n_stars, endpoint=True
        )
        # Remove duplicates and get the actual number of stars
        star_map.drop_duplicates(inplace=True, ignore_index=True)
        # Assign initial color to each star, favoring 'brighter' colors
        star_map["rgb"] = (
            self.vista.palette.palette.iloc[:, 0]
            .sample(n=len(star_map), replace=True)
            .reset_index(drop=True)
        )
        # Create a buffer of stars on the back of the map that duplicate the final
        # stars on the map so that gaps in the starfield aren't caused by rotations.
        full_scroll = star_map.copy()
        for zone in [-self.traverse, self.traverse, 2 * self.traverse]:
            buffer = star_map.copy()
            buffer.iloc[:, 0] = buffer["x"] + zone
            full_scroll = full_scroll.append(buffer, ignore_index=True)
        # Put the stars on our path.
        full_scroll["xy"] = full_scroll[["x", "y"]].apply(self.set_course, axis=1)
        return full_scroll

    def parallel_worlds(self, xy, velocity):
        """
        Bypasses the paralleling, which is unneccessary with that huge starfield!
        """
        return self.cart_to_screen(xy)

    def draw(self, image, drawing_frame, frame_n):
        def star_drift(row):
            """A little helper function to combine the 'x' and 'y' columns of
            self.star_map into a single column of vectors [x,y] and then run
            them through the drift."""
            xy = row.array[0]
            return self.drift(xy, velocity=self.velocity, frame_n=frame_n)

        self.star_map["drift"] = self.star_map[["xy"]].apply(star_drift, axis=1)

        for xy, rgb in self.star_map[["drift", "rgb"]].itertuples(
            index=False, name=None
        ):
            drawing_frame.point(tuple(xy), fill=(*rgb, 255))


class BasePlanet(CelestialBody):
    def __init__(self, vista):
        super().__init__(vista)
        # self.velocity = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 8, 9, 12)[
        #     self.layer + self.vista.RNG.integers(-1, 2)
        # ]
        self.velocity = max(1, self.layer // 2 + self.vista.RNG.integers(4))
        self.mass = int(
            self.vista.RNG.triangular(
                4, self.vista.height / 4, self.vista.height / np.sqrt(2)
            )
            + self.layer
        )
        self.height = self.width = self.mass
        self.sys_height = self.sys_width = self.mass + 4

        self.hue = self.vista.RNG.integers(self.vista.palette.hue_depth)
        self.shade = self.vista.RNG.integers(self.vista.palette.shade_depth)
        self.fill = self.vista.palette.get_color(self.hue, self.shade)
        self.colorful = self.vista.RNG.choice([True, False], p=[0.2, 0.8])
        # tilt_from_x rotates the image on the x,y plane
        self.tilt_from_x = self.vista.RNG.normal(self.vista.orbital_plane, 30)
        # tilt_from_y rotates the Rings and Satelite features on the y.z plane
        self.tilt_from_y = self.vista.RNG.integers(-45, 46)
        # Positive spin means foreground features move faster
        # Negative spin means background features move faster
        self.spin = self.vista.RNG.choice([-1, 1])
        self.features = self.generate_features()
        self.im = self.planetary_formation()
        # Initial position of the planet in the field of stars
        x = self.vista.RNG.integers(0, self.vista.length * self.velocity)
        y = self.vista.RNG.normal(self.vista.height // 2, self.vista.height // 4)
        self.x, self.y = self.set_course((x + self.sys_width // 2, y)) - [
            self.sys_width // 2,
            self.sys_height // 2,
        ]

    def generate_features(self):
        return []

    def planetary_formation(self):
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
        planetary_systems = self.drift(
            np.array([self.x, self.y]),
            velocity=self.velocity,
            frame_n=frame_n,
        )
        for x, y in planetary_systems:
            for feature in self.features:
                feature.drifting_background(x, y, image, drawing_frame, frame_n)
            image.paste(self.im, (int(x), int(y)), mask=self.im)
            for feature in self.features:
                feature.drifting_foreground(x, y, image, drawing_frame, frame_n)


class SwiftPlanet(BasePlanet):
    def __init__(self, vista, velocity=None):
        super().__init__(vista)
        self.x = 200
        self.y = 200
        if velocity is not None and 1200 % velocity == 0:
            self.velocity = velocity
        else:
            self.velocity = self.vista.RNG.choice(
                [8, 10, 12, 16], p=[0.3, 0.3, 0.3, 0.1]
            )


class FeaturePlanet(BasePlanet):
    def generate_features(self):
        features = []
        # surface features
        features.append(
            self.vista.RNG.choice(
                [Clouds(self), Continents(self), Craters(self)],
            )
        )
        # orbital features
        if self.vista.RNG.integers(4) == 0:
            features.append(Rings(self))
            for i in range(int(max(0, self.vista.RNG.normal(3, 1)))):
                features.append(
                    Rings(
                        self,
                        features[i + 1].diameter,
                        features[i + 1].ring_height,
                    )
                )
        for moon in range(
            self.vista.RNG.choice(
                [0, 1, 2, 3, self.vista.RNG.integers(3, 12)],
                p=[0.6, 0.25, 0.05, 0.05, 0.05],
            )
        ):
            features.append(Satellite(self))
        return features


class PlanetaryFeature:
    def __init__(self, planet):
        self.planet = planet
        self.sys_width = 0
        self.sys_height = 0

    def background(self, center, im, draw_im):
        pass

    def foreground(self, center, im, draw_im, planet_mask):
        pass

    def drifting_background(self, x, y, image, drawing_frame, frame_n):
        pass

    def drifting_foreground(self, x, y, image, drawing_frame, frame_n):
        pass


class Satellite(PlanetaryFeature):
    def __init__(self, planet):
        super().__init__(planet)
        self.vista = self.planet.vista
        self.colorful = self.planet.colorful

        # Position and orbit
        self.start_frame = self.planet.vista.RNG.integers(1200)
        self.orbit_velocity = 1  # self.planet.vista.RNG.integers(1, 3)
        self.orbit_radius = self.planet.mass * 2 / self.orbit_velocity
        self.orbit_matrix = np.array(
            [
                [np.cos(self.planet.tilt_from_x), -1 * np.sin(self.planet.tilt_from_x)],
                [np.sin(self.planet.tilt_from_x), np.cos(self.planet.tilt_from_x)],
            ]
        )

        # Appearance
        self.mass = max(
            1,
            self.planet.vista.RNG.integers(
                self.planet.mass // 10, self.planet.mass // 4
            ),
        )
        if self.colorful:
            self.hue = self.planet.vista.RNG.choice(
                [
                    hue
                    for hue in range(self.planet.vista.palette.hue_depth)
                    if hue != self.planet.hue
                ]
            )
            self.shade = self.planet.shade
        else:
            self.hue = self.planet.hue
            self.shade = self.planet.vista.RNG.choice(
                [
                    shade
                    for shade in range(self.planet.vista.palette.shade_depth - 1)
                    if shade != self.planet.shade
                ]
            )
        self.fill = self.planet.vista.palette.get_color(hue=self.hue, shade=self.shade)
        self.irregular = self.planet.vista.RNG.choice([True, False], p=[0.1, 0.9])
        if self.mass < 10:
            self.surface = PlanetaryFeature(self)
        elif self.irregular:
            self.surface = Craters(self)
        else:
            self.surface = self.planet.vista.RNG.choice(
                [PlanetaryFeature, Clouds, Continents, Craters],
                p=[0.1, 0.05, 0.15, 0.7],
            )(self)
        self.im = self.generate_sat()

    def generate_sat(self):
        sat_im = Image.new("RGBA", (self.mass + 4, self.mass + 4), (0, 0, 0, 0))
        sat_mask = sat_im.copy()
        draw_sat = ImageDraw.Draw(sat_im)
        draw_mask = ImageDraw.Draw(sat_mask)
        x_shift = self.irregular * self.vista.RNG.integers(
            self.mass // -5, self.mass // 5 + 1
        )
        y_shift = self.irregular * self.vista.RNG.integers(
            self.mass // -5, self.mass // 5 + 1
        )
        center = np.array((self.mass + 4, self.mass + 4)) // 2
        self.surface.background(center, sat_im, draw_sat)
        draw_sat.ellipse(
            [2, 2, self.mass + 2 + x_shift, self.mass + 2 + y_shift],
            fill=(*self.fill, 255),
        )
        draw_mask.ellipse(
            [2, 2, self.mass + 2 + x_shift, self.mass + 2 + y_shift],
            fill=(*self.fill, 255),
        )
        if self.irregular:
            print('Irregular moon!')
            style = self.vista.RNG.choice(
                [
                    (  # Cut-out Style
                        [
                            self.vista.RNG.integers(
                                self.mass // -10, self.mass // 10 + 1
                            ),
                            self.vista.RNG.integers(
                                self.mass // -10, self.mass // 10 + 1
                            ),
                            self.mass + 4 + x_shift // 3,
                            self.mass + 4 + y_shift // 3,
                        ],
                        (0, 0, 0, 0),
                    ),
                    (  # Add-on style
                        [
                            self.vista.RNG.integers(
                                0, self.mass // 8 + 1
                            ),
                            self.vista.RNG.integers(
                                0, self.mass // 8 + 1
                            ),
                            self.mass + 4 + x_shift // 2,
                            self.mass + 4 + y_shift // 2,
                        ],
                        (*self.fill, 255),
                    ),
                ]
            ).tolist()
            draw_mask.ellipse(style[0], tuple(style[1]))
            if self.vista.RNG.integers(2) == 0:
                sat_im.transpose(Image.FLIP_TOP_BOTTOM)
            sat_im.rotate(self.vista.RNG.integers(360))
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
        # Find where the sat is in its orbit around the origin in Cartesian
        shift_x = self.orbit_radius * np.cos(phase)
        shift_y = (
            self.orbit_radius
            * abs(self.planet.tilt_from_y / 90)
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
        phase = (
            (self.orbit_velocity * frame_n + self.start_frame)
            / self.planet.vista.length
            * 2
            * np.pi
        )
        if self.planet.tilt_from_y * np.sin(phase) <= 0:
            x, y = self.oribit_shift(x, y, image, frame_n, phase)
            image.paste(self.im, (int(x), int(y)), mask=self.im)

    def drifting_foreground(self, x, y, image, drawing_frame, frame_n):
        phase = (
            (self.orbit_velocity * frame_n + self.start_frame)
            / self.planet.vista.length
            * 2
            * np.pi
        )
        if self.planet.tilt_from_y * np.sin(phase) > 0:
            x, y = self.oribit_shift(x, y, image, frame_n, phase)
            image.paste(self.im, (int(x), int(y)), mask=self.im)


class Continents(PlanetaryFeature):
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

    def foreground(self, center, im, draw_im, planet_mask):
        surface = coastlines(
            self.planet.vista.RNG, im, self.planet.fill, self.land_colors
        )
        im.paste(surface, (0, 0), mask=planet_mask)


class Clouds(PlanetaryFeature):
    def __init__(self, planet):
        super().__init__(planet)
        self.n_clouds = self.planet.vista.RNG.integers(
            1, max(1, int(np.log(self.planet.mass))) + 2
        )
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
        surface = im.copy()
        clouds = [
            (
                n,
                nebula_like(
                    self.planet.vista.RNG,
                    foreground_color=tuple(
                        self.planet.vista.RNG.choice(self.cloud_colors).tolist()
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


class Craters(PlanetaryFeature):
    def __init__(self, planet):
        super().__init__(planet)
        self.n_craters = self.planet.vista.RNG.integers(1, self.planet.mass // 2 + 2)

    def background(self, center, im, draw_im):
        pass

    def foreground(self, center, im, draw_im, planet_mask):
        surface = im.copy()
        scar_surface = ImageDraw.Draw(surface)
        left_x, top_y = center - self.planet.mass // 2
        right_x, bottom_y = center + self.planet.mass // 2
        for crater in range(self.n_craters):
            diameter = int(
                self.planet.vista.RNG.normal(self.planet.mass / 8, self.planet.mass / 8)
            )
            x = self.planet.vista.RNG.integers(left_x - diameter, right_x)
            y = self.planet.vista.RNG.integers(top_y - diameter, bottom_y)
            hue = self.planet.hue
            shade = self.planet.vista.RNG.choice(
                [
                    shade
                    for shade in range(self.planet.vista.palette.shade_depth)
                    if shade != self.planet.shade
                ]
            )
            width = diameter // 10 + self.planet.vista.RNG.integers(3)
            outline = self.planet.vista.palette.get_color(hue=hue, shade=shade) + (255,)
            scar_surface.ellipse(
                [x, y, x + diameter, y + diameter],
                fill=self.planet.fill,
                outline=outline,
                width=width,
            )
            ripples = [
                self.planet.vista.RNG.integers(0, 360, 2)
                for dummy in range(self.planet.vista.RNG.integers(3))
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


class Rings(PlanetaryFeature):
    def __init__(self, planet, med_diameter=None, local_height=None):
        super().__init__(planet)
        if self.planet.colorful:
            hue = self.planet.vista.RNG.choice(
                [
                    hue
                    for hue in range(self.planet.vista.palette.hue_depth)
                    if hue != self.planet.hue
                ]
            )
            shade = self.planet.shade
        else:
            hue = self.planet.hue
            shade = self.planet.vista.RNG.choice(
                [
                    shade
                    for shade in range(self.planet.vista.palette.shade_depth - 2)
                    if shade != self.planet.shade
                ]
            )
        self.fill = self.planet.vista.palette.get_color(hue=hue, shade=shade)
        self.alpha = (255,)  # (self.planet.vista.RNG.integers(64, 192),)
        self.thickness = self.planet.vista.RNG.integers(1, self.planet.mass // 4 + 2)
        if med_diameter == None:
            self.med_diameter = self.planet.mass * 2
        else:
            self.med_diameter = med_diameter
        self.diameter = (
            int(self.planet.vista.RNG.normal(self.med_diameter, self.planet.mass / 6))
            + self.thickness
        )
        if local_height == None:
            self.ring_height = int(self.diameter * self.planet.tilt_from_y / 90)
            if self.ring_height == 0:
                self.ring_height = -1
            # self.ring_height = int(
            #     self.planet.vista.RNG.triangular(
            #         3, self.planet.mass / 20 + 3, self.diameter * 0.9 + 3
            #     )
            # )
        else:
            self.ring_height = local_height
        self.sys_width = self.diameter + self.thickness * 2
        self.sys_height = abs(self.ring_height) + self.thickness * 2
        self.center = np.array((self.diameter, abs(self.ring_height))) // 2

        # self.inner_diameter = int(
        #     self.planet.vista.RNG.normal(
        #         self.planet.mass * 1.25, self.planet.mass * 0.4
        #     )
        # )
        # self.width = self.outer_diameter = int(
        #     self.planet.vista.RNG.normal(
        #         self.inner_diameter + self.planet.mass / 10, self.planet.mass / 30
        #     )
        # )
        # self.inner_height = int(
        #     self.planet.vista.RNG.triangular(2, self.planet.mass / 3, self.inner_diameter)
        # )
        # self.height = self.outer_height = int(
        #     self.planet.vista.RNG.triangular(
        #         self.inner_height,
        #         (self.inner_height + self.outer_diameter) / 2,
        #         self.outer_diameter,
        #     )
        # )

    def ring_maker(self, im, center):
        # x,y = (center - self.center).tolist()
        # self.ring = Image.new('RGBA', im.size, (0,0,0,0))
        # self.draw_ring = ImageDraw.Draw(self.ring)
        # self.draw_ring.ellipse([x,y,x+self.width, y+self.height], fill=self.fill+self.alpha)
        # self.draw_ring.ellipse(
        #     (center - np.array((self.inner_diameter, self.inner_height))//2).tolist() +
        #     (center + np.array((self.inner_diameter, self.inner_height))//2).tolist(),
        #     fill=(0,0,0,0)
        #     )
        x, y = (center - self.center)
        self.ring = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.draw_ring = ImageDraw.Draw(self.ring)
        self.draw_ring.ellipse(
            [x, y, x + self.diameter, y + abs(self.ring_height)],
            fill=(0, 0, 0, 0),
            outline=self.fill + self.alpha,
            width=self.thickness,
        )
        # Cut-out for planet
        clearance = abs(int(self.planet.mass * self.planet.tilt_from_y / 90))
        co_x = center[0] - self.planet.width//2
        co_y = center[1] - clearance//2
        self.draw_ring.ellipse(
            [co_x-1, co_y-1, co_x + self.planet.mass+2, co_y + clearance+2],
            fill=(0, 0, 0, 0),
        )

    def background(self, center, im, draw_im):
        self.ring_maker(im, center)
        im.alpha_composite(self.ring)

    def foreground(self, center, im, draw_im, planet_mask):
        x, y = center - self.center
        background = [0, 0, im.size[0], center[1]]
        self.draw_ring.rectangle(background, fill=(0, 0, 0, 0))
        if self.ring_height > 0:
            self.ring.transpose(Image.FLIP_TOP_BOTTOM)
        im.alpha_composite(self.ring)


class Interior(CelestialBody):
    def __init__(self, vista, file_path, invert_colors=False):
        super().__init__(vista)
        self.bgr = RNG.permutation(range(3))
        if invert_colors:
            self.invert_color = RNG.integers(2, size=3)
        else:
            self.invert_color = (0, 0, 0)
        self.activate_camera(file_path)

    def activate_camera(self, file_path):
        self.im = self.recolor(Image.open(file_path))

    def recolor(self, im):
        # Convert image to RGB mode so we can quantize the colors to the palette.
        *rgb, a = im.split()
        reorder = []
        for i, c in zip(self.invert_color, self.bgr):
            if i:
                reorder.append(ImageChops.invert(rgb[c]))
            else:
                reorder.append(rgb[c])
        recolor_im = Image.merge("RGB", reorder)
        recolor_im = recolor_im.quantize(
            colors=32,
            palette=self.vista.palette.get_image(),
            method=Image.FASTOCTREE,
            dither=0,
        )
        # Convert back and re-establish the alpha channel through some hoops
        im = Image.merge("RGBA", (*recolor_im.convert("RGB").split(), a))
        return im

    def draw(self, image, drawing_frame, frame_n):
        image.alpha_composite(self.im)


class AstroGarden(Interior):
    def __init__(self, vista, file_path="garden_stroll.png"):
        super().__init__(vista, file_path)

    def activate_camera(self, file_path):
        file_name, ext = file_path.split(".")
        self.film_strip = [
            self.recolor(Image.open(f"{file_name}{n}.{ext}")) for n in range(3)
        ]

    def draw(self, image, drawing_frame, frame_n):
        blink = frame_n % 96
        if blink < 24:
            im = self.film_strip[0]
        elif 48 <= blink < 72:
            im = self.film_strip[2]
        else:
            im = self.film_strip[1]
        image.alpha_composite(im)


class ExtraVehicularActivity(Interior):
    def __init__(self, vista, file_path="BubbleHelmet.png"):
        super().__init__(vista, file_path)
        self.fog = fog_like_alpha(self.vista.size, [100, 220, 300, 420])
        self.font_color = vista.palette.get_color(1, 0)
        self.font = ImageFont.truetype(".fonts/HP-15C_Simulator_Font.ttf", 2)
        self.lines = self.readout()

    def readout(self):
        lines = [
            line.ljust(23) + "\n"
            for line in (
                "Swords Without Master",
                "is in",
                "Worlds Without Master,",
                "issue 3",
                " --Epidiah Ravachol",
            )
        ]
        return cycle(lines)

    def draw(self, image, drawing_frame, frame_n):
        fog = ImageChops.multiply(
            Image.new(
                "L",
                image.size,
                color=int(np.ceil(50 + 10 * np.cos(frame_n / 120 * np.pi))),
            ),
            self.fog,
        )
        breath = Image.new("RGBA", image.size, "white")
        breath.putalpha(fog)
        image.alpha_composite(breath)
        image.alpha_composite(self.im)
        # Text between 168, 372 and 190,394
        if (frame_n % 120) == 0:
            text = [next(self.lines) for dummy in range(4)]
            self.leading_lines = "".join(text[:3])
            self.last_line = text[3]
        drawing_frame.multiline_text(
            (168, 372),
            text=self.leading_lines + self.last_line[: (frame_n % 120) // 5],
            fill=self.font_color,
            font=self.font,
            spacing=2,
            align="left",
        )


class StellarCafe(Interior):
    def __init__(self, vista, file_path="cafe.png"):
        super().__init__(vista, file_path)
        self.last_steaming = None
        self.steam_box_left = [120, 280, 135, 314]
        self.steam_box_right = [225, 286, 240, 320]
        self.left_mug = (RNG.choice([True, False]), False)
        self.right_mug = (RNG.choice([True, False]), False)

    def draw(self, image, drawing_frame, frame_n):
        if (self.last_steaming is None) or (frame_n % 4 == 0):
            steam = flame_like(self.im.size, self.steam_box_left, *self.left_mug)
            steaming = ImageChops.lighter(self.im, steam)
            steam = flame_like(steaming.size, self.steam_box_right, *self.right_mug)
            self.last_steaming = ImageChops.lighter(steaming, steam)
        image.alpha_composite(self.last_steaming)


class Engineering(Interior):
    def __init__(self, vista, file_path="engineering.png"):
        super().__init__(vista, file_path)
        self.osc_bg = self.vista.palette.get_color(hue=-1, shade=-2)
        # if self.vista.palette.n_colors < 32:
        #     self.osc_shine = self.vista.palette.get_color(hue=-1, shade=0)
        # else:
        #     self.osc_shine = self.vista.palette.get_color(hue=-1, shade=1)
        self.osc_shine = self.vista.palette.get_color(hue=-1, shade=1)
        self.osc_fg0 = self.vista.palette.get_color(hue=-2, shade=0)
        self.osc_fg1 = self.vista.palette.get_color(hue=2, shade=0)
        self.osc_lines = self.vista.palette.get_color(hue=1, shade=1)
        self.diagnostics = RNG.choice(["sins", "soothes"], p=[0.75, 0.25])

    def activate_camera(self, file_path):
        file_name, ext = file_path.split(".")
        self.flicker = RNG.choice(["s-", "f-"], p=[0.8, 0.2])
        self.film_strip = [
            self.recolor(Image.open(f"{self.flicker}{file_name}{n}.{ext}"))
            for n in range(4)
        ]

    def draw(self, image, drawing_frame, frame_n):
        blink = frame_n % 96
        if blink < 24:
            op = RNG.choice([0, 1], p=[0.75, 0.25])
            im = self.film_strip[op]
        elif 48 <= blink < 72:
            op = RNG.choice([2, 3], p=[0.75, 0.25])
            im = self.film_strip[op]
        else:
            op = RNG.choice([1, 2], p=[0.75, 0.25])
            im = self.film_strip[op]
        # Oscilloscope
        osc = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        dr_osc = ImageDraw.Draw(osc)
        dr_osc.ellipse(
            [0, 0, 100, 100], fill=self.osc_bg, outline=self.osc_fg0, width=3
        )
        dr_osc.ellipse([60, 10, 80, 30], fill=self.osc_shine)
        dr_osc.line([50, 0, 50, 100], fill=self.osc_lines)
        dr_osc.line([0, 50, 100, 50], fill=self.osc_lines)
        mask = osc.copy()
        xs = np.arange(101)
        if self.diagnostics == "sins":
            y0s = np.sin((xs - 50 + frame_n) * np.pi / 60) * 15 + 50
            y1s = np.sin((xs - 50) * (frame_n / 120 % 60)) * 20 + 50
            dr_osc.line(
                [(x, y) for x, y in zip(xs, y0s)], fill=self.osc_fg0, joint="curve"
            )
            dr_osc.point([(x, y) for x, y in zip(xs, y1s)], fill=self.osc_fg1)
        elif self.diagnostics == "soothes":
            y0s = np.sin(xs - 50 + (frame_n / 40 % 60)) * 15 + 50
            y1s = np.sin((xs - 50) + (frame_n / 20 % 60)) * 20 + 50
            dr_osc.point([(x, y) for x, y in zip(xs, y0s)], fill=self.osc_fg0)
            dr_osc.point([(x, y) for x, y in zip(xs, y1s)], fill=self.osc_fg1)
        im.paste(osc, box=(290, 10), mask=mask)

        image.alpha_composite(im)


class Palette:
    """
    Hue, brightest = 0, darkest = -1
    Shade, brightest = 0, darkest = -1
    """

    def __init__(self, palette=None, hue_depth=6, shade_depth=5):
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
        palette = pd.Series(RNG.choice(COLOR_NAMES, hue_depth, replace=False)).apply(
            ImageColor.getrgb
        )
        palette = self.sort_palette(palette)
        palette = palette.to_frame(name=0)
        return self.fill_shades(palette)

    def sort_palette(self, palette):
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
        if hue is not None and shade is not None:
            return self.palette.iloc[hue, shade]
        elif hue is not None:
            return self.palette.iloc[hue, :].tolist()
        elif shade is not None:
            return self.palette.iloc[:, shade].tolist()
        else:
            raise ValueError("Must have `hue` or `shade` or both, but not neither.")

    def get_image(self):
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


class PastellerPalette(Palette):
    def fill_shades(self, palette):
        """
        Takes a pd.Series of colors in RGB tuple and returns a pd.DataFrame of
        the full palette of colors with the columns representing the shades.
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


class SplitComplementaryPalette(Palette):
    def random_palette(self, hue_depth=6):
        base_hue = self.vista.RNG.integers(0, 360)
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


class MonoPalette16(Palette):
    def __init__(self, palette=None, hue=None, hue_depth=4, shade_depth=4):
        self.hue = hue % 360
        super().__init__(palette=None, hue_depth=4, shade_depth=4)
        self.n_colors = 16

    def random_palette(self, hue_depth=3):
        if self.hue is not None:
            return self.build_palette(self.hue)
        base_hue = self.vista.RNG.integers(0, 360)
        return self.build_palette(base_hue)

    def build_palette(self, base_hue):
        gaps = np.linspace(60, 10, 4)
        palette = pd.DataFrame(
            np.reshape(
                [
                    f"hsl({base_hue}, {s+5*(4-i)}%, {l}%)"
                    for i, s in enumerate(gaps)
                    for l in gaps
                ],
                (4, 4),
            ),
            columns=range(4),
        )
        palette = palette.applymap(ImageColor.getrgb)
        return palette

    def get_image(self):
        h, s = self.palette.shape
        plt_im = Image.new("RGBA", (200 * h, 200 * s), color="black")
        plt_drw = ImageDraw.Draw(plt_im)
        for x, hue in enumerate(self.palette.itertuples(index=False, name=None)):
            for y, (r, g, b) in enumerate(hue):
                plt_drw.rectangle(
                    [200 * x, 200 * y, 200 * (x + 1), 200 * (y + 1)],
                    fill=(r, g, b),
                )
        return plt_im.convert("P")


class ComplementaryPalette16(Palette):
    def __init__(self, palette=None, hue_depth=3, shade_depth=4):
        super().__init__(palette=None, hue_depth=3, shade_depth=4)

    def random_palette(self, hue_depth=3):
        base_hue = self.vista.RNG.integers(0, 360)
        palette = pd.Series(
            [ImageColor.getrgb(f"hsl({base_hue}, {sat}%, 50%)") for sat in (20, 50)]
            + [
                ImageColor.getrgb(f"hsl({(base_hue+180)%360}, 50%, 50%)"),
            ]
        )
        palette = self.sort_palette(palette)
        palette = palette.to_frame(name=0)
        return self.fill_shades(palette)


## Helper Functions


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


def coastlines(coords, im, sea_color, land_colors):
    sketch = Image.new("RGBA", im.size, sea_color)
    sketcher = ImageDraw.Draw(sketch)
    for n in range(coords.integers(3, 13)):
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


def flame_like(full_size, box, left_curl=False, whisp=False, color=None):
    # Make some noise!
    haze = Image.effect_noise(
        [box[2] - box[0], box[3] - box[1]],
        RNG.integers(290, 311),
    ).convert("RGBA")
    x, y = haze.size
    # Now let's shape that noise so it vaguely fits the silhouette of fire
    drw_haze = ImageDraw.Draw(haze)
    drw_haze.ellipse(
        [x * left_curl - x // 2, 0, x * left_curl + x // 2, y // 2],
        (0, 0, 0, 0),
    )
    if whisp:
        drw_haze.ellipse(
            [x * (not left_curl) - x // 2, 0, x * (not left_curl) + x // 2, y // 2],
            (0, 0, 0, 0),
        )
    shifts = RNG.integers(-4, 5, 4)
    mask = Image.new("RGBA", haze.size, (0, 0, 0, 0))
    drw_mask = ImageDraw.Draw(mask)
    drw_mask.ellipse(((0, 0) + haze.size + shifts).tolist(), "white")
    flames = Image.new("RGBA", full_size)
    flames.paste(haze, box=box, mask=mask)
    # Now spread it around and blur it!
    flames = flames.effect_spread(2).filter(ImageFilter.GaussianBlur(3))
    # Fade it out…
    flames.putalpha(96)
    return flames


def fog_like_alpha(full_size, box, color=None):
    # Make some noise!
    haze = Image.effect_noise(
        [box[2] - box[0], box[3] - box[1]],
        RNG.integers(290, 311),
    )
    haze = ImageChops.multiply(haze, haze)
    rad = Image.radial_gradient("L").resize(haze.size)
    haze = ImageChops.subtract(haze, rad)
    # Now let's shape that noise
    mask = Image.new("L", haze.size, 0)
    drw_mask = ImageDraw.Draw(mask)
    drw_mask.ellipse((0, 0) + haze.size, "white")
    fog = Image.new("L", full_size, color="black")
    fog.paste(haze, box=box, mask=mask)
    fog = fog.effect_spread(60).filter(ImageFilter.GaussianBlur(5))
    return fog


def random_spacescape(length=1200):
    coords = Coordinates()
    p = pd.Series([tuple(coords.integers(0, 256, 3)) for dummy in range(6)])
    spacescape = Vista(
        coords=coords,
        palette=coords.choice([Palette, PastellerPalette], p=(0.5, 0.5))(p),
        length=length,
    )
    stars = StarField(spacescape, coords.integers(450, 551))
    total_planets = abs(int(coords.normal(5, 1)))
    for n in range(total_planets):
        print(f"Surveying planet {n+1} of {total_planets}…")
        FeaturePlanet(spacescape)
    RNG.choice(
        [
            partial(Interior, file_path="observation_windows.png"),
            AstroGarden,
            Engineering,
            StellarCafe,
            ExtraVehicularActivity,
        ]
    )(spacescape)
    print("Painting spacescape!")
    spacescape.save()
    im = spacescape.palette.get_image()
    im.save("palette.png")
    im.close()
    s_noun = RNG.choice(DETECTOR)
    s_verb = RNG.choice(DETECTION)
    if s_noun[-1] != "s":
        s_verb += "s"
    computer_readout = {
        "coords": f"{RNG.choice(COORD_GERUNDS)} coordinates {coords}…",
        "star density": f"Star density = {spacescape.bodies[0].n_stars/(spacescape.bodies[0].fieldsize)}",
        "planetoids": f"{s_noun} {s_verb} {total_planets} planetoids.",
    }
    if isinstance(spacescape.bodies[-1], ExtraVehicularActivity):
        computer_readout["task"] = RNG.choice(EVA_GERUNDS) + RNG.choice(EVAS)
    return computer_readout

def spacescape(coords=None, length=1200, room=None):
    if coords is None:
        coords = Coordinates()
    elif type(coords) == str:
        coords = Coordinates(coords)
    p = pd.Series([tuple(coords.integers(0, 256, 3)) for dummy in range(6)])
    painting = Vista(
        coords=coords,
        palette=coords.choice([Palette, PastellerPalette], p=(0.5, 0.5))(p),
        length=length,
    )
    stars = StarField(painting, coords.integers(450, 551))
    total_planets = abs(int(coords.normal(5, 1)))
    for n in range(total_planets):
        print(f"Surveying planet {n+1} of {total_planets}…")
        FeaturePlanet(painting)
    if room is None:
        room = RNG.choice(
            [
                partial(Interior, file_path="observation_windows.png"),
                AstroGarden,
                Engineering,
                StellarCafe,
                ExtraVehicularActivity,
            ]
        )
    room(painting)
    s_noun = RNG.choice(DETECTOR)
    s_verb = RNG.choice(DETECTION)
    if s_noun[-1] != "s":
        s_verb += "s"
    computer_readout = {
        "coords": f"{RNG.choice(COORD_GERUNDS)} coordinates {coords}…",
        "star density": f"Star density = {painting.bodies[0].n_stars/(painting.bodies[0].fieldsize)}",
        "planetoids": f"{s_noun} {s_verb} {total_planets} planetoids.",
    }
    if isinstance(painting.bodies[-1], ExtraVehicularActivity):
        computer_readout["task"] = RNG.choice(EVA_GERUNDS) + RNG.choice(EVAS)
    return painting, room, computer_readout


## Exploratory Functions


def rot_m(bearing):
    return np.array(
        [
            [np.cos(np.radians(bearing)), -1 * np.sin(np.radians(bearing))],
            [np.sin(np.radians(bearing)), np.cos(np.radians(bearing))],
        ]
    )


def c2s(xy):
    return tuple(xy @ FLIP_Y + np.array([200, 200]))


def s2c(xy):
    return tuple((xy - np.array([200, 200])) @ FLIP_Y)


def testers():
    im = Image.new("RGBA", (400, 400), color="darkblue")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 200, 400, 400), fill="DarkSalmon")
    d.rectangle((0, 0, 200, 200), fill="orange")
    return im, d


def remixer(im, order=None, inverts=None):
    *rgb, a = im.split()
    if order is None:
        order = RNG.permutations(range(3))
    if inverts is None:
        inverts = RNG.integers(2, size=3)
    reorder = []
    for i, c in zip(inverts, order):
        if i:
            reorder.append(ImageChops.invert(rgb[c]))
        else:
            reorder.append(rgb[c])
    reorder.append(a)
    reordered_im = Image.merge("RGBA", reorder)
    return reordered_im


# Testing


class GuidanceSystems(Interior):
    def __init__(self, vista, mirrors=False):
        super(Interior, self).__init__(vista)
        self.font = ImageFont.truetype(".fonts/HP-15C_Simulator_Font.ttf", 18)
        self.mirrors = mirrors

    def draw(self, image, drawing_frame, frame_n):
        targets = self.vista.bodies[1:-1]
        fill_colors = cycle(self.vista.palette.get_color(shade=0))
        for i, body in enumerate(targets):
            fill = next(fill_colors)
            drawing_frame.text(
                (50, 0 + 60 * i),
                text=f"Planet-{str(i).zfill(2)}",
                fill=fill,
                font=self.font,
                spacing=2,
                align="left",
            )
            for j, (x, y) in enumerate(
                body.drift(
                    np.array([body.x, body.y]),
                    velocity=body.velocity,
                    frame_n=frame_n,
                )
            ):
                if self.mirrors or j == 0:
                    drawing_frame.line(
                        [
                            150,
                            60 * i + 20 * j,
                            x + body.im.size[0] // 2,
                            y + body.im.size[1] // 2,
                        ],
                        fill=fill,
                    )
                    xstr = str(int(x)).rjust(5)
                    ystr = str(int(y)).rjust(5)
                    drawing_frame.text(
                        (150, 60 * i + 20 * j),
                        text=f"{xstr}, {ystr}",
                        fill="white",
                        font=self.font,
                        spacing=2,
                        align="left",
                    )


class TestPlanet(BasePlanet):
    def __init__(self, vista):
        super().__init__(vista)
        self.velocity = 0
        self.x = 160
        self.y = 160
        self.tilt_from_x = 45
        self.mass = 80
        self.sys_height = self.sys_width = self.height = self.width = self.mass
        self.features = self.generate_features()
        self.im = self.planetary_formation()

    def generate_features(self):
        # features = [Continents(self)]
        features = [Clouds(self)]
        # features = [Craters(self)]
        features.append(Rings(self))
        for i in range(self.vista.RNG.integers(3)):
            features.append(
                Rings(self, features[i + 1].diameter, features[i + 1].ring_height)
            )
        for moon in range(12):
            features.append(Satellite(self))
        return features


def run_test():
    coords = Coordinates()
    p = pd.Series([tuple(coords.integers(0, 256, 3)) for dummy in range(6)])
    test = Vista(
        coords=coords,
        palette=coords.choice([Palette, PastellerPalette], p=(0.5, 0.5))(p),
        # bearing=225,
        length=1200,
        fps=24,
    )
    stars = StarField(test, coords.integers(450, 551))
    # f = Guidance_Calibration(test, x=1200, y=200, velocity=6)
    TestPlanet(test)
    # total_planets = abs(int(coords.normal(5, 1)))
    # for n in range(total_planets):
    #     print(f"Surveying planet {n+1} of {total_planets}…")
    #     TestPlanet(test)
    # RNG.choice(
    #     [
    #         partial(Interior, file_path="observation_windows.png"),
    #         AstroGarden,
    #         Engineering,
    #         StellarCafe,
    #         ExtraVehicularActivity,
    #     ]
    # )(test)
    # GuidanceSystems(test)
    print("Painting spacescape!")
    test.save("starry.gif")

    im = test.palette.get_image()
    im.save("palette.png")
    im.close()
    s_noun = RNG.choice(DETECTOR)
    s_verb = RNG.choice(DETECTION)
    if s_noun[-1] != "s":
        s_verb += "s"
    computer_readout = {
        "coords": f"{RNG.choice(COORD_GERUNDS)} coordinates {coords}…",
        "star density": f"Star density = {test.bodies[0].n_stars/(test.bodies[0].fieldsize)}",
        # "planetoids": f"{s_noun} {s_verb} {total_planets} planetoids.",
    }
    if isinstance(test.bodies[-1], ExtraVehicularActivity):
        computer_readout["task"] = RNG.choice(EVA_GERUNDS) + RNG.choice(EVAS)
    # print("\a")
    for r in computer_readout.values():
        print(r)
    return test
