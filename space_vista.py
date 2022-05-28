# space_vista.py
#
# Here we assemble a randomly generate gif that is a view of some astronomical
# wonders from various rooms inside our space craft.

import imageio
import numpy as np
import pandas as pd
import subprocess

from collections import Counter
from functools import partial, reduce
from itertools import cycle
from pathlib import Path
from PIL import (
    Image,
    ImageChops,
    ImageColor,
    ImageDraw,
    ImageFilter,
    ImageFont,
)
from tqdm import tqdm
from planetary_features.clouds import Clouds
from planetary_features.continents import Continents
from planetary_features.craters import Craters
from planetary_features.rings import Rings
from planetary_features.satellite_cluster import SatelliteCluster
from palette.palette import Palette
from palette.pasteller_palette import PastellerPalette
## Experimental Palettes
# from palette.split_complementary import SplitComplementaryPalette
# from palette.complementary_palette_16 import ComplementaryPalette16
# from palette.mono_palette_16 import MonoPalette16

# Constants

IMAGE_PATH = Path("visual")



# So I don't have to figure this out every time I flip from screen to cartesian
FLIP_Y = np.array([[1, 0], [0, -1]])

# For when I just need a random number generator not tied to coordinates or
# location on the ship. Or when I need to jump start another RNG
RNG = np.random.default_rng()

# Eventually these will move into the classes that use them
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


class RandomNumberGenerator(np.random.Generator):
    """
    A basic randome number generator to subclass from.
    """

    def __init__(self, seed=None):
        if type(seed) is np.random._pcg64.PCG64:
            bg = seed
        elif seed is None:
            bg = np.random.default_rng().bit_generator
        else:
            bg = np.random.default_rng(seed).bit_generator
        super().__init__(bg)


class Coordinates(RandomNumberGenerator):
    """
    The random number generator for your location in the cosmos.
    Given a legal set of coordinates, it will reproduce all parts of the vista
    that are not associated with your location in the ship.
    This includes determining the overall palette.
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


class ShipLocation(RandomNumberGenerator):
    """
    The random number generator for your location within and in the immediate vicinity
    of the ship. Given a legal location, it will reproduce all parts of the vista
    that are associated with your location in the ship, but not the rest of the cosmos
    nor the specific palette.
    """

    def __init__(self, location=None):
        if type(location) is np.random._pcg64.PCG64:
            bg = location
        elif location is None:
            location = RNG.integers(1002)
        elif type(location) is str:
            location = int(str)
        self.location = location
        bg = np.random.default_rng(self.location).bit_generator
        super().__init__(bg)


class Vista:
    """
    The container of all the layers in view.
    Call this first, it'll be used as an argument to all your layers.
    Then call your backdrop followed by all the layers in order of furthest to nearest.
    End with an Interior.
    """

    def __init__(
        self,
        coords,
        shiploc,
        velocity=None,
        field_width=720,
        field_height=720,
        palette=None,
        bearing=None,
        length=1200,
        fps=24,
    ):
        self.coords = coords
        self.shiploc = shiploc
        print(f"Approaching coordinates {coords}…")
        if velocity is None:
            self.velocity = self.coords.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        else:
            self.velocity = velocity
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
            self.palette = Palette(self.coords)
        if bearing is not None:
            self.bearing = np.radians(bearing)
        else:
            self.bearing = np.radians(self.coords.integers(0, 360))
        self.rot_matrix = np.array(
            [
                [np.cos(self.bearing), -1 * np.sin(self.bearing)],
                [np.sin(self.bearing), np.cos(self.bearing)],
            ]
        )
        self.bodies = []
        self.length = length
        self.fps = fps
        self.orbital_plane = self.coords.integers(0, 360)
        self.status_dict = {}
        self.generate_view()

    def generate_view(self):
        """
        This method is called to randomly create all the bodies in the vista.
        """
        self.star_field = StarField(self, self.coords.integers(450, 551))
        self.planets = []
        for trash in range(abs(int(self.coords.normal(3, 2))) + 1):
            self.planets.append(RandomPlanet(self, 20))
        for trash in range(abs(int(self.coords.normal(2, 1)))):
            self.planets.append(RandomPlanet(self, 12))
        for trash in range(abs(int(self.coords.normal(1, 1)))):
            self.planets.append(RandomPlanet(self, 4))
        self.total_planets = len(self.planets)
        self.interior = self.shiploc.choice(
            [
                partial(
                    Interior, file_path=Path(IMAGE_PATH, "observation_windows.png")
                ),
                AstroGarden,
                Engineering,
                StellarCafe,
                ExtraVehicularActivity,
            ]
        )(self)

    def add_celestial_body(self, body):
        """
        Called by a CelestialBody to add itself to the Vista and to calculate its
        layer and velocity.

        Returns the layer and velocity for the body.
        """
        layer = len(self.bodies)
        self.bodies.append(body)
        return layer, self.velocity

    def draw_bodies(self):
        """
        A generator function that draws each body in the void frame by frame.
        Called upon by the self.save method.
        """
        voids = (
            Image.new("RGBA", (self.width, self.height), color="Black")
            for trash in range(self.length)
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
        """

        Returns the total number of pixels shown in each frame.
        """
        return self.width * self.height

    def save_gif(self, file_name="starry.gif"):
        """
        Saves and then optimizes the whole Vista as an animated gif.
        """
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
                "-O3",
                "--no-extensions",
                "--optimize",
                "--colors",
                str(self.palette.n_colors),
                "--output",
                file_name,
            ]
        )
        first.close()

    def save_video(self, file_name="starry.mp4"):
        """
        Saves and then optimizes the whole Vista as an animated gif.
        """
        slides = (np.asarray(im) for im in self.draw_bodies())
        imageio.mimwrite(
            uri=file_name,
            fps=24,
            ims=slides,
        )


class CelestialBody:
    """
    The parent class of all layers to your vista.
    """

    def __init__(self, vista):
        self.vista = vista
        self.layer, self.velocity = self.vista.add_celestial_body(self)

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
        """
        Called by the Vista when it comes time to draw a frame of the animated gif.
        """
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
        """
        Determines where the CelestialBody will be self.vista.length frames into
        the future and into the past so that possible gaps don't appear in the
        animation.

        Returns present location, past location, future location.
        """
        mirror_vector = self.vista.length * velocity * np.array([1, 0])
        mirror_xy = xy - self.vista.rot_matrix @ mirror_vector
        parallel_xy = xy + self.vista.rot_matrix @ mirror_vector
        return (
            self.cart_to_screen(xy),
            self.cart_to_screen(mirror_xy),
            self.cart_to_screen(parallel_xy),
        )

    def set_course(self, xy):
        """
        Rotates xy cordinates around the center of the frame so that they are
        guaranteed to be within the path of the spacecraft.


        Returns the rotated xy as screen coordinates.
        """
        return self.cart_to_screen(
            self.vista.rot_matrix @ self.screen_to_cart(np.array(xy))
        )


class StarField(CelestialBody):
    """
    The standard backdrop for the Vista. A randomly generated field of stars
    """

    def __init__(self, vista, n_stars=500):
        super().__init__(vista)
        self.velocity /= 3
        hypotenuse = np.ceil(np.sqrt(self.vista.width ** 2 + self.vista.height ** 2))
        self.leeway = (hypotenuse - self.vista.height) // 2
        self.traverse = self.vista.length * self.velocity
        self.fieldsize = self.traverse * hypotenuse
        self.star_map = self.let_there_be_light(n_stars)
        self.vista.status_dict["star density"] = self.n_stars / self.fieldsize
        print(f"Star density = {self.vista.status_dict['star density']}")

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
            self.vista.coords.integers(
                0, self.vista.length * self.velocity, n_stars, endpoint=True
            ),
            columns=["x"],
        )
        star_map["y"] = self.vista.coords.integers(
            -1 * self.leeway, self.vista.height + self.leeway, n_stars, endpoint=True
        )
        # Remove duplicates and get the actual number of stars
        star_map.drop_duplicates(inplace=True, ignore_index=True)
        self.n_stars = len(star_map)
        # Assign initial color to each star, favoring 'brighter' colors
        star_map["rgb"] = (
            self.vista.palette.palette.iloc[:, 0]
            .sample(n=len(star_map), replace=True)
            .reset_index(drop=True)
        )
        # Create a buffer of duplicate stars on either end of the map so that gaps
        # in the starfield aren't created by rotations and other manipulations.
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
        Bypasses the paralleling, which would be reduntant in this case.
        """
        return self.cart_to_screen(xy)

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the Starfield on the frame.
        """

        def star_drift(row):
            """
            A little helper function to combine the 'x' and 'y' columns of
            self.star_map into a single column of vectors [x,y] and then run
            them through the drift.
            """
            xy = row.array[0]
            return self.drift(xy, velocity=self.velocity, frame_n=frame_n)

        self.star_map["drift"] = self.star_map[["xy"]].apply(star_drift, axis=1)

        for xy, rgb in self.star_map[["drift", "rgb"]].itertuples(
            index=False, name=None
        ):
            drawing_frame.point(tuple(xy), fill=(*rgb, 255))


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


class RandomPlanet(BasePlanet):
    """
    The standard planet.
    A randomly generated planet with a randomly generated set of features.
    """

    def generate_features(self):
        """
        Uses the Vista's Coordinates to randomly generate the planet's set of
        features.

        Returns a list of features.
        """
        features = []
        # Static surface features
        features.append(
            self.vista.coords.choice(
                [Clouds(self), Continents(self), Craters(self)],
            )
        )
        # Static orbital features
        if self.vista.coords.integers(4) == 0:
            features.append(Rings(self))
            for i in range(int(max(0, self.vista.coords.normal(3, 1)))):
                features.append(
                    Rings(
                        self,
                        features[i + 1].diameter,
                        features[i + 1].ring_height,
                    )
                )
        # Dynamic orbital features
        if self.vista.coords.choice([True, False], p=[0.6, 0.4]):
            features.append(SatelliteCluster(self))
        return features

class Interior(CelestialBody):
    """
    The base class for an interior. Can be used for any interior that has no animated
    elements. Otherwise subclassed for an animate interior
    """

    def __init__(self, vista, file_path, invert_colors=False):
        super().__init__(vista)
        self.bgr = RNG.permutation(range(3))
        if invert_colors:
            self.invert_color = RNG.integers(2, size=3)
        else:
            self.invert_color = (0, 0, 0)
        self.activate_camera(file_path)

    def activate_camera(self, file_path):
        """
        A catch-all method for the image processing and setup done to the interior's
        image file(s) before drawing. In this base class it just recolors the image.
        """
        self.im = self.recolor(Image.open(file_path))

    def recolor(self, im):
        """
        This method takes the interior's image and recolors it to match the palette
        of the rest of the Vista.

        Returns recolored PIL.image
        """
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
        """
        Draws the interior image over everything else in the frame.
        """
        image.alpha_composite(self.im)


class AstroGarden(Interior):
    """
    Provides small animation capabilities for the series of astrogarden images that
    include slowly blinking lights along the path.
    """

    def __init__(self, vista, file_path="astrogarden.png"):
        super().__init__(vista, file_path)

    def activate_camera(self, file_path):
        """
        The method for assembling the film_strip of interior images and ensuring each
        image matches the Vista's palette.
        """
        file_name, ext = file_path.split(".")
        self.film_strip = [
            self.recolor(Image.open(Path(IMAGE_PATH, f"{file_name}{n}.{ext}")))
            for n in range(3)
        ]

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the interior image over everything else in the frame, advancing through
        the film_strip at regular intervals.
        """
        blink = frame_n % 96
        if blink < 24:
            im = self.film_strip[0]
        elif 48 <= blink < 72:
            im = self.film_strip[2]
        else:
            im = self.film_strip[1]
        image.alpha_composite(im)


class ExtraVehicularActivity(Interior):
    """
    The exterior interior, a view from inside a space helmet as our viewer floats
    outside the craft.
    """

    def __init__(self, vista, file_path=Path(IMAGE_PATH, "bubble_helmet.png")):
        super().__init__(vista, file_path)
        self.fog = fog_like_alpha(self.vista.size, [180, 396, 540, 756])
        self.font_color = vista.palette.get_color(1, 0)
        self.font = ImageFont.truetype(".fonts/HP-15C_Simulator_Font.ttf", 4)
        self.lines = self.readout()
        self.on_screen_text = [next(self.lines) for trash in range(4)]

    def readout(self):
        """
        Provides a vital message to be printed on the space helmet's readout.
        """
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
        """
        Draws the interior image over everything else in the frame, including animating
        the fog from the astronaut's breath on the helmet glass and the helment's text
        display.
        """
        fog = ImageChops.multiply(
            Image.new(
                "L",
                image.size,
                color=int(np.ceil(70 + 60 * np.cos(frame_n / 120 * np.pi))),
            ),
            self.fog,
        )
        breath = Image.new("RGBA", image.size, "white")
        breath.putalpha(fog)
        image.alpha_composite(breath)
        image.alpha_composite(self.im)
        # Print text between screen coordinates (168, 372) and (190, 394)
        if (frame_n % 120) == 0:
            self.on_screen_text = self.on_screen_text[1:] + [next(self.lines)]
            self.leading_lines = "".join(self.on_screen_text[:3])
            self.last_line = self.on_screen_text[3]
        drawing_frame.multiline_text(
            (302, 675),
            text=self.leading_lines + self.last_line[: (frame_n % 120) // 5],
            fill=self.font_color,
            font=self.font,
            spacing=2,
            align="left",
        )


class StellarCafe(Interior):
    """
    A view from inside a on ship cafe. Includes animation for steam rising from mugs.
    """

    def __init__(self, vista, file_path=Path(IMAGE_PATH, "stellar_cafe.png")):
        super().__init__(vista, file_path)
        self.steam_box_left = [236, 520, 262, 580]
        self.steam_box_right = [423, 533, 450, 594]
        self.left_mug = RNG.choice([True, False], 2)
        self.right_mug = RNG.choice([True, False], 2)
        self.steam = self.steam_cycle()

    def generate_mug_steam(self):
        """
        Creates a snapshot of steam that rising from the mugs.
        """
        steam_left = flame_like_alpha(self.im.size, self.steam_box_left, *self.left_mug)
        steam_right = flame_like_alpha(
            self.im.size, self.steam_box_right, *self.right_mug
        )
        steam = ImageChops.lighter(steam_left, steam_right)
        steaming = Image.new("RGBA", self.vista.size, "white")
        steaming.putalpha(steam)
        return steaming

    def steam_cycle(self):
        first_steaming = next_steaming = self.generate_mug_steam()
        for n in range(self.vista.length - 12):
            if n % 12 == 0:
                prev_steaming, next_steaming = next_steaming, self.generate_mug_steam()
            yield Image.blend(prev_steaming, next_steaming, (n % 12 / 12))
        for n in range(12):
            yield Image.blend(next_steaming, first_steaming, n / 12)

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the interior image over everything else in the frame, including animating
        the steam pouring off of two hot beverages.
        """
        current_im = self.im.copy()
        current_im.alpha_composite(next(self.steam))
        image.alpha_composite(current_im)


class Engineering(Interior):
    """
    A view through a heptagonal portal in engineering. Includes animations for various
    readouts and the occasional flickering lighting effect.
    """

    def __init__(self, vista, file_path="engineering.png"):
        super().__init__(vista, file_path)
        self.osc_bg = self.vista.palette.get_color(hue=-1, shade=-2)
        self.osc_shine = self.vista.palette.get_color(hue=-1, shade=1)
        self.osc_fg0 = self.vista.palette.get_color(hue=-2, shade=0)
        self.osc_fg1 = self.vista.palette.get_color(hue=2, shade=0)
        self.osc_lines = self.vista.palette.get_color(hue=1, shade=1)
        self.diagnostics = RNG.choice(["sins", "soothes"], p=[0.75, 0.25])

    def activate_camera(self, file_path):
        """
        The method for assembling the film_strip of interior images and ensuring each
        image matches the Vista's palette.
        """
        file_name, ext = file_path.split(".")
        self.flicker = RNG.choice(["s-", "f-"], p=[0.8, 0.2])
        self.film_strip = [
            self.recolor(
                Image.open(Path(IMAGE_PATH, f"{self.flicker}{file_name}{n}.{ext}"))
            )
            for n in range(4)
        ]

    def draw(self, image, drawing_frame, frame_n):
        """
        Draws the interior image over everything else in the frame, advancing through
        the film_strip at regular intervals. Includes drawing an animated oscilloscope
        on top of everything else.
        """
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
        osc = Image.new("RGBA", (180, 180), (0, 0, 0, 0))
        dr_osc = ImageDraw.Draw(osc)
        dr_osc.ellipse(
            [0, 0, 180, 180], fill=self.osc_bg, outline=self.osc_fg0, width=4
        )
        dr_osc.ellipse([108, 18, 144, 54], fill=self.osc_shine)
        dr_osc.line([90, 0, 90, 180], fill=self.osc_lines)
        dr_osc.line([0, 90, 180, 90], fill=self.osc_lines)
        mask = osc.copy()
        # xs = np.arange(180)
        xs = np.linspace(0, 180, num=360, endpoint=False)
        if self.diagnostics == "sins":
            y0s = np.sin((xs - 90 + frame_n) * np.pi / 60) * 27 + 90
            y1s = np.sin((xs - 90) * frame_n / 600 * np.pi) * 36 + 90
            dr_osc.line(
                [(x, y) for x, y in zip(xs, y0s)],
                fill=self.osc_fg0,
                width=2,
                joint="curve",
            )
            lxs, ly1s = xs.tolist(), y1s.tolist()
            dr_osc.point([(x, y) for x, y in zip(lxs, ly1s)], fill=self.osc_fg1)
            dr_osc.point(
                [(x, y) for x, y in zip(lxs[1:] + [lxs[0]], ly1s[1:] + [ly1s[0]])],
                fill=self.osc_fg1,
            )
        elif self.diagnostics == "soothes":
            y0s = np.sin(xs - 90 + (frame_n / 40 % 60)) * 27 + 90
            y1s = np.sin((xs - 90) + (frame_n / 20 % 60)) * 36 + 90
            dr_osc.point([(x, y) for x, y in zip(xs, y0s)], fill=self.osc_fg0)
            dr_osc.point([(x, y) for x, y in zip(xs, y1s)], fill=self.osc_fg1)
        im.paste(osc, box=(522, 18), mask=mask)

        image.alpha_composite(im)

## Helper Functions

def flame_like_alpha(full_size, box, left_curl=False, whisp=False):
    # Make some noise!
    haze = Image.effect_noise(
        [box[2] - box[0], box[3] - box[1]],
        RNG.integers(290, 311),
    )
    x, y = haze.size
    # Now let's shape that noise so it vaguely fits the silhouette of fire
    drw_haze = ImageDraw.Draw(haze)
    drw_haze.ellipse(
        [x * left_curl - x // 2, -y / 8, x * left_curl + x // 2, y * 5 / 8],
        "black",
    )
    if whisp:
        drw_haze.ellipse(
            [
                x * (not left_curl) - x // 2,
                y * 3 / 8,
                x * (not left_curl) + x // 2,
                y * 9 / 8,
            ],
            "black",
        )
    shifts = RNG.integers(-4, 5, 4)
    mask = Image.new("L", haze.size, "black")
    drw_mask = ImageDraw.Draw(mask)
    drw_mask.ellipse(((0, 0) + haze.size + shifts).tolist(), 128)
    flames = Image.new("L", full_size, color="black")
    flames.paste(haze, box=box, mask=mask)
    # Now spread it around and blur it!
    flames = flames.effect_spread(3).filter(ImageFilter.GaussianBlur(3))
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
    shiploc = ShipLocation()
    p = pd.Series([tuple(coords.integers(0, 256, 3)) for trash in range(6)])
    spacescape = Vista(
        coords=coords,
        shiploc=shiploc,
        palette=coords.choice([Palette, PastellerPalette], p=(0.3, 0.7))(coords, p),
        length=length,
    )
    print("Painting spacescape!")
    spacescape.save_video()
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
        "planetoids": f"{s_noun} {s_verb} {spacescape.total_planets} planetoids.",
    }
    if isinstance(spacescape.bodies[-1], ExtraVehicularActivity):
        computer_readout["task"] = RNG.choice(EVA_GERUNDS) + RNG.choice(EVAS)
    return computer_readout


def spacescape(coords=None, shiploc=None, length=1200):
    if coords is None:
        coords = Coordinates()
    elif type(coords) == str:
        coords = Coordinates(coords)
    if shiploc is None:
        shiploc = ShipLocation()
    p = pd.Series([tuple(coords.integers(0, 256, 3)) for trash in range(6)])
    painting = Vista(
        coords=coords,
        shiploc=shiploc,
        palette=coords.choice([Palette, PastellerPalette], p=(0.3, 0.7))(coords, p),
        length=length,
    )
    s_noun = RNG.choice(DETECTOR)
    s_verb = RNG.choice(DETECTION)
    if s_noun[-1] != "s":
        s_verb += "s"
    computer_readout = {
        "coords": f"{RNG.choice(COORD_GERUNDS)} coordinates {coords}…",
        "star density": f"Star density = {painting.bodies[0].n_stars/(painting.bodies[0].fieldsize)}",
        "planetoids": f"{s_noun} {s_verb} {painting.total_planets} planetoids.",
    }
    if isinstance(painting.bodies[-1], ExtraVehicularActivity):
        computer_readout["task"] = RNG.choice(EVA_GERUNDS) + RNG.choice(EVAS)
    return painting, computer_readout


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


# Testing & Debugging


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


class TestVista(Vista):
    def generate_view(self):
        """
        This method is called to create all the bodies needing testing.
        """
        self.star_field = StarField(self, self.coords.integers(450, 551))
        self.total_planets = 1
        self.layers = max(self.layers, self.total_planets + 1)
        self.planets = []
        self.planets.append(TestPlanet(self))
        self.interior = GuidanceSystems(self)


class TestPlanet(BasePlanet):
    def __init__(self, vista):
        super().__init__(vista)
        self.velocity = 0
        self.colorful = True
        self.x = 100
        self.y = 100
        self.tilt_from_y = 5
        self.bearing = 0
        self.spin = -1
        self.mass = 80
        self.sys_height = self.sys_width = self.height = self.width = self.mass
        self.features = self.generate_features()
        self.im = self.planetary_formation()

    def generate_features(self):
        # features = [Continents(self)]
        features = [Clouds(self)]
        # features = [Craters(self)]
        features.append(Rings(self))
        for i in range(self.vista.coords.integers(3)):
            features.append(
                Rings(self, features[i + 1].diameter, features[i + 1].ring_height)
            )
        features.append(SatelliteCluster(self))
        return features


def run_test():
    coords = Coordinates()
    shiploc = ShipLocation()
    tv = TestVista(coords, shiploc)
    tv.save_video()
    print(tv.planets[0].spin)
