# space_vista.py
#
# Here we assemble a randomly generate gif that is a view of some astronomical
# wonders from various rooms inside our space craft.

import imageio
import numpy as np
import pandas as pd
import subprocess

from functools import reduce
from itertools import cycle
from pathlib import Path
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFont,
)
from tqdm import tqdm
from interior.astro_garden import AstroGarden
from interior.engineering import Engineering
from interior.extra_vehicular_activity import ExtraVehicularActivity
from interior.stellar_cafe import StellarCafe
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
from celestial_bodies.base_planet import BasePlanet
from celestial_bodies.star_field import StarField
from interior.interior import Interior

# Constants
BASE_DIR = Path(".")

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
                Interior,
                AstroGarden,
                Engineering,
                StellarCafe,
                ExtraVehicularActivity,
            ]
        )(self, file_path=BASE_DIR / "assets/visual")

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


# def c2s(xy):
#     return tuple(xy @ FLIP_Y + np.array([200, 200]))


# def s2c(xy):
#     return tuple((xy - np.array([200, 200])) @ FLIP_Y)


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
