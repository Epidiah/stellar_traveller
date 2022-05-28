# space_vista.py
#
# Here we assemble a randomly generate gif that is a view of some astronomical
# wonders from various rooms inside our space craft.

import imageio
import numpy as np
import pandas as pd
import subprocess

from functools import partial, reduce
from itertools import cycle
from pathlib import Path
from PIL import (
    Image,
    ImageChops,
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
from celestial_bodies.base_planet import BasePlanet
from celestial_bodies.interior import Interior
from celestial_bodies.star_field import StarField

# Constants
IMAGE_PATH = Path("visual")

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
