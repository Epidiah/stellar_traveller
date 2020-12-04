# space_vista.py
#
# Here we assemble a randomly generate gif that is a view of some astronomical
# wonders from various rooms inside our space craft.

from PIL import Image, ImageDraw, ImageColor
from random import choice, choices, gauss, random
from functools import partial
from pygifsicle import optimize
from tqdm import tqdm
import numpy as np
import pandas as pd

# Constants

COLOR_NAMES = pd.read_csv("colors.csv", squeeze=True, names=["color"])
FLIP_Y = np.array([[1, 0], [0, -1]])
RNG = np.random.default_rng()

# Classes


class Vista:
    """
    The container of all the layers in view.
    """

    COUNT = 0

    def __init__(
        self,
        field_width=400,
        field_height=None,
        palette=None,
        velocity=None,
        bearing=None,
        length=1200,
    ):
        self.width = field_width
        if field_height is not None:
            self.height = field_height
        else:
            self.height = field_width
        if palette is not None:
            self.palette = palette
        else:
            self.palette = Palette()
        if velocity is not None:
            self.velocity = velocity
        else:
            self.velocity = np.random.randint(1, 4)
        if bearing is not None:
            self.bearing = np.radians(bearing)
        else:
            self.bearing = np.radians(np.random.randint(0, 360))
        self.rot_matrix = np.array(
            [
                [np.cos(self.bearing), -1 * np.sin(self.bearing)],
                [np.sin(self.bearing), np.cos(self.bearing)],
            ]
        )
        self.size = (self.width, self.height)
        self.bodies = []
        self.length = length
        self.frames = [
            Image.new("RGBA", self.size, color="Black") for _ in range(self.length)
        ]

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
        Draws all the bodies on the frames, advancing them after each frame.
        """
        drawing_frames = tqdm(
            [(im, ImageDraw.Draw(im, mode="RGBA")) for im in self.frames]
        )
        for frame_n, (im, df) in enumerate(drawing_frames):
            for body in self.bodies:
                body.draw(im, df, frame_n)

    def total_pixels(self):
        return self.width * self.height

    def save(self, file_title="starry"):
        Vista.COUNT += 1
        file_name = f"{file_title}-{Vista.COUNT:06d}.gif"
        self.frames[0].save(
            file_name,
            save_all=True,
            include_color_table=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=1000 / 24,
            loop=0,
        )
        optimize(file_name)


class CelestialBody:
    def __init__(self, vista):
        self.vista = vista
        self.layer = self.vista.add_celestial_body(self)
        self.center = np.array([self.vista.width // 2, self.vista.height // 2])

    def screen_to_cart(self, xy):
        """
        Assumes `xy` is an array-like object representing the vector [x, y] in
        screen coordinates where the origin is at the top-left corner of the screen
        and positive y goes down.
        Returns a numpy array representing the vector [x, y] in Cartesian coordinates
        where the origin is at the middle of the screen and positive y goes up.
        """
        return (xy - self.center) @ FLIP_Y

    def cart_to_screen(self, xy):
        """
        Assumes `xy` is an array-like object representing the vector [x, y] in
        Cartesian coordinates where the origin is at the middle of the screen
        and positive y goes up.
        Returns a numpy array representing the vector [x, y] in screen coordinates
        where the origin is at the top-left corner of the screen and positive y
        goes down.
        """
        return xy @ FLIP_Y + self.center

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
        nudge_matrix = (velocity * frame_n * np.array([1, 0])) @ self.vista.rot_matrix
        xy = xy - nudge_matrix
        # Check for wrap-around.
        cosmic_boundaries = self.vista.length * velocity
        test = self.vista.rot_matrix @ xy
        if abs(test[0]) > cosmic_boundaries / 2:
            xy = xy + (cosmic_boundaries * np.array([1, 0])) @ self.vista.rot_matrix
        return self.cart_to_screen(xy)


class StarField(CelestialBody):
    def __init__(self, vista):
        super().__init__(vista)
        self.velocity = self.vista.velocity / 3
        hypotenuse = np.ceil(np.sqrt(self.vista.width ** 2 + self.vista.height ** 2))
        self.leeway = (hypotenuse - self.vista.height) // 2
        self.fieldsize = self.vista.length * self.velocity * hypotenuse
        self.star_map = self.let_there_be_light(500)
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
            RNG.integers(0, self.vista.length * self.velocity, n_stars, endpoint=True),
            columns=["x"],
        )
        star_map["y"] = RNG.integers(
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
        buffered_zone = self.vista.length * self.velocity - self.leeway
        buffer = star_map[star_map["x"] > buffered_zone].copy()
        buffer.iloc[:, 0] = buffer["x"] - buffered_zone + self.leeway
        star_map.append(buffer, ignore_index=True)

        star_map["xy"] = star_map[["x", "y"]].apply(
            lambda row: self.cart_to_screen(
                self.screen_to_cart(np.array(row)) @ self.vista.rot_matrix
            ),
            axis=1,
        )
        return star_map

    def draw(self, image, drawing_frame, frame_n):
        def star_drift(row):
            """A little helper function to combine the 'x' and 'y' columns of
            self.star_map into a single column of vectors [x,y] and then run
            them through the drift."""
            xy = row.array[0]
            return self.drift(xy, velocity=self.velocity, frame_n=frame_n)

        # star_drift = partial(self.drift, velocity=self.velocity, frame_n=frame_n)
        self.star_map["drift"] = self.star_map[["xy"]].apply(star_drift, axis=1)
        # self.star_map["drift"] = self.star_map["xy"].apply(star_drift)
        drifted_xs = self.star_map["drift"].str[0]
        drifted_ys = self.star_map["drift"].str[1]
        window_x = (drifted_xs >= 0) & (drifted_xs <= self.vista.width)
        window_y = (drifted_ys >= -self.leeway) & (
            drifted_ys <= self.vista.height + self.leeway
        )
        window = self.star_map[window_x & window_y]
        for xy, rgb in window[["drift", "rgb"]].itertuples(index=False, name=None):
            drawing_frame.point(tuple(xy), fill=(*rgb, 255))


class Planet(CelestialBody):
    """
    Planets can only exist in layers 1 through 3.
    """

    def __init__(self, vista):
        super().__init__(vista)
        min_x, min_y = self.cart_to_screen(
            np.array([self.vista.length * self.layer / -2, self.vista.height])
        )
        max_x, max_y = self.cart_to_screen(
            np.array([self.vista.length * self.layer / 2, self.vista.height * -1])
        )
        self.x = np.random.randint(min_x, max_x)
        self.y = np.random.randint(min_y, max_y)
        self.mass = int(
            np.random.triangular(
                4, self.vista.height / 3, self.vista.height / np.sqrt(2)
            )
        )
        self.fill = self.vista.palette.random_color(col_n=self.layer - 1)
        self.moons = [Moon(self, self.vista) for _ in range(np.random.randint(5))]

    def draw(self, image, drawing_frame, frame_n):
        x, y = self.drift(
            np.array([self.x, self.y]), velocity=self.layer, frame_n=frame_n
        )
        drawing_frame.ellipse(
            [x, y, x + self.mass, y + self.mass], fill=(*self.fill, 255)
        )
        for moon in self.moons:
            moon.draw(image, drawing_frame, frame_n)
        # self.drift(frame_n, velocity, bearing)

    # def drift(self, frame_n):
    #     # if vector[0]:
    #     #     self.x -= np.sign(vector[0]) * self.layer
    #     # if vector[1]:
    #     #     self.y -= np.sign(vector[1]) * self.layer
    #     # for moon in self.moons:
    #     #     moon.drift(frame_n, velocity, bearing)
    #     pass


class Moon(Planet):
    def __init__(self, planet, vista):
        self.planet = planet
        self.vista = vista
        self.center = np.array([self.vista.width // 2, self.vista.height // 2])
        self.layer = min(5, max(1, self.planet.layer + choice([-1, 1])))
        short_dim = planet.mass
        self.x = self.planet.x + np.random.randint(
            -int(short_dim / 2), int(short_dim * 1.5)
        )
        self.y = self.planet.x + np.random.randint(
            -int(short_dim / 2), int(short_dim * 1.5)
        )
        self.mass = int(
            np.random.triangular(np.sqrt(short_dim), short_dim / 4, short_dim / 3)
        )
        self.fill = self.vista.palette.random_color(col_n=self.layer - 1)
        self.moons = []


class Interior(CelestialBody):
    def __init__(self, vista, file_path):
        super().__init__(vista)
        self.im = Image.open(file_path)
        self.recolor()

    def recolor(self):
        # Convert image to RGB mode so we can quantize the colors to the palette.
        recolor_im = self.im.convert("RGB")
        recolor_im = recolor_im.quantize(
            colors=32,
            palette=self.vista.palette.get_image(),
            method=Image.FASTOCTREE,
            dither=0,
        )
        # Convert back and re-establish the alpha channel through some hoops
        recolor_im = recolor_im.convert("RGBA")
        temp_im = Image.new("RGBA", self.im.size, color=(0, 0, 0, 0))
        temp_im.paste(recolor_im, mask=self.im)
        self.im = temp_im

    def draw(self, image, drawing_frame, frame_n):
        layer_image(image, self.im)


class Palette:
    def __init__(self, palette=None):
        if palette is not None:
            self.palette = palette
        else:
            self.palette = Palette.random_palette(color_depth=6, shade_depth=5)

    @classmethod
    def random_palette(cls, color_depth=6, shade_depth=5):
        """
        Returns a pd.DataFrame with colors as the rows, sorted from brightest to
        darkest, and shades of the color as the columns, sorted from darkest to
        brightest.
        """

        def color_sort_key(colors):
            lums = (0.2126, 0.7152, 0.0722)
            return colors.apply(lambda x: sum(c * l for c, l in zip(x, lums)))

        def color_dive(color, *, s=0):
            depth = shade_depth - s
            factor = 1 - depth * 0.9 / (shade_depth)
            return tuple(int(c * factor) for c in color)

        palette = COLOR_NAMES.sample(color_depth).apply(ImageColor.getrgb)
        palette = palette.sort_values(ignore_index=True, key=color_sort_key)
        palette = palette.to_frame(name=str(shade_depth - 1))
        for shade in range(shade_depth - 2, -1, -1):
            palette[str(shade)] = palette[str(shade_depth - 1)].apply(
                partial(color_dive, s=shade)
            )
        return palette

    def random_color(self, row_n=None, col_n=None):
        if row_n is not None:
            return self.palette.iloc[row_n].sample().array[0]
        if col_n is not None:
            return self.palette[str(col_n)].sample().array[0]
        return self.palette.sample().T.sample().array[0]

    def get_color(self, row_n=None, col_n=None):
        p = self.palette.copy()
        if row_n is not None:
            p = p.iloc[row_n]
        if col_n is not None:
            p = p[str(col_n)]
        return p

    def get_image(self):
        h, s = self.palette.shape
        plt_im = Image.new("RGBA", (200 * (h + 1), 200 * s), color="black")
        plt_drw = ImageDraw.Draw(plt_im)
        for x, hue in enumerate(self.palette.itertuples(index=False, name=None)):
            for y, (r, g, b) in enumerate(hue):
                plt_drw.rectangle(
                    [200 * x, 200 * y, 200 * (x + 1), 200 * (y + 1)], fill=(r, g, b)
                )
        plt_drw.rectangle([200 * h, 0, 200 * (h + 1), 100 * s], fill=(255, 255, 255))
        return plt_im.convert("P")


## Helper Functions


def layer_image(base_im, top_im):
    base_im.paste(top_im, top_im)


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


## Testing

test = Vista()
stars = StarField(test)
# [choices([Planet(test), CelestialBody(test)], [3,1])[0] for _ in range(4)]
for _ in range(5):
    Planet(test)
interior = Interior(test, "garden_stroll.png")
test.draw_bodies()
test.save()
im = test.palette.get_image()
im.save("palette.png")
