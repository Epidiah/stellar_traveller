# space_vista.py
#
# Here we assemble a randomly generate gif that is a view of some astronomical
# wonders from various rooms inside our space craft.

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter
from functools import partial
from itertools import product
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
    Call this first, it'll be used as an argument to all your layers.
    Then call your backdrop.
    Followed by all the layers in order of furthest to nearest.
    End with an interior.
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
            self.velocity = RNG.integers(1, 4, endpoint=True)
        if bearing is not None:
            self.bearing = np.radians(bearing)
        else:
            self.bearing = np.radians(RNG.integers(0, 360))
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
            Image.new("RGBA", self.size, color="Black") for dummy in range(self.length)
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
        optimize(
            file_name,
            colors=self.palette.n_colors,
            options=["-O3", "--no-extensions"],
        )


class CelestialBody:
    """
    The parent class of all layers to your vista.
    """

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

    # def drift(self, xy, velocity, frame_n):
    #     """
    #     Takes in a vector of screen coordinates `xy`.
    #     Returns new screen coordinates that assure an Atari wrap-around situation.
    #     """
    #     # Transform screen coordinates to Cartesian because we're not barbarians here.
    #     xy = self.screen_to_cart(xy)
    #     # Our rocket runs on that sweet, sweet linear algebra.
    #     nudge_matrix = (velocity * frame_n * np.array([1, 0])) @ self.vista.rot_matrix
    #     xy = xy - nudge_matrix
    #     # Check for wrap-around.
    #     cosmic_boundaries = self.vista.length * velocity
    #     test_x, test_y = (self.vista.rot_matrix @ xy)
    #     if abs(test_x) > cosmic_boundaries / 2:
    #         xy = xy - np.sign(test_x)*(cosmic_boundaries * np.array([1, 0])) @ self.vista.rot_matrix
    #     return self.cart_to_screen(xy)

    # def drift(self, xy, velocity, frame_n):
    #     """
    #     Takes in a vector of screen coordinates `xy`.
    #     Returns new screen coordinates that assure an Atari wrap-around situation.
    #     """
    #     # Transform screen coordinates to Cartesian because we're not barbarians here.
    #     xy = self.screen_to_cart(xy)
    #     # Our rocket runs on that sweet, sweet linear algebra.
    #     nudge_matrix = self.vista.rot_matrix @ (velocity * frame_n * np.array([1, 0]))
    #     xy = xy - nudge_matrix
    #     # Check for wrap-around.
    #     cosmic_boundaries = self.vista.length * velocity
    #     test_x, test_y = xy @ self.vista.rot_matrix
    #     if abs(test_x) > cosmic_boundaries / 2:
    #         xy = xy + self.vista.rot_matrix @ (np.sign(test_x)*cosmic_boundaries * np.array([1, 0]))
    #     return self.cart_to_screen(xy)


class StarField(CelestialBody):
    def __init__(self, vista, n_stars=500):
        super().__init__(vista)
        self.velocity = self.vista.velocity / 3
        hypotenuse = np.ceil(np.sqrt(self.vista.width ** 2 + self.vista.height ** 2))
        self.leeway = (hypotenuse - self.vista.height) // 2
        self.fieldsize = self.vista.length * self.velocity * hypotenuse
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
        # Put the stars on our path.
        star_map["xy"] = self.rotate_field(star_map)
        return star_map

    def rotate_field(self, star_map):
        """
        A helper function for let_there_be_light that fixes all the star coordinates
        to the path of the starship.
        """
        return star_map[["x", "y"]].apply(
            lambda row: self.cart_to_screen(
                self.screen_to_cart(np.array(row)) @ self.vista.rot_matrix
            ),
            axis=1,
        )

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


class BasePlanet(CelestialBody):
    def __init__(self, vista):
        super().__init__(vista)
        self.velocity = min(6, self.layer // 3 + 1)
        min_x = min_y = -10
        max_x, max_y = 395, 600
        self.x = RNG.integers(min_x, max_x, endpoint=True)
        self.y = RNG.integers(min_y, max_y, endpoint=True)
        self.mass = int(
            RNG.triangular(4, self.vista.height / 3, self.vista.height / np.sqrt(2))
        )
        self.fill = self.vista.palette.random_color()
        self.features = []

    def draw(self, image, drawing_frame, frame_n):
        x, y = self.drift(
            np.array([self.x, self.y]),
            velocity=self.velocity,
            frame_n=frame_n,
        )
        drawing_frame.ellipse(
            [x, y, x + self.mass, y + self.mass], fill=(*self.fill, 255)
        )
        for feature in self.features:
            feature.draw(image, drawing_frame, frame_n)


class SwiftPlanet(BasePlanet):
    def __init__(self, vista, velocity=None):
        super().__init__(vista)
        self.x = 200
        self.y = 200
        if velocity is not None and 1200 % velocity == 0:
            self.velocity = velocity
        else:
            self.velocity = RNG.choice([8, 10, 12, 16], p=[0.3, 0.3, 0.3, 0.1])


class CappedPlanet(BasePlanet):
    def __init__(self, vista):
        super().__init__(vista)
        self.im = self.cap()

    def cap(self):
        new = Image.new("RGBA", (self.mass + 10, self.mass + 10), (0, 0, 0, 0))
        x = RNG.integers(new.size[0])
        y = RNG.integers(new.size[1])
        cap = new.copy()
        dn = ImageDraw.Draw(new)
        dn.ellipse([5, 5, 5 + self.mass, 5 + self.mass], fill=(*self.fill, 255))
        mask = new.copy()
        dc = ImageDraw.Draw(cap)
        dc.ellipse(
            [x, y, x + self.mass // 10, y + self.mass // 10], fill=(255, 255, 255, 112)
        )

        # mask.paste(cap, mask=mask)
        dc.bitmap((0, 0), mask)
        layer_image(new, cap)
        return new  # new.filter(ImageFilter.UnsharpMask())

    def draw(self, image, drawing_frame, frame_n):
        x, y = self.drift(
            np.array([self.x, self.y]),
            velocity=min(6, self.layer // 3 + 1),
            frame_n=frame_n,
        )
        image.paste(self.im, box=(int(x), int(y)))
        for feature in self.features:
            feature.draw(image, drawing_frame, frame_n)


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

    # def recolor(self, im):
    #     # Convert image to RGB mode so we can quantize the colors to the palette.
    #     recolor_im = im.convert("RGB")
    #     rgb = recolor_im.split()
    #     recolor_im = Image.merge('RGB', [rgb[c] for c in self.bgr])
    #     recolor_im = recolor_im.quantize(
    #         colors=32,
    #         palette=self.vista.palette.get_image(),
    #         method=Image.FASTOCTREE,
    #         dither=0,
    #     )
    #     # Convert back and re-establish the alpha channel through some hoops
    #     recolor_im = recolor_im.convert("RGBA")
    #     temp_im = Image.new("RGBA", im.size, color=(0, 0, 0, 0))
    #     temp_im.paste(recolor_im, mask=im)
    #     return temp_im

    def draw(self, image, drawing_frame, frame_n):
        layer_image(image, self.im)


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
        layer_image(image, im)


# class StellarCafe(Interior):
#     def __init__(self, vista, file_path='cafe.png'):
#         super().__init__(vista, file_path)

#     def activate_camera(self, file_path):
#         file_name, ext = file_path.split('.')
#         self.film_strip = [
#             self.recolor(Image.open(f"{file_name}{n}.{ext}")) for n in range(6)
#         ]

#     def draw(self, image, drawing_frame, frame_n):
#         mist = frame_n % 96
#         im = self.film_strip[mist//16]
#         layer_image(image, im)


class StellarCafe(Interior):
    def __init__(self, vista, file_path="cafe.png"):
        super().__init__(vista, file_path)
        self.last_steam = None

    def draw(self, image, drawing_frame, frame_n):
        if (self.last_steam is None) or (frame_n % 4 == 0):
            steam = steamer(self.im, [120, 280, 135, 314])
            self.last_steam = steamer(steam, [225, 286, 240, 320])
        layer_image(image, self.last_steam)

    # def draw(self, image, drawing_frame, frame_n):
    #     if (self.last_steam is None) or (frame_n % 4 == 0):
    #         steam = self.steamer(self.im, [120, 280, 135, 314], frame_n=frame_n)
    #         self.last_steam = self.steamer(steam, [225, 286, 240, 320], frame_n=frame_n)
    #     layer_image(image, self.last_steam)


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

        layer_image(image, im)


class Palette:
    """
    Hue, brightest = 0, darkest = -1
    Shade, brightest = 0, darkest = -1
    """

    def __init__(self, palette=None, color_depth=6, shade_depth=5):
        self.color_depth = color_depth
        self.shade_depth = shade_depth
        if palette is None:
            self.palette = self.random_palette(color_depth=6)
        elif type(palette) == pd.Series:
            self.palette = self.fill_shades(self.sort_palette(palette).to_frame(name=0))
        elif type(palette) == pd.DataFrame:
            self.palette = palette
        self.n_colors = int(2 ** np.ceil(np.log(self.palette.size + 2) / np.log(2)))

    def random_palette(self, color_depth=6):
        """
        Returns a pd.DataFrame with colors as the rows, sorted from brightest (hue=0)
        to darkest (hue=color_depth-1), and shades of the color as the columns,
        sorted from brightest (shade=0) to darkest (shade=shade_depth-1).
        """
        palette = COLOR_NAMES.sample(color_depth).apply(ImageColor.getrgb)
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

    def random_color(self, hue=None, shade=None):
        if hue is not None:
            return self.palette.iloc[hue].sample().array[0]
        if shade is not None:
            return self.palette[:, shade].sample().array[0]
        return self.palette.sample().T.sample().iloc[0, 0]

    def get_color(self, hue=None, shade=None):
        if hue is not None and shade is not None:
            return self.palette.iloc[hue, shade]
        elif hue is not None:
            return self.palette.iloc[hue, :]
        elif shade is not None:
            return self.palette.iloc[:, shade]
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


class SplitComplementaryPalette(Palette):
    def random_palette(self, color_depth=6):
        base_hue = RNG.integers(0, 360)
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
    def __init__(self, palette=None, hue=None, color_depth=4, shade_depth=4):
        self.hue = hue % 360
        super().__init__(palette=None, color_depth=4, shade_depth=4)
        self.n_colors = 16

    def random_palette(self, color_depth=3):
        if self.hue is not None:
            return self.build_palette(self.hue)
        base_hue = RNG.integers(0, 360)
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
    def __init__(self, palette=None, color_depth=3, shade_depth=4):
        super().__init__(palette=None, color_depth=3, shade_depth=4)

    def random_palette(self, color_depth=3):
        base_hue = RNG.integers(0, 360)
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


def layer_image(base_im, top_im):
    base_im.paste(top_im, top_im)

def steamer(im, steam_box):
        # Make some noise!
        mist = Image.effect_noise(
            [steam_box[2] - steam_box[0], steam_box[3] - steam_box[1]],
            RNG.integers(290, 311),
        ).convert("RGBA")
        # Confine it to a randomly sized ellipse
        shifts = RNG.integers(-4,5,4)
        mask = Image.new('RGBA', mist.size, (0,0,0,0))
        drw_mask = ImageDraw.Draw(mask)
        drw_mask.ellipse(((0,0)+ mist.size + shifts).tolist(), 'white')
        steams = Image.new('RGBA', im.size)
        steams.paste(mist, box=steam_box, mask=mask)
        steams = steams.effect_spread(2).filter(ImageFilter.GaussianBlur(3))
        steams.putalpha(64)
        return ImageChops.lighter(im,steams)


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

# p = Palette()
# print(p.palette)

# p = pd.Series([
#         "Gold",
#         "Tomato",
#         "RebeccaPurple",
#         "SpringGreen",
#         "MediumSlateBlue",
#         "Maroon"
#     ]).apply(ImageColor.getrgb)
# p = pd.Series(
#     [(RNG.integers(256), RNG.integers(256), RNG.integers(256)) for dummy in range(6)]
# )
# test = Vista(palette=Palette(p))
# scp = SplitComplementaryPalette()
# mp = MonoPalette16(hue=240)
test = Vista()
stars = StarField(test, 500)
for dummy in range(RNG.integers(1, 13)):
    BasePlanet(test)
# for v in range(1,4):
#     SwiftPlanet(test, velocity=v*-2)

# Interior
# interior = Interior(test, "observation_windows.png")
# interior = AstroGarden(test)
# interior = Engineering(test)
interior = StellarCafe(test)
test.draw_bodies()
test.save()
im = test.palette.get_image()
im.save("palette.png")
# print("\a")
