import numpy as np
from .interior import Interior
from PIL import Image, ImageDraw

class Engineering(Interior):
    """
    A view through a heptagonal portal in engineering. Includes animations for various
    readouts and the occasional flickering lighting effect.
    """

    RNG = np.random.default_rng()

    def __init__(self, vista, file_path):
        super().__init__(vista, file_path, "engineering.png")
        self.osc_bg = self.vista.palette.get_color(hue=-1, shade=-2)
        self.osc_shine = self.vista.palette.get_color(hue=-1, shade=1)
        self.osc_fg0 = self.vista.palette.get_color(hue=-2, shade=0)
        self.osc_fg1 = self.vista.palette.get_color(hue=2, shade=0)
        self.osc_lines = self.vista.palette.get_color(hue=1, shade=1)
        self.diagnostics = self.RNG.choice(["sins", "soothes"], p=[0.75, 0.25])

    def activate_camera(self):
        """
        The method for assembling the film_strip of interior images and ensuring each
        image matches the Vista's palette.
        """
        file_name, ext = self.image_name.split(".")
        self.flicker = self.RNG.choice(["s-", "f-"], p=[0.8, 0.2])
        self.film_strip = [
            self.recolor(
                Image.open(self.file_path / f"{self.flicker}{file_name}{n}.{ext}")
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
            op = self.RNG.choice([0, 1], p=[0.75, 0.25])
            im = self.film_strip[op]
        elif 48 <= blink < 72:
            op = self.RNG.choice([2, 3], p=[0.75, 0.25])
            im = self.film_strip[op]
        else:
            op = self.RNG.choice([1, 2], p=[0.75, 0.25])
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
