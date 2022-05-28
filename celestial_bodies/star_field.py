from .celestial_body import CelestialBody
import numpy as np
import pandas as pd

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

