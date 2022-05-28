import numpy as np

class CelestialBody:
    """
    The parent class of all layers to your vista.
    """

    # So I don't have to figure this out every time I flip from screen to cartesian
    FLIP_Y = np.array([[1, 0], [0, -1]])

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
        return (xy - self.vista.center) @ self.FLIP_Y

    def cart_to_screen(self, xy):
        """
        Assumes `xy` is an array-like object representing the vector [x, y] in
        Cartesian coordinates where the origin is at the middle of the screen
        and positive y goes up.

        Returns a numpy array representing the vector [x, y] in screen coordinates
        where the origin is at the top-left corner of the screen and positive y
        goes down.
        """
        return xy @ self.FLIP_Y + self.vista.center

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
