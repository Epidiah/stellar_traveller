class PlanetaryFeature:
    """
    A basic class to subclass other planetary features from.
    """

    def __init__(self, planet):
        self.planet = planet
        self.sys_width = 0
        self.sys_height = 0

    def background(self, center, im, draw_im):
        """
        Called once to draw a feature that might be occluded by the planet on the
        planet's base image.
        Only used by features that do not move with respect to the planet.
        """
        pass

    def foreground(self, center, im, draw_im, planet_mask):
        """
        Called once to draw a feature that will not be occluded by the planet on the
        planet's base image.
        Only used by features that do not move with respect to the planet.
        """
        pass

    def drifting_background(self, x, y, image, drawing_frame, frame_n):
        """
        Called every frame to draw a feature that might be occuluded by the planet
        as it moves with respect to that planet.
        """
        pass

    def drifting_foreground(self, x, y, image, drawing_frame, frame_n):
        """
        Called every frame to draw a feature that will not be occuluded by the planet
        as it moves with respect to that planet.
        """
        pass
      