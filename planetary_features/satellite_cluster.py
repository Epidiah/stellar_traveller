from .planetary_feature import PlanetaryFeature
from .satellite import Satellite
import numpy as np

class SatelliteCluster(PlanetaryFeature):
    """
    A container for a planet's satellites that keeps them in line so that they overlap
    each other in the proper order and maintain enough of a distance to avoid other
    odd behaviors.
    """

    def __init__(self, planet):
        super().__init__(planet)
        self.band = set(range(self.planet.vista.length))
        self.satellites = self.create_sats()

    def create_sats(self):
        """
        Randomly determines the number of satellites orbiting a planet and assembles
        a list of those satellites.

        Returns a list of all the satellites
        """
        satellites = []
        n_satellites = self.planet.vista.coords.choice(
            [1, 2, 3, self.planet.vista.coords.integers(3, 12)],
            p=[0.625, 0.125, 0.125, 0.125],
        )
        for moon in range(n_satellites):
            satellites.append(Satellite(self.planet, self))
        return satellites

    def get_start_location(self, sat):
        """
        Called by the Satellite to find where in its orbit around the planet it will
        start when we begin the animation. This method sets all the satellites in the
        cluster with start_locations that ensure they will not overlap when directly
        in front of the planet. This is necessary because this is the spot where two
        satellites might appear to move through each other when one moves ahead of
        the other in the sorting.
        """
        start_location = self.planet.vista.coords.choice(list(self.band))
        while set(range(start_location, start_location + sat.mass)) - self.band:
            start_location = self.planet.vista.coords.choice(list(self.band))
        self.band -= set(range(start_location, start_location + sat.mass))
        return start_location

    def drifting_background(self, x, y, image, drawing_frame, frame_n):
        """
        Sorts the satellites in the background so that those closer to the viewer are
        drawn last. Then calls on the Satellite's drifting_background method to draw it.
        """
        # The greater abs(cos) should be in front
        def sort_key(sat):
            phase = (
                (sat.orbit_velocity * frame_n + sat.start_location)
                / sat.planet.vista.length
                * 2
                * np.pi
            )
            return abs(np.cos(phase))

        for sat in sorted(self.satellites, key=sort_key):
            sat.drifting_background(x, y, image, drawing_frame, frame_n)

    def drifting_foreground(self, x, y, image, drawing_frame, frame_n):
        """
        Sorts the satellites in the foreground so that those closer to the viewer are
        drawn last. Then calls on the Satellite's drifting_foreground method to draw it.
        """
        # The lesser abs(cos) should be in front
        def sort_key(sat):
            phase = (
                (sat.orbit_velocity * frame_n + sat.start_location)
                / sat.planet.vista.length
                * 2
                * np.pi
            )
            return -abs(np.cos(phase))

        for sat in sorted(self.satellites, key=sort_key):
            sat.drifting_foreground(x, y, image, drawing_frame, frame_n)
