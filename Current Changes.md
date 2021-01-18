# 2021-01-18 a couple moments or two after that
Commit Message: "Tidying up another bug"
* Just doing my job.

# 2021-01-18 moments later
Commit Message: "Brought spacescape function up to date."
* Had random_spacescape working, but needed to update spacescape

# 2021-01-18
Commit Message: "Switched from GIFs to MP4s."
* Continuing to add comments and documentation to space_vista.py up to line 1516 now.
* Moved random selection of the bodies in view inside the Vista class
* Added distances to the planets to offer a little more room to adjust how they appear and behave.
* Made the peak in the random distribution of planet masses based on their distance, so that further planets tended to appear smaller.
* Can now save the animations as .MP4 which may be better suited to this endeavor—less blinking planets, more control over the flame-like and fog-like effects, smaller file formats, still loops on Twitter, and the possibility of adding ambient music in the future.
* Updated bot.py to Tweet out videos instead of gifs. Death to gifs.
* Adjusted ExtraVehicularActivity fog so that it looks better in the video format.
* Adjusted StellarCafe steaming mugs so that they look better in the video format.
* audio_test.py added to see if we can get audio into these videos. Turns out, we can!
* Swapped the order the orbit_matrix is multiplied by the xy coordinates of the satellite because I suspect that'll bring the moon orbits in line with the rings. OH AND CONVERTED DEGREES TO RADIANS BECAUSE, DUH!
* Made a Coordinates a parameter for Palette so that any randomness in the palette is dependent on the location in the galaxy.
* Broadened the wiggle room irregular moons have in their frame. Hopefully now none of them will be cut off by their borders.

# 2021-01-12 a half hour later
Commit Message: "Uncommented out the various interior choices in random_spacescape."
* Correcting the most recent commit which accidentally left all but the EVA interior commented out.

# 2021-01-12
Commit Message: "Preparing for the leap to interactivity."
* Added numerous comments and documentation to space_vista.py to make things a bit clearer for outside observers and future Eppies who might want to know what's going on. Left off at line 929.
* Moved somee status content inside the classes that generate it.
* Add more random number generators for events not tied to your galactic coordinates.
* Changed random velocity generation so that it favors slower speeds.
* Created SatelliteCluster object to hold all a planet's satellites and make sure they don't overlap in weird ways as they orbit the planet.
* Fixed the rotation of the irregular moon so it wouldn't crop at the corners of the image.

# 2020-12-26
Commit Message: "Brought flame_like in line with fog_like."
* Switched the steam on the mugs in the StellarCafe to flame_like_alpha and slowed the animation a bit. Looks better and might be a bit more reliable.

# 2020-12-25
Commit Message: "Adjustments to the tilt, velocity, and dimensions of planets."
* Made sure everyone was using sin on Planet.tilt_from_y and not just a fraction over 90.
* Moved velocity into the Vista. It is given out when a body is added to the Vista, just in case I want to make it a list of possible velocities instead or hand them out under some specific criteria.
* Changed Planet.velocity to be a fraction of Vista.velocity plus a fraction of the Planet.layer to give each planet a unique but slower velocity.
* Adjusted the Planet's sys_dimensions to reflect the actual size of the Planet.im.
