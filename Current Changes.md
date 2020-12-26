# 2020-12-26
Commit Message: "Brought flame_like in line with fog_like."
* Switched the steam on the mugs in the StellarCafe to flame_like_alpha and slowed the animation a bit. Looks better and might be a bit more reliable.

# 2020-12-25
Commit Message: "Adjustments to the tilt, velocity, and dimensions of planets."
* Made sure everyone was using sin on Planet.tilt_from_y and not just a fraction over 90.
* Moved velocity into the Vista. It is given out when a body is added to the Vista, just in case I want to make it a list of possible velocities instead or hand them out under some specific criteria.
* Changed Planet.velocity to be a fraction of Vista.velocity plus a fraction of the Planet.layer to give each planet a unique but slower velocity.
* Adjusted the Planet's sys_dimensions to reflect the actual size of the Planet.im.
