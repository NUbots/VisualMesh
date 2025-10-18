import math

import tensorflow as tf


def _inverse_coefficents(k):
    return [
        -k[0],
        3.0 * (k[0] * k[0]) - k[1],
        -12.0 * (k[0] * k[0]) * k[0] + 8.0 * k[0] * k[1],
        55.0 * (k[0] * k[0]) * (k[0] * k[0]) - 55.0 * (k[0] * k[0]) * k[1] + 5.0 * (k[1] * k[1]),
    ]


def _distort(r, k):
    ik = _inverse_coefficents(k)
    return r * (
        1.0
        + ik[0] * (r * r)
        + ik[1] * ((r * r) * (r * r))
        + ik[2] * ((r * r) * (r * r)) * (r * r)
        + ik[3] * ((r * r) * (r * r)) * ((r * r) * (r * r))
    )


def _equidistant_r(theta, f):
    return f * theta


def _rectilinear_r(theta, f):
    return f * tf.math.tan(tf.clip_by_value(theta, 0.0, math.pi * 0.5))


def _equisolid_r(theta, f):
    return 2.0 * f * tf.math.sin(theta * 0.5)


def _undistort(r, k):
    return r * (1.0 + k[0] * (r * r) + k[1] * ((r * r) * (r * r)))


def _equidistant_theta(r, f):
    return r / f


def _rectilinear_theta(r, f):
    return tf.math.atan(r / f)


def _equisolid_theta(r, f):
    return 2.0 * tf.math.asin(r / (2.0 * f))


def project(V, dimensions, projection, f, centre, k):

    #  Perform the projection math
    theta = tf.math.acos(V[:, 0])
    rsin_theta = tf.math.rsqrt(1.0 - tf.square(V[:, 0]))
    if projection == "RECTILINEAR":
        r_u = _rectilinear_r(theta, f)
    elif projection == "EQUISOLID":
        r_u = _equisolid_r(theta, f)
    elif projection == "EQUIDISTANT":
        r_u = _equidistant_r(theta, f)
    else:
        r_u = tf.zeros_like(theta)

    r_d = _distort(r_u, k)

    # Screen as y,x
    screen = tf.stack([r_d * V[:, 2] * rsin_theta, r_d * V[:, 1] * rsin_theta], axis=1)

    # Sometimes floating point error makes x > 1.0
    # In this case we are basically the centre of the screen anway
    screen = tf.where(tf.math.is_finite(screen), screen, 0.0)

    # Convert to pixel coordinates
    return (tf.cast(dimensions, tf.float32) * 0.5) - screen - centre


def unproject(px, dimensions, projection, f, centre, k):
    """
    Unprojects pixel coordinates into unit vectors.

    Args:
        px: [N, 2] pixel coordinates (y, x format to match dimensions)
        dimensions: [2] image dimensions [height, width]
        projection: string projection type ("RECTILINEAR", "EQUISOLID", "EQUIDISTANT")
        f: focal length
        centre: [2] lens centre offset
        k: [2] distortion coefficients

    Returns:
        [N, 3] unit vectors in camera space [x, y, z] where x is forward
    """
    # Transform to screen coordinates (centre of screen is origin, y up, x right)
    screen = (tf.cast(dimensions, tf.float32) * 0.5) - px - centre

    # Get radial distance
    r_d = tf.norm(screen, axis=-1)

    # Handle zero radius case (looking straight ahead)
    is_zero = tf.equal(r_d, 0.0)

    # Undistort
    r_u = _undistort(r_d, k)

    # Convert back to theta based on projection type
    if projection == "RECTILINEAR":
        theta = _rectilinear_theta(r_u, f)
    elif projection == "EQUISOLID":
        theta = _equisolid_theta(r_u, f)
    elif projection == "EQUIDISTANT":
        theta = _equidistant_theta(r_u, f)
    else:
        theta = tf.zeros_like(r_u)

    # Convert to unit vector
    sin_theta = tf.math.sin(theta)
    cos_theta = tf.math.cos(theta)

    # Avoid division by zero
    safe_r_d = tf.where(is_zero, 1.0, r_d)

    # Unit vector components [x forward, y left, z up]
    x = cos_theta
    y = tf.where(is_zero, 0.0, sin_theta * screen[:, 1] / safe_r_d)  # screen[:, 1] is x component
    z = tf.where(is_zero, 0.0, sin_theta * screen[:, 0] / safe_r_d)  # screen[:, 0] is y component

    return tf.stack([x, y, z], axis=-1)
