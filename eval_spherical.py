'''
To implement https://github.com/sxyu/svox2/blob/master/svox2/utils.py#L115

voxel grid with spherical harmonics

https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc

'''


from spherical_constants import COEFFICIENTS

def eval_sphericals_function(k, d):
    x = d[..., 0:1]
    y = d[..., 1:2]
    z = d[..., 2:3]

    result = (
        COEFFICIENTS['C0'] * k[..., 0] +
        COEFFICIENTS['C1'] * y * k[..., 1] +
        COEFFICIENTS['C1'] * z * k[..., 2] +
        COEFFICIENTS['C1'] * x * k[..., 3] +
        COEFFICIENTS['C2'] * x * y * k[..., 4] +
        COEFFICIENTS['C3'] * y * z * k[..., 5] +
        COEFFICIENTS['C4'] * (2.0 * z * z - x * x - y * y) * k[..., 6] +
        COEFFICIENTS['C5'] * x * z * k[..., 7] +
        COEFFICIENTS['C6'] * (x * x - y * y) * k[..., 8]
    )

    return result