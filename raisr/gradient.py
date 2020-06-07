import numpy as np
import numpy.core.numeric as _nx
from numba import jit
from numpy.core.multiarray import typeinfo, dtype

@jit(nopython=True)
def gradient_helper(f):
    # f = np.asanyarray(f)
    f = np.array(f)
    N = f.ndim  # number of dimensions

    axes = range(N)

    len_axes = len(axes)

    dx = [1.0] * len_axes

    edge_order = 1

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype
    # if issubdtype(otype, np.inexact):
    #     pass
    # else:
    #     # All other types convert to floating point.
    #     # First check if f is a numpy integer type; if so, convert f to float64
    #     # to avoid modular arithmetic when computing the changes in f.
    #     if issubdtype(otype, np.integer):
    #         f = f.astype(np.float64)
    #     otype = np.float64
    
    f = f.astype(np.float64)
    otype = np.float64

    result = []

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required.")
        # result allocation
        out = None

        # spacing for the current axis
        uniform_spacing = np.array(ax_dx).ndim == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)
        
        f_slice2 = f
        for x in slice2:
            f_slice2 = f_slice2[x]

        f_slice3 = f
        for x in slice3:
            f_slice3 = f_slice3[x]    
        
        f_slice4 = f
        for x in slice4:
            f_slice4 = f_slice4[x]
                
        if uniform_spacing:
            out = (f_slice4 - f_slice2) / (2. * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2)/(dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            a.shape = b.shape = c.shape = shape
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out = a * f_slice2 + b * f_slice3 + c * f_slice4

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out = (f_slice2 - f_slice3) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out = (f_slice2 - f_slice3) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2. / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2. * dx1 + dx2)/(dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = - dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out = a * f_slice2 + b * f_slice3 + c * f_slice4

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2. / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = - (dx2 + dx1) / (dx1 * dx2)
                c = (2. * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out = a * f_slice2 + b * f_slice3 + c * f_slice4
        
        result.append((out, slice1))

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)
    
    return result

def gradient(f):
    slices, otype = gradient_helper(f)
    outvals = []
    for res, slice1 in slices:
        out = np.empty_like(f, dtype=otype)
        out[tuple(slice1)] = res

        outvals.append(out)

    if f.ndim == 1:
        return outvals[0]
    else:
        return outvals

print(gradient([1,2,3,4]))