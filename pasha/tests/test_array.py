
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

import pytest

import numpy as np
from pasha.context import MapContext, HeapContext


class _AllocTestContext(MapContext):
    """Simple context implementing a minimal allocation machinery."""

    def empty(self, shape, dtype=np.float64, order='C'):
        return np.empty(shape, dtype=dtype, order=order)


@pytest.mark.parametrize(
    'ctx_cls', [_AllocTestContext, HeapContext],
    ids=['MapContext', 'HeapContext'])
@pytest.mark.parametrize(
    ['method', 'expected', 'extra_kwargs'],
    [('empty', None, {}), ('zeros', 0.0, {}), ('ones', 1.0, {}),
     ('full', 42, {'fill_value': 42})],
    ids=['empty', 'zeros', 'ones', 'full'])
@pytest.mark.parametrize(
    ['shape_in', 'shape_out'], [(3, (3,)), ((3, 2), (3, 2))],
    ids=['int', 'tuple'])
@pytest.mark.parametrize(
    ['order_in', 'order_flag'], [('C', 'c'), ('F', 'f')], ids=['C', 'Fortran'])
def test_array(ctx_cls, method, expected, extra_kwargs, shape_in, shape_out,
               order_in, order_flag):
    """Test direct allocation."""

    ctx = ctx_cls(num_workers=3)

    array = getattr(ctx, method)(shape_in, dtype=np.uint32, order=order_in,
                                 **extra_kwargs)
    assert array.shape == shape_out
    assert array.dtype == np.uint32
    assert getattr(array.flags, f'{order_flag}_contiguous')

    if expected is not None:
        np.testing.assert_allclose(array, expected)


@pytest.mark.parametrize(
    'ctx_cls', [_AllocTestContext, HeapContext],
    ids=['MapContext', 'HeapContext'])
@pytest.mark.parametrize(
    ['method', 'expected', 'extra_kwargs'],
    [('empty', None, {}), ('zeros', 0.0, {}), ('ones', 1.0, {}),
     ('full', 42, {'fill_value': 42})],
    ids=['empty', 'zeros', 'ones', 'full'])
@pytest.mark.parametrize(
    ['shape_in', 'shape_out'], [(None, (2, 3, 4)), ((6, 4), (6, 4))],
    ids=['keep_shape', 'override_shape'])
@pytest.mark.parametrize(
    ['order_in', 'order_out', 'order_flag'],
    [('C', 'C', 'c'), ('F', 'C', 'c'), ('C', 'F', 'f'), ('F', 'F', 'f'),
     ('C', 'A', 'c'), ('F', 'A', 'f'), ('C', 'K', 'c'), ('F', 'K', 'f')],
    ids=['C/C', 'F/C', 'C/F', 'F/F', 'C/A', 'F/A', 'C/K', 'F/K'])
def test_array_like(ctx_cls, method, expected, extra_kwargs, shape_in,
                    shape_out, order_in, order_out, order_flag):
    """Test allocation based on existing array."""

    ctx = ctx_cls(num_workers=3)

    array_in = np.random.rand(2, 3, 4).astype(np.float32, order=order_in)
    array_out = getattr(ctx, f'{method}_like')(array_in, order=order_out,
                                               shape=shape_in, **extra_kwargs)

    assert array_out.shape == shape_out
    assert array_out.dtype == array_out.dtype
    assert getattr(array_out.flags, f'{order_flag}_contiguous')

    if expected is not None:
        np.testing.assert_allclose(array_out, expected)


@pytest.mark.parametrize(
    'ctx_cls', [_AllocTestContext, HeapContext],
    ids=['MapContext', 'HeapContext'])
@pytest.mark.parametrize(
    ['method', 'expected', 'extra_kwargs'],
    [('empty', None, {}), ('zeros', 0.0, {}), ('ones', 1.0, {}),
     ('full', 42, {'fill_value': 42})],
    ids=['empty', 'zeros', 'ones', 'full'])
@pytest.mark.parametrize(
    ['shape_in', 'shape_out'], [(3, (3,)), ((3, 2), (3, 2))],
    ids=['int', 'tuple'])
@pytest.mark.parametrize(
    ['order_in', 'order_flag'], [('C', 'c'), ('F', 'f')], ids=['C', 'Fortran'])
def test_array_per_worker(ctx_cls, method, expected, extra_kwargs, shape_in,
                          shape_out, order_in, order_flag):
    """Test allocation per worker."""

    ctx = ctx_cls(num_workers=3)

    array = getattr(ctx, f'{method}_per_worker')(
        shape_in, dtype=np.uint32, order=order_in, **extra_kwargs)
    assert array.shape == (ctx.num_workers,) + shape_out
    assert array.dtype == np.uint32
    assert getattr(array.flags, f'{order_flag}_contiguous')

    if expected is not None:
        np.testing.assert_allclose(array, expected)
