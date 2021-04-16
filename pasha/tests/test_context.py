
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

import pytest

import numpy as np
import pasha as psh
from pasha.context import MapContext, ProcessContext
from pasha.functor import Functor


@pytest.mark.parametrize(
    'ctx_cls', [MapContext, ProcessContext],
    ids=['MapContext', 'ProcessContext'])
@pytest.mark.parametrize(
    ['shape_in', 'shape_out'], [(3, (3,)), ((3, 2), (3, 2))],
    ids=['int', 'tuple'])
@pytest.mark.parametrize(
    ['dtype_in', 'dtype_out'], [(None, np.float64), (np.uint16, np.uint16)],
    ids=['default_dtype', 'fix_dtype'])
@pytest.mark.parametrize(
    ['order_in', 'order_flag'], [('C', 'c'), ('F', 'f')], ids=['C', 'Fortran'])
@pytest.mark.parametrize(
    'fill', [None, 0, 1, 42], ids=['empty', 'zero', 'one', 'real'])
@pytest.mark.parametrize(
    'per_worker', [False, True], ids=['single', 'per_worker'])
def test_alloc_direct(ctx_cls, shape_in, shape_out, dtype_in, dtype_out,
                      order_in, order_flag, fill, per_worker):
    """Test direct allocation."""

    ctx = ctx_cls(num_workers=3)

    array = ctx.alloc(shape=shape_in, dtype=dtype_in, order=order_in,
                      fill=fill, like=None, per_worker=per_worker)

    if per_worker:
        if order_in == 'C':
            assert array.shape == (ctx.num_workers,) + shape_out
        elif order_in == 'F':
            assert array.shape == shape_out + (ctx.num_workers,)
    else:
        assert array.shape == shape_out

    assert array.dtype == dtype_out
    assert getattr(array.flags, f'{order_flag}_contiguous')

    if fill is not None:
        np.testing.assert_allclose(array, fill)


@pytest.mark.parametrize(
    'ctx_cls', [MapContext, ProcessContext],
    ids=['MapContext', 'ProcessContext'])
@pytest.mark.parametrize(
    ['shape_in', 'shape_out'], [(None, (2, 3, 4)), ((6, 4), (6, 4))],
    ids=['keep_shape', 'fix_shape'])
@pytest.mark.parametrize(
    ['dtype_in', 'dtype_out'], [(None, np.float32), (np.uint16, np.uint16)],
    ids=['keep_dtype', 'fix_dtype'])
@pytest.mark.parametrize(
    ['order_like', 'order_in', 'order_flag'],
    [('C', None, 'c'), ('F', None, 'f'), ('C', 'C', 'c'), ('F', 'F', 'f'),
     ('C', 'F', 'f'), ('F', 'C', 'c')],
    ids=['keep_C', 'keep_F', 'fix_CC', 'fix_FF', 'fix_CF', 'fix_FC'])
@pytest.mark.parametrize(
    'per_worker', [False, True], ids=['single', 'per_worker'])
@pytest.mark.parametrize(
    'fill', [None, 0, 1, 42], ids=['empty', 'zero', 'one', 'real'])
def test_alloc_like(ctx_cls, shape_in, shape_out, dtype_in, dtype_out,
                    order_like, order_in, order_flag, fill, per_worker):
    """Test allocation based on existing ArrayLike."""

    ctx = ctx_cls(num_workers=3)

    array_in = np.random.rand(2, 3, 4).astype(np.float32, order=order_like)
    array_out = ctx.alloc(shape=shape_in, dtype=dtype_in, order=order_in,
                          fill=fill, like=array_in, per_worker=per_worker)

    if per_worker:
        if order_in == 'C':
            assert array_out.shape == (ctx.num_workers,) + shape_out
        elif order_in == 'F':
            assert array_out.shape == shape_out + (ctx.num_workers,)
    else:
        assert array_out.shape == shape_out

    assert array_out.dtype == dtype_out
    assert getattr(array_out.flags, f'{order_flag}_contiguous')

    if fill is not None:
        np.testing.assert_allclose(array_out, fill)


def test_run_worker():
    """Test default implementation in MapContext.run_worker()."""

    _kernel_i = 0

    def kernel(worker_id, index, value):
        nonlocal _kernel_i

        assert worker_id == 0
        assert index == _kernel_i
        assert value == -_kernel_i

        _kernel_i += 1

    class TestFunctor(Functor):
        def iterate(self, share):
            yield from enumerate(share)

    functor = TestFunctor()
    MapContext.run_worker(kernel, functor, range(0, -10, -1), 0)


@pytest.mark.parametrize(
    'ctx', [psh, psh.SerialContext(), psh.ThreadContext(),
            psh.ProcessContext()],
    ids=['default', 'serial', 'thread', 'process'])
def test_map(ctx):
    """Test map operation for each context type."""

    inp = np.random.rand(100)
    outp = ctx.alloc(shape=inp.shape, dtype=inp.dtype)

    def multiply(worker_id, index, value):
        outp[index] = 3 * value

    ctx.map(multiply, inp)
    np.testing.assert_allclose(outp, inp*3)


def test_initial_default_context():
    """Test the initial default context type.

    All following tests may not be moved before this one in the module
    as they change the default context!
    """

    assert isinstance(psh.get_default_context(), psh.ProcessContext)


def test_set_default_context_direct():
    """Test setting default context directly."""

    ctx = psh.SerialContext()
    psh.set_default_context(ctx)

    assert psh.get_default_context() is ctx


_context_strs = {'serial': psh.SerialContext,
                 'threads': psh.ThreadContext,
                 'processes': psh.ProcessContext}


@pytest.mark.parametrize(
    ['ctx_str', 'expected_type'], _context_strs.items(),
    ids=list(_context_strs.keys()))
def test_set_default_context_string(ctx_str, expected_type):
    """Test setting default context by string."""

    psh.set_default_context(ctx_str)
    assert isinstance(psh.get_default_context(), expected_type)
