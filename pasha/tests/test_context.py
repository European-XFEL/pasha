
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

import pytest

import numpy as np
import pasha as psh
from pasha.context import MapContext, HeapContext
from pasha.functor import Functor


class _AllocTestContext(MapContext):
    """Simple context implementing a minimal allocation machinery."""

    def empty(self, shape, dtype=np.float64):
        return np.empty(shape, dtype=dtype)


@pytest.mark.parametrize(
    'ctx_cls', [_AllocTestContext, HeapContext],
    ids=['MapContext', 'HeapContext'])
@pytest.mark.parametrize(
    ['method', 'expected'], [('empty', None), ('zeros', 0.0), ('ones', 1.0)],
    ids=['empty', 'zeros', 'ones'])
@pytest.mark.parametrize(
    ['shape_in', 'shape_out'], [(3, (3,)), ((3, 2), (3, 2))],
    ids=['int', 'tuple'])
def test_array(ctx_cls, method, expected, shape_in, shape_out):
    """Test direct allocation."""

    ctx = ctx_cls(num_workers=3)

    array = getattr(ctx, method)(shape_in, np.uint32)
    assert array.shape == shape_out
    assert array.dtype == np.uint32

    if expected is not None:
        np.testing.assert_allclose(array, expected)


@pytest.mark.parametrize(
    'ctx_cls', [_AllocTestContext, HeapContext],
    ids=['MapContext', 'HeapContext'])
@pytest.mark.parametrize(
    ['method', 'expected'], [('empty', None), ('zeros', 0.0), ('ones', 1.0)],
    ids=['empty', 'zeros', 'ones'])
def test_array_like(ctx_cls, method, expected):
    """Test allocation based on existing array."""

    ctx = ctx_cls(num_workers=3)

    array_in = np.random.rand(2, 3, 4).astype(np.float32)
    array_out = getattr(ctx, f'{method}_like')(array_in)

    assert array_in.shape == array_out.shape
    assert array_in.dtype == array_out.dtype

    if expected is not None:
        np.testing.assert_allclose(array_out, expected)


@pytest.mark.parametrize(
    'ctx_cls', [_AllocTestContext, HeapContext],
    ids=['MapContext', 'HeapContext'])
@pytest.mark.parametrize(
    ['method', 'expected'], [('empty', None), ('zeros', 0.0), ('ones', 1.0)],
    ids=['empty', 'zeros', 'ones'])
@pytest.mark.parametrize(
    ['shape_in', 'shape_out'], [(3, (3,)), ((3, 2), (3, 2))],
    ids=['int', 'tuple'])
def test_array_per_worker(ctx_cls, method, expected, shape_in, shape_out):
    """Test allocation per worker."""

    ctx = ctx_cls(num_workers=3)

    array = getattr(ctx, f'{method}_per_worker')(shape_in, np.uint32)
    assert array.shape == (ctx.num_workers,) + shape_out
    assert array.dtype == np.uint32

    if expected is not None:
        np.testing.assert_allclose(array, expected)


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
    outp = ctx.empty(inp.shape, inp.dtype)

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
