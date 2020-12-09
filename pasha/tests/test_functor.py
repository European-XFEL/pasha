
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

import pytest

import numpy as np
import xarray as xr
import pasha as psh
from pasha.functor import Functor


_supported_try_wrap = [
    ([1, 2, 3], psh.SequenceFunctor),
    ((1, 2, 3), psh.SequenceFunctor),
    (np.arange(3), psh.NdarrayFunctor),
    (xr.DataArray(np.arange(3)), psh.DataArrayFunctor)]


def test_Functor_subtype_register():
    """Test registering a Functor subtype for value wrapping."""

    test_value = object()

    with pytest.raises(ValueError):
        Functor.try_wrap(test_value)

    class TestFunctor(Functor):
        @classmethod
        def wrap(cls, value):
            if value is test_value:
                return cls()

    functor = Functor.try_wrap(test_value)
    assert isinstance(functor, TestFunctor)


@pytest.mark.parametrize(
    'value, expected_type', _supported_try_wrap,
    ids=[type(x).__name__ for x, _ in _supported_try_wrap])
def test_Functor_try_wrap_supported(value, expected_type):
    """Test wrapping supported types."""

    assert isinstance(Functor.try_wrap(value), expected_type)


def test_Functor_try_wrap_unsupported():
    """Test wrapping unsupported types."""

    with pytest.raises(ValueError):
        Functor.try_wrap({1, 2, 3})


_sequence_like_values = [
    ([1, 2, 3, 4], psh.SequenceFunctor),
    ((1, 2, 3, 4), psh.SequenceFunctor),
    (np.arange(4)+1, psh.NdarrayFunctor),
    (xr.DataArray(np.arange(4)+1), psh.DataArrayFunctor),
]


@pytest.mark.parametrize(
    ['value', 'expected_type'], _sequence_like_values,
    ids=[type(x).__name__ for x, _ in _sequence_like_values])
def test_sequence_like_functor(value, expected_type):
    """Test SequenceFunctor and its subtypes."""

    functor = Functor.try_wrap(value)

    assert isinstance(functor, expected_type)
    np.testing.assert_allclose(functor.split(2), [[0, 1], [2, 3]])
    np.testing.assert_allclose(list(functor.iterate([1, 2])), [[1, 2], [2, 3]])


@pytest.mark.parametrize(
    ['axis', 'expected_value'], [(0, [0, 1, 2]), (1, [0, 3, 6])],
    ids=['axis=0', 'axis=1'])
def test_NdarrayFunctor_axis(axis, expected_value):
    """Test explicit iteration axis for NdarrayFunctor."""

    inp = np.arange(9).reshape(3, 3)

    functor = psh.NdarrayFunctor(inp, axis=axis)
    np.testing.assert_allclose(functor.split(3), [[0], [1], [2]])

    _, value = next(iter(functor.iterate([0])))
    assert isinstance(value, np.ndarray)
    np.testing.assert_allclose(value, expected_value)


@pytest.mark.parametrize(
    ['dim', 'expected_value'], [('a', [0, 1, 2]), ('b', [0, 3, 6])],
    ids=['dim=a', 'dim=b'])
def test_DataArrayFunctor_dim(dim, expected_value):
    """Test explicit iteration dimension for DataArrayFunctor."""

    inp = xr.DataArray(np.arange(9).reshape(3, 3), dims=['a', 'b'])

    functor = psh.DataArrayFunctor(inp, dim=dim)
    np.testing.assert_allclose(functor.split(3), [[0], [1], [2]])

    _, value = next(iter(functor.iterate([0])))
    assert isinstance(value, xr.DataArray)
    assert value.dims == tuple(set(inp.dims) - {dim})
    np.testing.assert_allclose(value, expected_value)


@pytest.mark.xfail()
def test_ExtraDataFunctor():
    """Test ExtraDataFunctor."""

    # Should be tested as part of EXtra-data.
    assert False
