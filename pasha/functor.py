
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

import math
import sys
from collections.abc import Sequence

import numpy as np


def gen_split_slices(total_len, part_len=None, n_parts=None):
    """Generate slices to split a sequence.

    Returns a list of slices to split a sequence into smaller pieces.

    Args:
        total_len (int): Length of the full sequence.
        part_len (int, optional): Length of each piece, may be None if
            n_parts is specified.
        n_parts (int, optional): Number of pieces, ignored if part_len
            is specified.

    Returns:
        (list of slice) Slices to split the sequence.
    """

    if part_len is None:
        if n_parts is None:
            raise ValueError('must specify either part_len or n_parts')

    else:
        n_parts = math.ceil(total_len / part_len)

    return [slice(i * total_len // n_parts, (i+1) * total_len // n_parts)
            for i in range(n_parts)]


class Functor:
    """Mappable object.

    In functional programming, a functor is something that can be mapped
    over. This interface specifically provides the machinery to
    distribute a share of the value to each worker. The simplest functor
    is SequenceFunctor, which assigns a slice to each worker and then
    iterates over the indices of that slice in its body.

    If a custom type extends this class and implements the wrap
    classmethod, then it can take advantage of automatic wrapping of
    values passed to a map call.

    Alternatively, a type may implement a '_pasha_functor_ method to
    return a suitable functor. This will always take precedence over
    the automatic detection.
    """

    _functor_types = []

    @classmethod
    def __init_subclass__(cls):
        cls._functor_types.append(cls)

    @classmethod
    def try_wrap(cls, value):
        """Attempt to wrap a value in a functor.

        Args:
            value (Any): Value to wrap for map operation.

        Returns:
            (Functor) Functor wrapping the given value or the same value
                if already a subtype of Functor.

        Raises:
            ValueError: If no or more than one default functor types
                could be applied.
        """

        if isinstance(value, cls):
            return value

        if hasattr(value, '_pasha_functor_'):
            return value._pasha_functor_()

        functor = None

        for functor_type in cls._functor_types:
            cur_functor = functor_type.wrap(value)

            if cur_functor is not None:
                if functor is not None:
                    raise ValueError('default functor is ambiguous')

                functor = cur_functor

        if functor is None:
            raise ValueError(f'no default functor for {type(value)}')

        return functor

    @classmethod
    def wrap(self, value):
        """Wrap value in this functor type, if possible.

        Args:
            value (Any): Value to wrap for map operation.

        Returns:
            (Functor) Functor if wrapping is possible or None.
        """

        return

    def split(self, num_workers):
        """Split this functor into work units.

        The values contained in the returned Iterable are passed to this
        functor's iterate method later on. It may consist of any value
        suitable to describe each work unit, e.g. an iterable of indices
        of a sequence.

        Args:
            num_workers (int): Number of workers processing this
                functor.

        Returns:
            (Iterable) Iterable of elements for each work unit.
        """

        raise NotImplementedError('split')

    def iterate(self, share):
        """Iterate over a share of this functor.

        Args:
            share (Any): Element of the Iterable returned by split() to
                iterate over. The exact type of specification is up to
                the `Functor` implementation.

        Returns:
            None
        """

        raise NotImplementedError('iterate')


class SequenceFunctor(Functor):
    """Functor wrapping a generic sequence.

    This functor can wrap any type implementing the
    collections.abc.Sequence type, i.e. is indexable by integers and
    slices and has a length. Instances of this abstract type are
    automatically wrapped, but the functor should work with any
    ArrayLike object. The kernel is passed the current index and
    sequence value at that index.
    """

    def __init__(self, sequence):
        """Initialize a sequence functor.

        Args:
            sequence (Sequence, ArrayLike): Sequence to process.
        """

        self.sequence = sequence

    @classmethod
    def wrap(cls, value):
        if isinstance(value, Sequence):
            return cls(value)

    def split(self, num_workers):
        return gen_split_slices(len(self.sequence), n_parts=num_workers)

    def iterate(self, share):
        yield from zip(range(*share.indices(len(self.sequence))),
                       self.sequence[share])


class NdarrayFunctor(SequenceFunctor):
    """Functor wrapping a numpy.ndarray.

    This functor extends SequenceFunctor to allow iterating over
    specific axes. While it should work for any ArrayLike object,
    specifying an explicit axis may cause conversion to an ndarray.
    """

    def __init__(self, array, axis=None):
        """Initialize an ndarray functor.

        Args:
            array (numpy.ndarray): Array to map over.
            axis (int, optional): Axis to map over, first axis by
                default or if None.
        """

        self.sequence = np.swapaxes(array, 0, axis) \
            if axis is not None else array

    @classmethod
    def wrap(cls, value):
        if isinstance(value, np.ndarray):
            return cls(value)


class DataArrayFunctor(NdarrayFunctor):
    """Functor wrapping an xarray.DataArray.

    This functor extends NdarrayFunctor for compatbility with xarray's
    DataArray type, in particular dimension labels and coordinates.
    """

    def __init__(self, array, dim=None):
        """Initialize a DataArray functor.

        Args:
            array (xarray.DataArray): Labeled array to map over.
            dim (str, optional): Dimension to map over, first dimension
                by default or if None.
        """

        self.sequence = array.transpose(dim, ...) \
            if dim is not None else array

    @classmethod
    def wrap(cls, value):
        if 'xarray' not in sys.modules:
            # If xarray has not been loaded yet, we can safely assume
            # this to not be a DataArray.
            return

        import xarray as xr

        if isinstance(value, xr.DataArray):
            return cls(value)


# Ideas for wrapping functors: xarray.Dataset, pandas


class ExtraDataFunctor(Functor):
    """Functor for EXtra-data objects.

    This functor wraps an EXtra-data DataCollection or KeyData and
    performs the map operation over its trains. The kernel is passed the
    current train's index in the collection, the train ID and the data
    mapping (for DataCollection) or data entry (for KeyData).
    """

    def __init__(self, obj):
        self.obj = obj
        self.n_trains = len(self.obj.train_ids)

    @classmethod
    def wrap(cls, value):
        if 'extra_data' not in sys.modules:
            # Same assumption as in DataArrayFunctor.
            return

        import extra_data as xd

        if isinstance(value, (xd.DataCollection, xd.keydata.KeyData)):
            return cls(value)

    def split(self, num_workers):
        return gen_split_slices(self.n_trains, n_parts=num_workers)

    def iterate(self, share):
        subobj = self.obj.select_trains(np.s_[share])

        # Close all file handles inherited from the parent collection
        # to force re-opening them in each worker process.
        for f in subobj.files:
            f.close()

        it = zip(range(*share.indices(self.n_trains)), subobj.trains())

        for index, (train_id, data) in it:
            yield index, train_id, data
