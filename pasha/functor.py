
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from collections.abc import Sequence
from importlib import import_module

import numpy as np


def import_if_type_likely(module_name, obj, type_names=None):
    """Import a module if an object's type matches by name.

    Args:
        module_name (str): Module to import.
        obj (Any): Object to check type of.
        type_names (str or Collection of str, optional): Full qualified
            type name(s) to check against, only module_name is tested
            by default.

    Returns:
        (module or None) Imported module object or None if type does not
            seem to match or an ImportError occured.
    """

    if type_names is not None:
        full_name = f'{type(obj).__module__}.{type(obj).__qualname__}'

        if isinstance(type_names, str):
            type_names = [type_names]

        if full_name not in type_names:
            return
    elif not type(obj).__module__.startswith(module_name):
        return

    try:
        module_obj = import_module(module_name)
    except ImportError:
        return

    return module_obj


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
            share (Any): Element of the Iterable returned by to iterate
                over.

        Returns:
            None
        """

        raise NotImplementedError('iterate')


class SequenceFunctor(Functor):
    """Functor wrapping a sequence.

    This functor can wrap any indexable collection, e.g. list, tuples,
    or any other type implementing __getitem__. It automatically wraps
    any value implementing the collections.abc.Sequence type. The kernel
    is passed the current index and sequence value at that index.
    """

    def __init__(self, sequence):
        """Initialize a sequence functor.

        Args:
            sequence (Sequence): Sequence to process.
        """

        self.sequence = sequence

    @classmethod
    def wrap(cls, value):
        if isinstance(value, Sequence):
            return cls(value)

    def split(self, num_workers):
        return np.array_split(np.arange(len(self.sequence)), num_workers)

    def iterate(self, indices):
        for index in indices:
            yield index, self.sequence[index]


class NdarrayFunctor(SequenceFunctor):
    """Functor wrapping an numpy.ndarray.

    This functor extends SequenceFunctor to use additional functionality
    provided by numpy ndarrays, e.g. iterating over specific axes and
    more efficient indexing and should works for any array_like object
    that supports numpy-style slicing. However, specifying an explicit
    axis may cause conversion to an ndarray or break unexpectedly.
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

    def iterate(self, indices):
        yield from zip(indices, self.sequence[indices])


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
        xr = import_if_type_likely('xarray', value,
                                   'xarray.core.dataarray.DataArray')

        if xr is not None and isinstance(value, xr.DataArray):
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

        import extra_data as xd
        ExtraDataFunctor.xd = xd

    @classmethod
    def wrap(cls, value):
        xd = import_if_type_likely(
            'extra_data', value,
            ['extra_data.reader.DataCollection', 'extra_data.keydata.KeyData'])

        if xd is not None and \
                isinstance(value, (xd.DataCollection, xd.keydata.KeyData)):
            return cls(value)

    def split(self, num_workers):
        return np.array_split(np.arange(len(self.obj.train_ids)), num_workers)

    def iterate(self, indices):
        subobj = self.obj.select_trains(ExtraDataFunctor.xd.by_index[indices])

        # Close all file handles inherited from the parent collection
        # to force re-opening them in each worker process.
        for f in subobj.files:
            f.close()

        for index, (train_id, data) in zip(indices, subobj.trains()):
            yield index, train_id, data
