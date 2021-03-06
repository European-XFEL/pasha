
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

__version__ = '1.0.0'

from functools import wraps

from .context import (
    MapContext, SerialContext, ThreadContext, ProcessContext)  # noqa
from .functor import (  # noqa
    SequenceFunctor, NdarrayFunctor, DataArrayFunctor, ExtraDataFunctor)


_default_context = None


def get_default_context():
    """Get default map context.

    By default, this returns a ProcessContext.

    Args:
        None

    Returns:
        (MapContext) Default map context.
    """

    global _default_context

    if _default_context is None:
        _default_context = ProcessContext()

    return _default_context


def set_default_context(ctx_or_method, *args, **kwargs):
    """Set default map context.

    Args:
        ctx_or_method (MapContext or str): New map context either
            directly or the parallelization method as a string, which
            may either be 'serial', 'threads' or 'processes'

        Any further arguments are passed to the created map context
        object if specified as a string.

    Returns:
        None
    """

    if isinstance(ctx_or_method, str):
        if ctx_or_method == 'serial':
            ctx_cls = SerialContext
        elif ctx_or_method == 'threads':
            ctx_cls = ThreadContext
        elif ctx_or_method == 'processes':
            ctx_cls = ProcessContext
        else:
            raise ValueError('invalid map method')

        ctx = ctx_cls(*args, **kwargs)
    else:
        ctx = ctx_or_method

    global _default_context
    _default_context = ctx


@wraps(MapContext.empty)
def empty(*args, **kwargs):
    return get_default_context().empty(*args, **kwargs)


@wraps(MapContext.empty_like)
def empty_like(*args, **kwargs):
    return get_default_context().empty_like(*args, **kwargs)


@wraps(MapContext.empty_per_worker)
def empty_per_worker(*args, **kwargs):
    return get_default_context().empty_per_worker(*args, **kwargs)


@wraps(MapContext.zeros)
def zeros(*args, **kwargs):
    return get_default_context().zeros(*args, **kwargs)


@wraps(MapContext.zeros_like)
def zeros_like(*args, **kwargs):
    return get_default_context().zeros_like(*args, **kwargs)


@wraps(MapContext.zeros_per_worker)
def zeros_per_worker(*args, **kwargs):
    return get_default_context().zeros_per_worker(*args, **kwargs)


@wraps(MapContext.ones)
def ones(*args, **kwargs):
    return get_default_context().ones(*args, **kwargs)


@wraps(MapContext.ones_like)
def ones_like(*args, **kwargs):
    return get_default_context().ones_like(*args, **kwargs)


@wraps(MapContext.ones_per_worker)
def ones_per_worker(*args, **kwargs):
    return get_default_context().ones_per_worker(*args, **kwargs)


@wraps(MapContext.full)
def full(*args, **kwargs):
    return get_default_context().full(*args, **kwargs)


@wraps(MapContext.full_like)
def full_like(*args, **kwargs):
    return get_default_context().full_like(*args, **kwargs)


@wraps(MapContext.full_per_worker)
def full_per_worker(*args, **kwargs):
    return get_default_context().full_per_worker(*args, **kwargs)


@wraps(MapContext.map)
def map(*args, **kwargs):
    return get_default_context().map(*args, **kwargs)
