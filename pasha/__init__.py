
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

__version__ = '1.0.0'


from .context import SerialContext, ThreadContext, ProcessContext  # noqa
from .functor import (  # noqa
    SequenceFunctor, NdarrayFunctor, DataArrayFunctor, ExtraDataFunctor)


_default_context = ProcessContext()


def get_default_context():
    """Get default map context.

    On startup, a LocalContext is used as the default context.

    Args:
        None

    Returns:
        (MapContext) Default map context.
    """

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


def array(*args, **kwargs):
    """Allocate an array shared with all workers.

    See MapContext.array(). This module-level function forwards the call
    to the default context.
    """

    return _default_context.array(*args, **kwargs)


def array_like(*args, **kwargs):
    """Allocate an array with the same shape and dtype as another.

    See MapContext.array_like(). This module-level function forwards the
    call to the default context.
    """

    return _default_context.array_like(*args, **kwargs)


def array_per_worker(*args, **kwargs):
    """Allocate a shared array for each worker.

    See MapContext.array_per_worker(). This module-level function
    forwards the call to the default context.
    """

    return _default_context.array_per_worker(*args, **kwargs)


def map(*args, **kwargs):
    """Apply kernel to each element in the target.

    See MapContext.map(). This module-level function forwards the call
    to the default context.
    """

    return _default_context.map(*args, **kwargs)
