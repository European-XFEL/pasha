
from copy import copy, deepcopy
from inspect import isgeneratorfunction
from types import FunctionType

import numpy as np


class MapHook:
    """Decorator to hook into the map process.

    Applied to a map function, a map hook allows to intervene during the
    map process, e.g. by changing the map function. Any number of hooks
    may be applied to a single function, and each may run at one or more
    stages.

    There are four stages a hook can run in:

    * pre_map: Runs just before the map operation starts on the host.
    * pre_worker: Runs just before iteration starts in each worker.
    * post_worker: Runs after iteration in each worker.
    * post_map: Runs after the map operation on the host.

    A hook implementation may override __init__ to collect its arguments
    and any of the function corresponding to the stages it wants to
    intervene in.
    """

    def __call__(self, function):
        try:
            hooks = function._pasha_hooks_
        except AttributeError:
            hooks = []
            function._pasha_hooks_ = hooks

        hooks.append(self)

        return function

    def _override_function_object(self, function, code=None, globals_=None,
                                  name=None, argdefs=None, closure=None):
        """Override one or more properties of a function object."""

        new_function = FunctionType(
            code=code or function.__code__,
            globals=globals_ or function.__globals__,
            name=name or function.__name__,
            argdefs=argdefs or function.__defaults__,
            closure=closure or function.__closure__
        )

        # Restore the _pasha_hooks_ attribute, which is guaranteed to be
        # present by now.
        new_function._pasha_hooks_ = function._pasha_hooks_

        return new_function

    def _inject_function_globals(self, function, globals_):
        """Inject additional symbols into a function's global scope."""

        return self._override_function_object(
            function, globals_=dict(function.__globals__, **globals_))

    def pre_map(self, context, function):
        """Hook before the map operation starts."""

        return function

    def pre_worker(self, context, function, worker_id):
        """Hook before a worker starts iterating."""

        return function

    def post_worker(self, context, function, worker_id):
        """Hook before a worker starts iterating."""

        return function

    def post_map(self, context, function):
        """Hook after the map operation ends."""

        return function


class with_init(MapHook):
    """Hook to run an init function in each worker.

    The function is run just before iteration takes place in the worker
    and is passed the worker_id.

    If it is a generator instead, all local variables of the init
    function are inserted into the global scope of the map function.
    """

    def __init__(self, init_func):
        self.init_func = init_func

    def pre_worker(self, context, function, worker_id):
        if isgeneratorfunction(self.init_func):
            # If the init function is actually a generator, run it until
            # its first yield statement and copy its local variables
            # into the global scope of the kernel function.

            init_gen = self.init_func(worker_id)
            next(init_gen)  # Advance to the first yield.

            # Create a new function object based on the previous one,
            # adding our init function locals to the global dictionary.
            function = self._inject_function_globals(
                function, init_gen.gi_frame.f_locals.copy())
        else:
            # If the init function is just a function, simply run it.
            self.init_func(worker_id)

        return function


class with_finalize(MapHook):
    """Hook to run a finalize function in each worker.

    The function is run just after iteration takes place in the worker
    and is passed the worker_id and global scope dictionary of the map
    function.
    """

    def __init__(self, del_func):
        self.del_func = del_func

    def post_worker(self, context, function, worker_id):
        self.del_func(worker_id, function.__globals__)
        return function


class with_local_copies(MapHook):
    """Hook to create local copies of global symbols in each worker.

    Each worker creates a copy (optionally a deep copy) of the symbols
    passed to the decorator just before iteration takes place, which are
    injected into the global scope of the map function instead of the
    original object. Each worker may then use this symbol independently
    of the others, e.g. for caching.
    """

    def __init__(self, *symbols, deep=False):
        self.symbols = symbols
        self.deep = deep

    def pre_worker(self, context, function, worker_id):
        local_copies = {}

        # Search the specified symbols in the function's global scope
        # and store new copies of them.
        for symbol in self.symbols:
            try:
                value = function.__globals__[symbol]
            except KeyError:
                pass
            else:
                local_copies[symbol] = deepcopy(value) \
                    if self.deep else copy(value)

        # Return a new function object replacing the original symbols by
        # their copies.
        return self._inject_function_globals(function, local_copies)


class with_reduction(MapHook):
    """Hook to automatize parallelized reduction.

    A common pattern for data reduction is to allocate a seperate
    reduction buffer for each worker, which are then reduced after the
    map operation to yield a single result. This decorator allows to
    manage such a case automatically, by creating local copies of the
    final reduction buffer, replacing it in the map function's global
    scope and reducing these per-worker copies afterwards into the
    original buffer.

    The symbols to use for reduction must be ArrayLike.
    """

    def __init__(self, *symbols, function=np.sum, kwargs=dict(axis=0)):
        self.symbols = symbols
        self.reduce_function = function
        self.reduce_kwargs = kwargs

    def pre_map(self, context, function):
        self.per_worker_arrays = {}
        self.result_scope = function.__globals__

        # Find global symbols, allocate per-worker versions of it
        for symbol in self.symbols:
            try:
                value = function.__globals__[symbol]
                value.shape
                value.dtype
            except KeyError:
                raise ValueError(f'reduction symbol {symbol} not found in '
                                 f'function\'s global scope') from None
            except AttributeError:
                raise TypeError(f'reduction symbol {symbol} must be '
                                f'ArrayLike') from None
            else:
                per_worker_array =  context.alloc_per_worker(like=value)
                per_worker_array[:] = value
                self.per_worker_arrays[symbol] = per_worker_array

        return function

    def pre_worker(self, context, function, worker_id):
        # Pick the per-worker version and inject into function scope.
        return self._inject_function_globals(
            function, {symbol: per_worker_array[worker_id]
                       for symbol, per_worker_array
                       in self.per_worker_arrays.items()})

    def post_map(self, context, function):
        # Reduce the per-worker arrays into the global symbols.
        for symbol in self.symbols:
            self.reduce_function(
                self.per_worker_arrays[symbol], out=self.result_scope[symbol],
                **self.reduce_kwargs)

        # Clear references to allow buffers to be collected.
        self.per_worker_arrays.clear()
        self.result_scope = None

        return function
