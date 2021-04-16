
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from numbers import Integral

import numpy as np

from .functor import Functor


class MapContext:
    """Context to execute map operations.

    A map operation applies a single callable to each item in a
    collection. The context define the runtime conditions for this
    operation, which may be in a process pool, for example.

    As some of these environments may require special memory semantics,
    the context also provides a method of allocating ndarrays. The only
    abstract method required by an implementation is map. It is
    recommended to use the staticmethod run_worker in this type for the
    actual mapping loop in each worker.
    """

    def __init__(self, num_workers):
        """Initialize this map context.

        In order to use the default allocation methods in this type,
        in particular array_per_worker(), the number of workers in this
        context need to be passed to its initializer or stored in a
        `num_workers` property.

        Args:
            num_workers (int): Number of workers used in this context.
        """

        self.num_workers = num_workers

    @staticmethod
    def _resolve_alloc(shape, dtype, order, like):
        """Resolve allocation arguments.

        This method resolves the arguments passed to
        :method:`MapContext.alloc` and fills any omitted values.

        Returns:
            (tuple, DtypeLike, str): Resolved shape, data-type and
                memory order.
        """
        if like is not None:
            shape = shape if shape is not None else like.shape
            dtype = dtype if dtype is not None else like.dtype

            try:
                if order is None:
                    # Check for C contiguity first, as one-dimensional
                    # arrays may be both C and F-style at the same time.
                    # If neither is the case, make the result C-style
                    # anyway.
                    if like.flags.c_contiguous:
                        order = 'C'
                    elif like.flags.f_contiguous:
                        order = 'F'
                    else:
                        order = 'C'
            except AttributeError:
                # Existance of the flags property may not be considered
                # required for an ArrayLike, so in case it's not just
                # go for C-style.
                order = 'C'

        elif shape is None:
            raise ValueError('array shape must be specified')

        else:
            dtype = dtype if dtype is not None else np.float64
            order = order if order is not None else 'C'

        if isinstance(shape, Integral):
            shape = (shape,)
        elif not isinstance(shape, tuple):
            shape = tuple(shape)

        return shape, dtype, order

    @staticmethod
    def _alloc(shape, dtype, order, fill):
        """Low-level allocation.

        In most instances, it is recommended to use the high-level
        method :method:`MapContext.alloc` instead. Context
        implemenations can override this method to provide alternative
        means of providing memory shared with all workers, e.g. via
        shared memory segments across processes.

        Note that none of the arguments of this method are optional.

        The default implementation allocates directly on the heap.

        Args:
            shape (tuple): Shape of the array.
            dtype (DtypeLike): Desired data-type for the array.
            order ('C' or 'F') C-style or Fortran-style memory order.
            fill (Scalar, ArrayLike or None): Initialization value or
                None if the array may be unitialized.

        Returns:
            (numpy.ndarray) Created array object.
        """

        if fill is None:
            return np.empty(shape, dtype, order)
        elif fill == 0:
            return np.zeros(shape, dtype, order)
        else:
            return np.full(shape, fill, dtype, order)

    def alloc(self, shape=None, dtype=None, *,
              order=None, fill=None, like=None, per_worker=False):
        """Allocate an array guaranteed to be shared with all workers.

        The implementation may decide how to back this memory, but it
        is required that all workers of this context may read AND write
        to this memory. By default, it allocates directly on the heap.

        Only the shape argument is required for allocation to succeed,
        however it may be omitted if an existing ArrayLike object is
        passed via the like argument whose shape is taken.

        The dtype and order arguments have the default values np.float64
        and 'C' if omitted or are taken from the ArrayLike object passed
        to like. Only C-style or Fortran-style order may be assumed to
        be supported by all context implementations.

        If fill is omitted, the resulting array is uninitialized.

        The per_worker argument automatically adds an axis to the
        resulting array corresponding to the number of workers in this
        context, i.e. on top of the shape specified via other means. In
        row-major order (C-style) the axis is prepended, in column-major
        order (Fortran-style) it is appended. This is useful for
        parallelized reduction operations, where each worker may work
        with its own accumulator, which are all reduced after mapping.

        Any specified argument always takes precedence over values
        inferred from an ArrayLike object passed to like.

        Args:
            shape (int or tuple of ints, optional): Shape of the array.
            dtype (DTypeLike, optional): Desired data-type for the array.
            order ('C' or 'F', optional): Whether to store multiple
                dimensions in row-major (C-style, default) or
                column-major (Fortran-style).
            fill (Scalar or ArrayLike, optional): Initialization value.
            like (ArrayLike): Array to cope shape, dtype and order from.
            per_worker (bool, optional): Whether to add an additional
                axis corresponding to the number of workers.

        Returns:
            (numpy.ndarray) Created array object.
        """

        shape, dtype, order = self._resolve_alloc(shape, dtype, order, like)

        if per_worker:
            if order == 'C':
                shape = (self.num_workers,) + shape
            elif order == 'F':
                shape = shape + (self.num_workers,)

        array = self._alloc(shape, dtype, order, fill)

        return array

    def map(self, function, functor):
        """Apply a function to a functor.

        This method performs the map operation, applying the function to
        each element of the functor. The functor may either be an
        explicit Functor object or any other supported type, which can
        be wrapped into a default functor.

        Args:
            function (Callable): Kernel function to map with.
            functor (Functor or Any): Functor to map over or any type
                with automatic wrapping support.

        Returns:
            None
        """

        raise NotImplementedError('map')

    @staticmethod
    def run_worker(function, functor, share, worker_id):
        """Main worker loop.

        This staticmethod contains the actual inner loop for a worker,
        i.e. iterating over the functor and calling the kernel function.

        Subtypes may call this method after sorting out the required
        parameters through their specific machinery.

        Args:
            function (Callable): Kernel function to map with.
            functor (Functor): Functor to map over.
            share (Any): Functor's share assigned to this worker.
            worker_id (int): Identification of this worker. All passed
                values must be between 0 and num_workers-1.

        Returns:
            None
        """

        for entry in functor.iterate(share):
            function(worker_id, *entry)


class SerialContext(MapContext):
    """Serial map context.

    Runs the map operation directly in the same process and thread
    without any actual parallelism.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(num_workers=1)

    def map(self, function, functor):
        functor = Functor.try_wrap(functor)
        self.run_worker(function, functor, next(iter(functor.split(1))), 0)


class PoolContext(MapContext):
    """Abstract map context for multiprocessing.Pool interface.

    This class contains the common machinery required for a map context
    based on the Pool interface. A subtype is still required to
    implemenent the map() method with its actual call signature and then
    call this type's map method in turn.
    """

    def __init__(self, num_workers=None):
        from os import cpu_count

        if num_workers is None:
            num_workers = min(cpu_count() // 2, 10)

        super().__init__(num_workers=num_workers)

    def map(self, function, functor, pool_cls):
        """Apply a function to a functor.

        Incomplete map method to be called by a subtype.

        Args:
            function (Callable): Kernel function to map with.
            functor (Functor or Any): Functor to map over or any type
                with automatic wrapping support.
            pool_cls (type): Pool implementation to use.

        Returns:
            None
        """

        self.function = function
        functor = Functor.try_wrap(functor)

        for worker_id in range(self.num_workers):
            self.id_queue.put(worker_id)

        with pool_cls(self.num_workers, self.init_worker, (functor,)) as p:
            p.map(self.run_worker, functor.split(self.num_workers))


class ThreadContext(PoolContext):
    """Map context using a thread pool.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from queue import Queue
        from threading import local

        self.id_queue = Queue()
        self.worker_storage = local()

    def map(self, function, functor):
        from multiprocessing.pool import ThreadPool
        super().map(function, functor, ThreadPool)

    def init_worker(self, functor):
        self.worker_storage.worker_id = self.id_queue.get()
        self.worker_storage.functor = functor

    def run_worker(self, share):
        super().run_worker(self.function, self.worker_storage.functor, share,
                           self.worker_storage.worker_id)


class ProcessContext(PoolContext):
    """Map context using a process pool.

    The memory allocated by this context is backed by anonymous mappings
    via `mmap` and thus shared for both reads and writes with all worker
    processes created after the allocation. This requires the start
    method to be `fork`, which is only supported on Unix systems.
    """

    _instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from multiprocessing import get_context

        try:
            self.mp_ctx = get_context('fork')
        except ValueError:
            raise ValueError('fork context required')

        self.id_queue = self.mp_ctx.Queue()

    def _alloc(self, shape, dtype, order, fill):
        # Allocate shared memory via mmap.

        n_elements = 1
        for _s in shape:
            n_elements *= _s

        import mmap
        n_bytes = n_elements * np.dtype(dtype).itemsize
        n_pages = n_bytes // mmap.PAGESIZE + 1

        buf = mmap.mmap(-1, n_pages * mmap.PAGESIZE,
                        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)

        array = np.frombuffer(memoryview(buf)[:n_bytes], dtype=dtype) \
            .reshape(shape, order=order)

        if fill is not None and fill != 0:
            # POSIX guarantess allocation with MAP_ANONYMOUS to be
            # initialized with zeroes.
            array[:] = fill

        return array

    def map(self, function, functor):
        super().map(function, functor, self.mp_ctx.Pool)

    def init_worker(self, functor):
        # Save reference in process-local copy
        self.__class__._instance = self

        self.worker_id = self.id_queue.get()
        self.functor = functor

    @classmethod
    def run_worker(cls, share):
        # map is a classmethod here and fetches its process-local
        # instance, as the instance in the parent process is not
        # actually part of the execution.

        self = cls._instance
        super(cls, self).run_worker(self.function, self.functor, share,
                                    self.worker_id)
