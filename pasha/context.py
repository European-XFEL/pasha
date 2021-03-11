
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

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
    def _deprecated_array_function_warning():
        from warnings import warn, filterwarnings
        filterwarnings('always')
        warn('The array[...]() family of functions has been replaced by '
             'zeros[...]() and will be removed in a future version',
             DeprecationWarning, stacklevel=3)

    def _per_worker_shape(self, extra_shape):
        """Preprend worker axis to an array shape."""

        if isinstance(extra_shape, int):
            return (self.num_workers, extra_shape)
        else:
            return (self.num_workers,) + tuple(extra_shape)

    @staticmethod
    def _like_kwargs(prototype, dtype, order, shape):
        if dtype is None:
            dtype = prototype.dtype

        order = order.upper()

        if order in ('K', 'A'):
            # Defining the branch in this order allows both cases to be
            # handled.
            order = 'F' if prototype.flags.f_contiguous else 'C'
        elif order not in ('C', 'F'):
            raise ValueError('invalid memory order')

        if shape is None:
            shape = prototype.shape

        return dict(dtype=dtype, order=order, shape=shape)

    def empty(self, shape, dtype=np.float64, order='C'):
        """Allocate an array shared with all workers.

        The implementation may decide how to back this memory, but it
        is required that all workers of this context may read AND write
        to this memory. The array values are not guaranteed to be
        initialized to any particular value.

        Args:
            shape (int or tuple of ints): Shape of the array.
            dtype (DTypeLike, optional): Data type of the array.
            order ('C' or 'F', optional): Whether store multiple
                dimensions in row-major (C-style, default) or
                column-major (Fortran-style) order.

        Returns:
            (numpy.ndarray) Created array object.
        """

        raise NotImplementedError('Context.empty')

    def empty_like(self, prototype, dtype=None, order='K', shape=None):
        """Allocate a shared array with the same shape/dtype as another.

        Args:
            prototype (ArrayLike): Array to copy properties from.
            dtype (DTypeLike, optional): Override data type of the
                resulting array.
            order ('C', 'F', 'A' or 'K', optional): Override memory
                order of the resulting array ('C' or 'F'), keep the
                same layout ('K') or use Fortran-style if and only if
                prototype is Fortran-style ('A').
            shape (int or tuple of ints): Override shape of the
                resulting array.

        Returns:
            (numpy.ndarray) Created array object.
        """

        return self.empty(**self._like_kwargs(prototype, dtype, order, shape))

    def empty_per_worker(self, shape, dtype=np.float64, order='C'):
        """Allocate a shared array for each worker.

        The returned array will contain an additional prepended axis
        with its shape corresponding to the number of workers in this
        context, i.e. with one dimension more than specified by the
        shape parameter. These are useful for parallelized reduction
        operations, where each worker may work with its own accumulator.

        Args:
            shape (int or tuple of ints): Shape of the array.
            dtype (DTypeLike, optional): Data type of the array.
            order ('C' or 'F', optional): Whether store multiple
                dimensions in row-major (C-style, default) or
                column-major (Fortran-style) order.

        Returns:
            (numpy.ndarray) Created array object.
        """

        return self.empty(self._per_worker_shape(shape), dtype=dtype,
                          order=order)

    def zeros(self, shape, dtype=np.float64, order='C'):
        """Allocate an initialized array.

        Equivalent to :meth:`Context.empty`, but all elements are
        initialized to zero.
        """

        array = self.empty(shape, dtype=dtype, order=order)
        array[:] = 0

        return array

    def zeros_like(self, prototype, dtype=None, order='K', shape=None):
        """Allocate an initialized array with the same shape/dtype.

        Equivalent to :meth:`Context.empty_like`, but all elements are
        initialized to zero.
        """

        return self.zeros(**self._like_kwargs(prototype, dtype, order, shape))

    def zeros_per_worker(self, shape, dtype=np.float64, order='C'):
        """Allocate an initialized array for each worker.

        Equivalent to :meth:`Context.empty_per_worker`, but all elements
        are initialized to zero.
        """

        return self.zeros(self._per_worker_shape(shape), dtype=dtype,
                          order=order)

    def ones(self, shape, dtype=np.float64, order='C'):
        """Allocate an initialized array shared with all workers.

        Equivalent to :meth:`Context.empty`, but all elements are
        initialized to one.
        """

        array = self.empty(shape, dtype=dtype, order=order)
        array[:] = 1

        return array

    def ones_like(self, prototype, dtype=None, order='K', shape=None):
        """Allocate an initialized array with the same shape/dtype.

        Equivalent to :meth:`Context.empty_like`, but all elements are
        initialized to one.
        """

        return self.ones(**self._like_kwargs(prototype, dtype, order, shape))

    def ones_per_worker(self, shape, dtype=np.float64, order='C'):
        """Allocate an initialized array for each worker.

        Equivalent to :meth:`Context.empty_per_worker`, but all elements
        are initialized to one.
        """

        return self.ones(self._per_worker_shape(shape), dtype=dtype,
                         order=order)

    def full(self, shape, fill_value, dtype=np.float64, order='C'):
        """Allocate an initialized array shared with all workers.

        Equivalent to :meth:`Context.empty`, but all elements are
        initialized to the specified fill value.

        Args:
            shape (int or tuple of ints): Shape of the array.
            fill_value (scalar or ArrayLike): Value to initialize to.
            dtype (DTypeLike, optional): Data type of the array.
            order ('C' or 'F', optional): Whether store multiple
                dimensions in row-major (C-style, default) or
                column-major (Fortran-style) order.

        Returns:
            (numpy.ndarray) Created array object.
        """

        array = self.empty(shape, dtype=dtype, order=order)
        array[:] = fill_value

        return array

    def full_like(self, prototype, fill_value, dtype=None, order='K',
                  shape=None):
        """Allocate an initialized array with the same shape/dtype.

        Equivalent to :meth:`Context.empty_like`, but all elements are
        initialized to the specified fill value.

        Args:
            prototype (ArrayLike): Array to copy properties from.
            fill_value (scalar or ArrayLike): Value to initialize to.
            dtype (DTypeLike, optional): Override data type of the
                resulting array.
            order ('C', 'F', 'A' or 'K', optional): Override memory
                order of the resulting array ('C' or 'F'), keep the
                same layout ('K') or use Fortran-style if and only if
                prototype is Fortran-style ('A').
            shape (int or tuple of ints): Override shape of the
                resulting array.

        Returns:
            (numpy.ndarray) Created array object.
        """

        return self.full(**self._like_kwargs(prototype, dtype, order, shape),
                         fill_value=fill_value)

    def full_per_worker(self, shape, fill_value, dtype=np.float64, order='C'):
        """Allocate an initialized array for each worker.

        Equivalent to :meth:`Context.empty_per_worker`, but all elements
        are initialized to the specified fill value.
        """

        return self.full(self._per_worker_shape(shape), fill_value,
                         dtype=dtype, order=order)

    def array(self, shape, dtype=np.float64):
        self._deprecated_array_function_warning()
        return self.zeros(shape, dtype=dtype)

    def array_like(self, other):
        self._deprecated_array_function_warning()
        return self.zeros_like(other)

    def array_per_worker(self, shape, dtype=np.float64):
        self._deprecated_array_function_warning()
        return self.zeros_per_worker(shape, dtype=dtype)

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


class HeapContext(MapContext):
    """Abstract map context allocating arrays on the heap."""

    def empty(self, shape, dtype=np.float64, order='C'):
        return np.empty(shape, dtype=dtype, order=order)

    def zeros(self, shape, dtype=np.float64, order='C'):
        return np.zeros(shape, dtype=dtype, order=order)

    def ones(self, shape, dtype=np.float64, order='C'):
        return np.ones(shape, dtype=dtype, order=order)

    def full(self, shape, fill_value, dtype=np.float64, order='C'):
        return np.full(shape, fill_value, dtype=dtype, order=order)


class SerialContext(HeapContext):
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


class ThreadContext(PoolContext, HeapContext):
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
    method to be `fork`, which is only supported on *nix systems.
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

    def empty(self, shape, dtype=np.float64, order='C'):
        if isinstance(shape, int):
            n_elements = shape
        else:
            n_elements = 1
            for _s in shape:
                n_elements *= _s

        import mmap
        n_bytes = n_elements * np.dtype(dtype).itemsize
        n_pages = n_bytes // mmap.PAGESIZE + 1

        buf = mmap.mmap(-1, n_pages * mmap.PAGESIZE,
                        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)

        return np.frombuffer(memoryview(buf)[:n_bytes],
                             dtype=dtype).reshape(shape, order=order)

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
