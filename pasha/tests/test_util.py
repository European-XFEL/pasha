
import numpy as np
import pasha as psh
from pasha.util import MapHook


class HookedMapHook(MapHook):
    """A hook for a hook. A Hookhook. A Hook²."""

    def __init__(self, context, function):
        self.context = context
        self.function = function

        self.pre_map_called = False
        self.pre_worker_called = [False] * context.num_workers
        self.kernel_called = False
        self.post_worker_called = [False] * context.num_workers
        self.post_map_called = False

    def assert_called(self, pre_map, pre_worker, kernel, post_worker,
                      post_map):
        assert self.pre_map_called == pre_map
        assert all([x == pre_worker for x in self.pre_worker_called])
        assert self.kernel_called == kernel
        assert all([x == post_worker for x in self.post_worker_called])
        assert self.post_map_called == post_map

    def pre_map(self, context, function):
        assert context is self.context
        assert function is self.function

        self.assert_called(False, False, False, False, False)
        self.pre_map_called = True

        return function

    def pre_worker(self, context, function, worker_id):
        assert context is self.context
        assert function is self.function

        self.assert_called(True, False, False, False, False)
        self.pre_worker_called[worker_id] = True

        return function

    def post_worker(self, context, function, worker_id):
        assert context is self.context
        assert function is self.function

        self.assert_called(True, True, True, False, False)
        self.post_worker_called[worker_id] = True

        return function

    def post_map(self, context, function):
        assert context is self.context
        assert function is self.function

        self.assert_called(True, True, True, True, False)
        self.post_map_called = True

        return function


def test_maphook():
    def kernel(wid, index, data):
        if index == 0:
            hook.assert_called(True, True, False, False, False)
            hook.kernel_called = True

    ctx = psh.SerialContext()

    hook = HookedMapHook(ctx, kernel)
    hook(kernel)

    inp = np.arange(10)
    ctx.map(kernel, inp)

    hook.assert_called(True, True, True, True, True)
