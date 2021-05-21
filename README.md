# pasha

pasha (**pa**rallelized **sha**red memory) provides tools to process data in parallel with an emphasis on shared memory and zero copy. It uses the map pattern similar to Python's builtin map() function, where a callable is applied to many elements in a collection. To avoid the high cost of IPC or other communication schemes, the results are meant to be written directly to memory shared between all workers as well as the calling site. The current implementations cover distribution across threads and processes on a single node.

## Quick guide

To use it, simply import it, define your kernel function of choice and map away!
```python
import numpy as np
import pasha as psh

# Get some random input data
inp = np.random.rand(100)

# Allocate output array via pasha. The returned array is
# guaranteed to be accessible from any worker, and may
# reside in shared memory.
outp = psh.alloc(like=inp)

# Define a kernel function multiplying each value with 3.
def triple_it(worker_id, index, value):
    outp[index] = 3 * value

# Map the kernel function.
psh.map(triple_it, inp)

# Check the result
np.testing.assert_allclose(outp, inp*3)
```
The runtime environment is controlled by a map context. The default context object is `ProcessContext`, which uses `multiprocessing.Pool` to distribute the work across several processes. This context only works on \*nix systems supporting the fork() system call, as it expects any input data to be shared. When the process context is selected, `psh.alloc()` creates arrays in shared memory, so workers can write output data there and the caller can retrieve it with no memory copies.

You may either create an explicit context object and use it directly or change the default context, e.g.

```python
psh.set_default_context('threads', num_workers=4)
```
There are three different context types builtin: `serial`, `threads` and `processes`.

The input array passed to map() is called a functor and is automatically wrapped in a suitable `Functor` object, here `SequenceFunctor`. This works for a number of common array and sequence types, but you may also implement your own `Functor` object to wrap anything else that can be iterated over.

For example, this is used to provide tight integration with [EXtra-data](https://github.com/European-XFEL/EXtra-data), a toolkit used to access scientific data recorded at [European XFEL](https://www.xfel.eu/). With this, you can map over `DataCollection` and `KeyData` objects to parallelize your data analysis.
```python
def analysis_kernel(worker_id, index, train_id, data):
    # Do something with the data and save it to shared memory.

run = extra_data.open_run(proposal=700000, run=1)
psh.map(analysis_kernel, run[source, key])
```