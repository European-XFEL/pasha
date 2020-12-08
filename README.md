# EXtra-pasha

pasha (**pa**rallelized **sha**red memory) provides tools to process data in a parallelized way with an emphasis on shared memory and zero copy. It uses the map pattern similar to Python's builtin map() function, where a callable is applied to potentially many elements in a collection. To avoid the high cost of IPC or other communication schemes, the results are meant to be written directly to memory shared between all workers as well as the calling site. The current implementations cover distribution across threads and processes on a single node.

## Quick guide

To use it, simply import it, define your kernel function of choice and map away!
```python
import numpy as np
import pasha as psh

# Get some random input data
inp = np.random.rand(100)

# Allocate output array via EXtra-pasha.
outp = psh.array(100)

# Define a kernel function multiplying each value with 3.
def triple_it(worker_id, index, value):
    outp[index] = 3 * value

# Map the kernel function.
psh.map(triple_it, inp)

# Check the result
np.testing.assert_allclose(outp, inp*3)
```
The runtime environment is controlled via a so called map context. The default
context object is `ProcessContext`, which uses `multiprocessing.Pool` to distribute the work across several processes. The output array returned by array() resides in shared memory with this context in order to modify it by the worker processes without the need to copy anything around. This context only works on \*nix systems supporting the fork() system call, as it expects any input data to be shared.

You may either create an explicit context object and use it directly or change the default context, e.g.

```python
psh.set_default_context('threads', num_workers=4)
```
There are three different context types builtin: `serial`, `threads` and `processes`.

The input array passed to map() is called a functor and automatically wrapped in a suitable `Functor` object, here `SequenceFunctor`. This works for a number of common array and collection types, but you may also implement your own `Functor` object to wrap anything else. For example, there is built-in support for `DataCollection` and `KeyData` objects from the EXtra-data toolkit accessing run files from the European XFEL facilty:
```python
def analysis_kernel(worker_id, index, train_id, data):
    # Do something with the data and save it to shared memory.

run = extra_data.open_run(proposal=700000, run=1)
psh.map(analysis_kernel, run[source, key])
```