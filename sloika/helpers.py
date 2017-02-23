from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import *
import pickle
from multiprocessing import Process
try:
    # Python 3.3 or later
    from multiprocessing import SimpleQueue
except ImportError:
    from multiprocessing.queues import SimpleQueue
import shutil
import sys
import tempfile


def _compile_model(outqueue, model_file, output_file=None):
    """  Compile network if necessary

    Where the network is already compiled, a temporary copy
    is created.

    :param outqueue: Queue to output filename
    :param model_file: File to read network from
    :param output_file: File to output to.  If None, generate a filename

    :returns: places name of a file containined compiled model into queue
    """
    from sloika import layers
    import theano

    if output_file is None:
        with tempfile.NamedTemporaryFile(mode='wb', dir='', suffix='.pkl', delete=False) as fh:
            output_file = fh.name

    sys.setrecursionlimit(10000)
    with open(model_file, 'rb') as fh:
        if sys.version_info[0] == 3:
            network = pickle.load(fh, encoding='latin1')
        else:
            network = pickle.load(fh)
    if isinstance(network, layers.Layer):
        #  File contains network to compile
        with open(output_file, 'wb') as fh:
            compiled_network = network.compile()
            pickle.dump(compiled_network, fh, protocol=pickle.HIGHEST_PROTOCOL)
    elif isinstance(network, theano.compile.function_module.Function):
        #  Network is already compiled - make temporary copy
        shutil.copy(model_file, output_file)
    else:
        sys.exit(1)

    outqueue.put(output_file)


def compile_model(model_file, output_file=None):
    """  Compile network in separate thread

    To avoid initialising Theano in main thread, compilation must be done in a
    separate process.  Where the network is already compiled, a temporary copy
    is created.

    :param model_file: File to read network from
    :param output_file: File to output to.  If None, generate a filename

    :returns: A filename containing a compiled network.
    """
    queue = SimpleQueue()
    p = Process(target=_compile_model, args=(queue, model_file, output_file))
    p.start()
    p.join()
    if p.exitcode != 0:
        output_file = None
        raise ValueError("model_file was neither a network nor compiled network")
    else:
        output_file = queue.get()

    return output_file
