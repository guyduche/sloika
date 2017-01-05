import cPickle
from multiprocessing import Process, SimpleQueue
import sys
import tempfile


def _compile_model(outqueue, model_file):
    """  Compile network if necessary and place filename

    Where the network is already compiled, a temporary copy
    is created.

    :param outqueue: Queue to output filename
    :param model_file: File to read network from

    """
    from sloika import layers
    import theano

    sys.setrecursionlimit(10000)
    with open(model_file, 'r') as fh:
        network = cPickle.load(fh)
    if isinstance(network, layers.Layer):
        #  File contains network to compile
        with tempfile.NamedTemporaryFile(mode='wb', dir='', suffix='.pkl', delete=False) as fh:
            compiled_file = fh.name
            compiled_network = network.compile()
            cPickle.dump(compiled_network, fh, protocol=cPickle.HIGHEST_PROTOCOL)
    elif isinstance(network, theano.compile.function_module.Function):
        #  Network is already compiled - make temporary copy
        with tempfile.NamedTemporaryFile(mode='wb', dir='', suffix='.pkl', delete=False) as fh:
            compiled_file = fh.name
        shutil.copy(model_file, compiled_file)
    else:
        sys.exit(1)

    outqueue.put(compiled_file)


def compile_model(model_file):
    """  Compile network in separate thread

    To avoid initialising Theano in main thread, compilation must be done in a
    separate process.  Where the network is already compiled, a temporary copy
    is created.

    :param model_file: File to read network from

    :returns: A filename containing a compiled network.
    """
    queue = SimpleQueue()
    p = Process(target=_compile_model, args=(queue, model_file))
    p.start()
    p.join()
    if p.exitcode < 0:
        raise ValueError("model_file was neither a network nor compiled network")
    compiled_file = queue.get()

    return compiled_file
