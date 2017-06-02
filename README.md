# Installation of system prerequisites

    sudo make deps

# Setting up clean development environment

    make cleanDevEnv

# Running unit tests in development mode

    make

For this step to function development environment needs to be set up, and make deps must have been installed.

**Note to experts**: These tests will run with the theano flags defined in the `environment` file. If you need to test sloika using a different set of theano flags you can edit this file before running make. Please do report any problems that you run into, although we cannot promise we can help with your configuration.

# Note on `THEANO_FLAGS`
To use Theano effectively, A typical set of Theano flags might look like:
```bash
export THEANO_FLAGS=openmp=True,floatX=float32,warn_float64=warn,optimizer=fast_run,device=gpu0,scan.allow_gc=False,lib.cnmem=0.3
```

| Flag                | Description |
|---------------------|-------------|
| openmp=True         | Use openmp for calculations. |
| floatX=float32      | Internal floats are single (32bit) precision. This is required for most GPUs. |
| warn_float64=warn   | Warn if double (64bit) precision floats are accidentally used but continue.  warn_float64=raise might be given instead to stop the calculation if a double precision float is encountered.|
| optimizer=fast_run  | Spend more time optimising the expression graph to make the code run faster. For testing optimizer=fast_compile might be used instead. |
| device=gpu0         | Which device to run the calculation on? Common options are cpu and gpuX, where X is the id of the GPU to be used (commonly gpu0). |
| scan.allow_gc=False | Don't allow garbage collection (freeing of memory) during 'scan' operations. This makes recurrent layers quicker at the expensive of higher memory usage. |
| lib.cnmem=0.4       | Use the CUDA CNMEM library for memory allocation. This will improve GPU performance but requires all the memory to be allocated at the beginning of the calculation. The argument is the proportion of the GPU memory to initially allocate.  As a guide, 0.4 is a good number for training since it allows two runs to both use the same GPU. For programs run on a per-read basis, basecalling and mapping, a smaller proportion like 0.05 is more appropriate. |

 
