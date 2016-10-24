Release 1.1
===========
* Activation functions have been separated into their own module and many new functions have been added.

    See https://wiki/display/~tmassingham/2016/10/17/Activation+functions
* Default for training and basecalling are transducer based models.
* Compilation of networks is handled by basecall_network.

    Compiled network is saved for future use.  
    compile_network.py executable has been removed.


### Minor changes

* Changed default trimming from ends of sequence.
* Minimum read length (in events) for chunking to take place.
* Removed vestigial networks.py file that has been replaced by the contents of the models/ directory.
* Seed for random number generator can be set on commandline of train_network.

Release 1.0
===========
Initial release
