Release 1.1
===========
* Activation functions have been separated into their own module and many new functions have been added.

    See https://wiki/display/~tmassingham/2016/10/17/Activation+functions  
    Note: this rearrangement breaks compatability with older model pickle files.
* Default for training and basecalling are transducer based models.
* Compilation of networks is handled by basecall\_network.
  * Compiled network is saved for future use.  
  * compile\_network.py executable has been removed.
* Recurrent layers
  * New recurrent unit types have been added.
  * Detailed tests to ensure recurrent layers work
  * Type of gate function is now an option on layer initialisation
* Pekarnya server for queuing model training



### Minor changes

* Changed default trimming from ends of sequence.
* Fix to allow trimming of zero events
* Minimum read length (in events) for chunking to take place.
* Removed vestigial networks.py file that has been replaced by the contents of the models/ directory.
* Seed for random number generator can be set on commandline of train\_network.
* Enable HDF5 compression
* Fix to ensure every chunk starts with a non-zero (not stay) label
* Trim first and last events from loss function calculation (burn-in)

Release 1.0
===========
Initial release
