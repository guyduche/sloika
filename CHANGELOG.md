Release 1.1
===========
* Activation functions have been separated into their own module and many new functions have been added.  
    See https://wiki/display/~tmassingham/2016/10/17/Activation+functions  
    Note: this rearrangement breaks compatability with older model pickle files.
* Refactoring of NBASE constant.  
    Now a single source of responsibility `sloika/variables.py`.  
    Models importing `_NBASE` from `sloika/module_tools.py` should now import `NBASE` instead.
* Default for training and basecalling are transducer based models.
* Compilation of networks is handled automatically by `basecall_network.py`
  * Compiled network may be saved for future use.  
  * `compile_network.py` executable has been removed.
* Recurrent layers.
  * New recurrent unit types have been added.
  * Detailed tests to ensure recurrent layers work.
  * Type of gate function is now an option on layer initialisation.
* Pekarnya server for queuing model training.
* Considerable work on the building and testing infrastucture.



### Minor changes

* Changed default trimming from ends of sequence.
* Fix to allow trimming of zero events.
* Minimum read length (in events) for chunking to take place.
* Removed vestigial `networks.py` file that has been replaced by the contents of the `models/` directory.
* Seed for random number generator can be set on commandline of `train_network.py`.
* Enable HDF5 compression.
* Fix to ensure every chunk starts with a non-zero (not stay) label.
* Trim first and last events from loss function calculation (burn-in).
* Fix bug in how kmers are merged into sequence in low complexity regions.

Release 1.0
===========
Initial release
