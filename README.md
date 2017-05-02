# Installation of system prerequisites

    sudo make deps

# Getting Sloika

After checking out sloika itself, check out and initialise submodules

    make checkout

# Setting up clean development environment

    make cleanDevEnv

# Running unit and acceptance tests in development mode

    make

For this step to function development environment needs to be set up, and make deps must have been installed.

**Note to experts**: These tests will run with the theano flags defined in the `environment` file. If you need to test sloika using a different set of theano flags you can edit this file before running make. Please do report any problems that you run into, although we cannot promise we can help with your configuration.

# Running unit and acceptance tests in CI mode

Please refer to gitlab script for the up-to-date make commands.

# Note on `THEANO_FLAGS`
Please refer to https://wiki/display/RES/Theano+flags for guidance on setting the `THEANO_FLAGS` environmental variables.  Getting these wrong may severely impact performance.

