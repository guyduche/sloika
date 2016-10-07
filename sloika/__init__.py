from theano import config
__version__ = '1.0rc1'
__version_info__ = tuple([int(num) for num in __version__.split('.')])

sloika_dtype = config.floatX
