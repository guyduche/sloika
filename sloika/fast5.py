import h5py
import numpy as np
import os
import sys
import warnings
from glob import glob
from copy import deepcopy
import numpy.lib.recfunctions as nprf
import re

from sloika.decorators import docstring_parameter
from sloika.maths import mad
from sloika.fileio import readtsv

warnings.simplefilter("always", DeprecationWarning)

__base_analysis__ = '/Analyses'
__event_detect_name__ = 'EventDetection'
__raw_path__ = '/Raw/Reads'
__raw_name_old__ = 'RawData'
__raw_path_old__ = '{}/{}/'.format(__base_analysis__, __raw_name_old__)
__raw_signal_path_old__ = '{}/Signal'.format(__raw_path_old__)
__raw_meta_path_old__ = '{}/Meta'.format(__raw_path_old__)
__channel_meta_path__ = '/UniqueGlobalKey/channel_id'
__tracking_id_path__ = 'UniqueGlobalKey/tracking_id'
__context_tags_path__ = 'UniqueGlobalKey/context_tags'

__default_event_path__ = 'Reads'

__default_basecall_2d_analysis__ = 'Basecall_2D'
__default_basecall_1d_analysis__ = 'Basecall_1D'

__default_seq_section__ = '2D'
__default_basecall_fastq__ = 'BaseCalled_{}/Fastq'
__default_basecall_1d_events__ = 'BaseCalled_{}/Events'
__default_basecall_1d_model__ = 'BaseCalled_{}/Model'
__default_basecall_1d_summary__ = 'Summary/basecall_1d_{}'

__default_alignment_analysis__ = 'Alignment'

__default_segmentation_analysis__ = 'Segmentation'
__default_section__ = 'template'
__split_summary_location__ = 'Summary/split_hairpin'

__default_mapping_analysis__ = 'Squiggle_Map'
__default_mapping_events__ = 'SquiggleMapped_{}/Events'
__default_mapping_model__ = 'SquiggleMapped_{}/Model'
__default_mapping_summary__ = 'Summary/squiggle_map_{}'

__default_substep_mapping_analysis__ = 'Substate_Map'
__default_substep_mapping_events__ = '/Events'

__default_basecall_mapping_analysis__ = 'AlignToRef'
__default_basecall_mapping_events__ = 'CurrentSpaceMapped_{}/Events/'
__default_basecall_mapping_model__ = 'CurrentSpaceMapped_{}/Model/'
__default_basecall_mapping_summary__ = '/Summary/current_space_map_{}/'  # under AlignToRef analysis
__default_basecall_alignment_summary__ = '/Summary/genome_mapping_{}/'  # under Alignment analysis

__default_engine_state_path__ = '/EngineStates/'
__temp_fields__ = ('heatsink', 'asic')


class Reader(h5py.File):
    """Class for grabbing data from single read fast5 files. Many attributes/
    groups are assumed to exist currently (we're concerned mainly with reading).
    Needs some development to make robust and for writing.

    """

    def __init__(self, fname):
        super(Reader, self).__init__(fname, 'r')

        # Attach channel_meta as attributes, slightly redundant
        for k, v in self[__channel_meta_path__].attrs.items():
            setattr(self, k, v)
        # Backward compat.
        self.sample_rate = self.sampling_rate

        self.filename_short = os.path.splitext(os.path.basename(self.filename))[0]
        short_name_match = re.search(re.compile(r'ch\d+_file\d+'), self.filename_short)
        self.name_short = self.filename_short
        if short_name_match:
            self.name_short = short_name_match.group()

    def _join_path(self, *args):
        return '/'.join(args)

    @property
    def channel_meta(self):
        """Channel meta information as python dict"""
        return dict(self[__channel_meta_path__].attrs)

    @property
    def tracking_id(self):
        """Tracking id meta information as python dict"""
        return dict(self[__tracking_id_path__].attrs)

    @property
    def attributes(self):
        """Attributes for a read, assumes one read in file"""
        return dict(self.get_read(group=True).attrs)

    def summary(self, rename=True, delete=True, scale=True):
        """A read summary, assumes one read in file"""
        to_rename = list(zip(
            ('start_mux', 'abasic_found', 'duration', 'median_before'),
            ('mux', 'abasic', 'strand_duration', 'pore_before')
        ))
        to_delete = ('read_number', 'scaling_used')

        data = deepcopy(self.attributes)
        data['filename'] = os.path.basename(self.filename)
        data['run_id'] = self.tracking_id['run_id']
        data['channel'] = self.channel_meta['channel_number']
        if scale:
            data['duration'] /= self.channel_meta['sampling_rate']
            data['start_time'] /= self.channel_meta['sampling_rate']

        if rename:
            for i, j in to_rename:
                try:
                    data[j] = data[i]
                    del data[i]
                except KeyError:
                    pass
        if delete:
            for i in to_delete:
                try:
                    del data[i]
                except KeyError:
                    pass

        for key in data:
            if isinstance(data[key], float):
                data[key] = np.around(data[key], 4)

        return data

    ###
    # Extracting read event data

    def get_reads(self, group=False, raw=False, read_numbers=None):
        """Iterator across event data for all reads in file

        :param group: return hdf group rather than event data
        """
        if not raw:
            event_group = self.get_analysis_latest(__event_detect_name__)
            event_path = self._join_path(event_group, __default_event_path__)
            reads = self[event_path]
        else:
            try:
                reads = self[__raw_path__]
            except:
                yield self.get_raw()[0]

        if read_numbers is None:
            it = list(reads.keys())
        else:
            it = (k for k in list(reads.keys())
                  if reads[k].attrs['read_number'] in read_numbers)

        if group == 'all':
            for read in it:
                yield reads[read], read
        elif group:
            for read in it:
                yield reads[read]
        else:
            for read in it:
                if not raw:
                    yield self._get_read_data(reads[read])
                else:
                    yield self._get_read_data_raw(reads[read])

    def get_read(self, group=False, raw=False, read_number=None):
        """Like get_reads, but only the first read in the file

        :param group: return hdf group rather than event/raw data
        """
        if read_number is None:
            return next(self.get_reads(group, raw))
        else:
            return next(self.get_reads(group, raw, read_numbers=[read_number]))

    def _get_read_data(self, read, indices=None):
        """Private accessor to read event data"""
        # We choose the following to always be floats
        float_fields = ('start', 'length', 'mean', 'stdv')

        events = read['Events']

        # We assume that if start is an int or uint the data is in samples
        #    else it is in seconds already.
        needs_scaling = False
        if events['start'].dtype.kind in ['i', 'u']:
            needs_scaling = True

        dtype = np.dtype([(
            d[0], 'float') if d[0] in float_fields else d
            for d in events.dtype.descr
        ])
        data = None
        with events.astype(dtype):
            if indices is None:
                data = events[()]
            else:
                try:
                    data = events[indices[0]:indices[1]]
                except:
                    raise ValueError(
                        'Cannot retrieve events using {} as indices'.format(indices)
                    )

        # File spec mentions a read.attrs['scaling_used'] attribute,
        #    its not clear what this is. We'll ignore it and hope for
        #    the best.
        if needs_scaling:
            data['start'] /= self.sample_rate
            data['length'] /= self.sample_rate
        return data

    def _get_read_data_raw(self, read, indices=None, scale=True):
        """Private accessor to read raw data"""
        raw = read['Signal']
        dtype = float if scale else int

        data = None
        with raw.astype(dtype):
            if indices is None:
                data = raw[()]
            else:
                try:
                    data = raw[indices[0]:indices[1]]
                except:
                    raise ValueError(
                        'Cannot retrieve events using {} as indices'.format(indices)
                    )

        # Scale data to pA
        if scale:
            meta = self.channel_meta
            raw_unit = meta['range'] / meta['digitisation']
            data = (data + meta['offset']) * raw_unit
        return data

    def get_read_stats(self):
        """Combines stats based on events with output of .summary, assumes a
        one read file.

        """
        data = deepcopy(self.summary())
        read = self.get_read()
        n_events = len(read)
        q = np.percentile(read['mean'], [10, 50, 90])
        data['range_current'] = q[2] - q[0]
        data['median_current'] = q[1]
        data['num_events'] = n_events
        data['median_sd'] = np.median(read['stdv'])
        data['median_dwell'] = np.median(read['length'])
        data['sd_current'] = np.std(read['mean'])
        data['mad_current'] = mad(read['mean'])
        data['eps'] = data['num_events'] / data['strand_duration']
        return data

    ###
    # Raw Data

    @docstring_parameter(__raw_path_old__)
    def get_raw(self, scale=True):
        """Get raw data in file, might not be present.

        :param scale: Scale data to pA? (rather than ADC values)

        .. warning::
            This method is deprecated and should not be used, instead use
            .get_read(raw=True) to read both MinKnow conformant files
            and previous Tang files.
        """
        warnings.warn(
            "'fast5.get_raw()' is deprecated, use 'fast5.get_read(raw=True)'.",
            DeprecationWarning,
            stacklevel=2
        )
        try:
            raw = self[__raw_signal_path_old__]
            meta = self[__raw_meta_path_old__].attrs
        except KeyError:
            raise KeyError('No raw data available.')

        raw_data = None
        if scale:
            raw_data = raw[()].astype('float')
            raw_unit = meta['range'] / meta['digitisation']
            raw_data = (raw_data + meta['offset']) * raw_unit
        else:
            raw_data = raw[()]
        return raw_data, meta['sample_rate']

    ###
    # Analysis path resolution

    def get_analysis_latest(self, name):
        """Get group of latest (present) analysis with a given base path.

        :param name: Get the (full) path of newest analysis with a given base
            name.
        """
        try:
            return self._join_path(
                __base_analysis__,
                sorted([x for x in list(self[__base_analysis__].keys()) if name in x])[-1]
            )
        except (IndexError, KeyError):
            raise IndexError('No analyses with name {} present.'.format(name))

    def get_analysis_new(self, name):
        """Get group path for new analysis with a given base name.

        :param name: desired analysis name
        """

        # Formatted as 'base/name_000'
        try:
            latest = self.get_analysis_latest(name)
            root, counter = latest.rsplit('_', 1)
            counter = int(counter) + 1
        except IndexError:
            # Nothing present
            root = self._join_path(
                __base_analysis__, name
            )
            counter = 0
        return '{}_{:03d}'.format(root, counter)

    def get_model(self, section=__default_section__, analysis=__default_mapping_analysis__):
        """Get model used for squiggle mapping"""
        base = self.get_analysis_latest(analysis)
        model_path = self._join_path(base, __default_mapping_model__.format(section))
        return self[model_path][()]

    # The remaining are methods to read and write data as chimaera produces
    #    It is necessarily all a bit nasty, but should provide a more
    #    consistent interface to the files. Paths are defaulted

    ###
    # Temperature etc.

    @docstring_parameter(__default_engine_state_path__)
    def get_engine_state(self, state, time=None):
        """Retrieve engine state from {}, either across the whole read
        (default) or at a given time.

        :param state: name of engine state
        :param time: time (in seconds) at which to retrieve temperature

        """
        location = self._join_path(
            __default_engine_state_path__, state
        )
        states = self[location][()]
        if time is None:
            return states
        else:
            i = np.searchsorted(states['time'], time) - 1
            return states[state][i]

    @docstring_parameter(__default_engine_state_path__, __temp_fields__)
    def get_temperature(self, time=None, field=__temp_fields__[0]):
        """Retrieve temperature data from {}, either across the whole read
        (default) or at a given time.

        :param time: time at which to get temperature
        :param field: one of {}

        """
        if field not in __temp_fields__:
            raise RuntimeError("'field' argument must be one of {}.".format(__temp_fields__))

        return self.get_engine_state('minion_{}_temperature'.format(field), time)

    ###
    # Template/complement splitting data
    @docstring_parameter(__base_analysis__)
    def get_split_data(self, analysis=__default_segmentation_analysis__):
        """Get template-complement segmentation data.

        :param analysis: Base analysis name (under {})
        """

        location = self._join_path(
            self.get_analysis_latest(analysis), __split_summary_location__
        )
        try:
            return dict(self[location].attrs)
        except Exception as e:
            raise ValueError(
                'Could not retrieve template-complement split point data from attributes of {}\n{}'.format(location,
                                                                                                           repr(e)))

    @docstring_parameter(__base_analysis__)
    def get_section_event_indices(self, analysis=__default_segmentation_analysis__):
        """Get two tuples indicating the event indices for the template and
        complement boundaries.

        :param analysis: Base analysis path (under {})
        """

        # TODO: if the below fails, calculating the values on the fly would be
        #       a useful feature. Which brings to mind could we do such a thing
        #       in all cases of missing data? Probably not reasonble.
        attrs = self.get_split_data(analysis)
        try:
            return (
                (attrs['start_index_temp'], attrs['end_index_temp']),
                (attrs['start_index_comp'], attrs['end_index_comp'])
            )
        except Exception as e:
            raise ValueError('Could not retrieve template-complement segmentation data\n{}'.format(repr(e)))

    @docstring_parameter(__base_analysis__)
    def get_section_events(self, section, analysis=__default_segmentation_analysis__):
        """Get the template event data.

        :param analysis: Base analysis path (under {})
        """

        indices = self.get_section_event_indices(analysis)
        read = self.get_read(group=True)
        events = None
        if section == 'template':
            events = self._get_read_data(read, indices[0])
        elif section == 'complement':
            events = self._get_read_data(read, indices[1])
        else:
            raise ValueError(
                '"section" parameter for fetching events must be "template" or "complement".'
            )
        return events

    @docstring_parameter(__base_analysis__)
    def get_basecall_data(self, section=__default_section__, analysis=__default_basecall_1d_analysis__):
        """Read the annotated basecall_1D events from the fast5 file.

        :param section: String to use in paths, e.g. 'template' or 'complement'.
        :param analysis: Base analysis name (under {})
        """

        base = self.get_analysis_latest(analysis)
        events_path = self._join_path(base, __default_basecall_1d_events__.format(section))
        try:
            return self[events_path][()]
        except:
            raise ValueError('Could not retrieve basecall_1D data from {}'.format(events_path))

    @docstring_parameter(__base_analysis__)
    def get_alignment_attrs(self, section=__default_section__, analysis=__default_alignment_analysis__):
        """Read the annotated alignment meta data from the fast5 file.

        :param section: String to use in paths, e.g. 'template' or 'complement'.
        :param analysis: Base analysis name (under {})

        """

        attrs = None
        base = self.get_analysis_latest(analysis)
        attr_path = self._join_path(base,
                                    __default_basecall_alignment_summary__.format(section))
        try:
            attrs = dict(self[attr_path].attrs)
        except:
            raise ValueError('Could not retrieve alignment attributes from {}'.format(attr_path))

        return attrs

    ###
    # Mapping data

    @docstring_parameter(__base_analysis__)
    def get_mapping_data(self, section=__default_section__, analysis=__default_mapping_analysis__, get_model=False):
        """Read the annotated mapping events from the fast5 file.

        .. note::
            The seq_pos column for the events table returned from basecall_mapping is
            adjusted to be the genome position (consistent with squiggle_mapping)

        :param section: String to use in paths, e.g. 'template' or 'complement'.
        :param analysis: Base analysis name (under {}). For basecall mapping
            use analysis = 'AlignToRef'.
        """

        events = None
        if analysis == __default_mapping_analysis__:
            # squiggle_mapping
            base = self.get_analysis_latest(analysis)
            event_path = self._join_path(base, __default_mapping_events__.format(section))
            try:
                events = self[event_path][()]
            except:
                raise ValueError('Could not retrieve squiggle_mapping data from {}'.format(event_path))
            if get_model:
                model_path = self._join_path(base, __default_mapping_model__.format(section))
                try:
                    model = self[model_path][()]
                except:
                    raise ValueError('Could not retrieve squiggle_mapping model from {}'.format(model_path))

            attrs = self.get_mapping_attrs(section=section)

        elif analysis == __default_substep_mapping_analysis__:
            # substep mapping
            base = self.get_analysis_latest(analysis)
            event_path = self._join_path(base, __default_substep_mapping_events__.format(section))
            try:
                events = self[event_path][()]
            except:
                raise ValueError('Could not retrieve substep_mapping data from {}'.format(event_path))
            attrs = None
            if get_model:
                raise NotImplementedError('Retrieving substep model not implemented.')

        else:
            # basecall_mapping
            base = self.get_analysis_latest(analysis)
            event_path = self._join_path(base, __default_basecall_mapping_events__.format(section))
            try:
                events = self[event_path][()]
            except:
                raise ValueError('Could not retrieve basecall_mapping data from {}'.format(event_path))
            if get_model:
                model_path = self._join_path(base, __default_basecall_mapping_model__.format(section))
                try:
                    model = self[model_path][()]
                except:
                    raise ValueError('Could not retrieve squiggle_mapping model from {}'.format(model_path))

            # Modify seq_pos to be the actual genome position (consistent with squiggle_map)
            attrs = self.get_mapping_attrs(section=section, analysis=__default_alignment_analysis__)
            if attrs['direction'] == '+':
                events['seq_pos'] = events['seq_pos'] + attrs['ref_start']
            else:
                events['seq_pos'] = attrs['ref_stop'] - events['seq_pos']

        # add transition field
        if attrs:
            move = np.ediff1d(events['seq_pos'], to_begin=0)
            if attrs['direction'] == '-':
                move *= -1
            if 'move' not in events.dtype.names:
                events = nprf.append_fields(events, 'move', move)
            else:
                events['move'] = move

        if get_model:
            return events, model
        else:
            return events

    def get_any_mapping_data(self, section=__default_section__, attrs_only=False, get_model=False):
        """Convenience method for extracting whatever mapping data might be
        present, favouring squiggle_mapping output over basecall_mapping.

        :param section: (Probably) one of '2D', 'template', or 'complement'
        :param attrs_only: Use attrs_only=True to return mapping attributes without events

        :returns: the tuple (events, attrs) or attrs only
        """
        events = None
        attrs = None

        try:
            if not attrs_only:
                events = self.get_mapping_data(section=section, get_model=get_model)
            attrs = self.get_mapping_attrs(section=section)
        except:
            try:
                if not attrs_only:
                    events = self.get_mapping_data(section=section,
                                                   analysis=__default_basecall_mapping_analysis__, get_model=get_model)
                attrs = self.get_mapping_attrs(section=section,
                                               analysis=__default_alignment_analysis__)
            except:
                raise ValueError(
                    "Cannot find any mapping data at paths I know about. "
                    "Consider using get_mapping_data() with analysis argument."
                )
        if not attrs_only:
            if get_model:
                return events[0], attrs, events[1]
            else:
                return events, attrs
        else:
            return attrs

    @docstring_parameter(__base_analysis__)
    def get_mapping_attrs(self, section=__default_section__, analysis=__default_mapping_analysis__):
        """Read the annotated mapping meta data from the fast5 file.
        Names which are inconsistent between squiggle_mapping and basecall_mapping are added to
        basecall_mapping (thus duplicating the attributes in basecall mapping).

        :param section: String to use in paths, e.g. 'template' or 'complement'.
        :param analysis: Base analysis name (under {})
                         For basecall mapping use analysis = 'Alignment'
        """

        attrs = None
        if analysis == __default_mapping_analysis__:
            # squiggle_mapping
            base = self.get_analysis_latest(analysis)
            attr_path = self._join_path(base, __default_mapping_summary__.format(section))
            try:
                attrs = dict(self[attr_path].attrs)
            except:
                raise ValueError('Could not retrieve squiggle_mapping meta data from {}'.format(attr_path))
        else:
            # basecall_mapping

            # AligToRef attributes (set AlignToRef first so that Alignment attrs are not overwritten)
            base = self.get_analysis_latest(__default_basecall_mapping_analysis__)
            attr_path = self._join_path(base, __default_basecall_mapping_summary__.format(section))
            try:
                attrs = dict(self[attr_path].attrs)
            except:
                raise ValueError('Could not retrieve basecall_mapping meta data from {}'.format(attr_path))

            # Rename some of the fields
            rename = [
                ('genome_start', 'ref_start'),
                ('genome_end', 'ref_stop'),
            ]
            for old, new in rename:
                attrs[new] = attrs.pop(old)

            # Alignment attributes
            base = self.get_analysis_latest(analysis)
            attr_path = self._join_path(
                base, __default_basecall_alignment_summary__.format(section))
            try:
                genome = self[attr_path].attrs.get('genome')
            except:
                raise ValueError('Could not retrieve basecall_mapping genome field from {}'.format(attr_path))

            try:
                attrs['reference'] = self.get_reference_fasta(section=section).decode('utf-8').split('\n')[1]
            except:
                raise ValueError('Could not retrieve basecall_mapping fasta from Alignment analysis')

            # Add attributes with keys consistent with Squiggle_map
            rc = b'_rc'
            is_rc = genome.endswith(rc)
            attrs['ref_name'] = genome[:-len(rc)] if is_rc else genome
            attrs['direction'] = '-' if is_rc else '+'

        # Trim any other fields, the allowed are those produced by
        #   squiggle_mapping. We allow strand_score but do not require
        #   it since our writer does not require it.
        required = [
            'direction', 'ref_start', 'ref_stop', 'ref_name',
            'num_skips', 'num_stays', 'reference'
        ]
        additional = ['strand_score', 'shift', 'scale', 'drift', 'var', 'scale_sd', 'var_sd']
        keep = required + additional
        assert set(required).issubset(set(attrs)), 'Required mapping attributes not found'
        for key in (set(attrs) - set(keep)):
            del(attrs[key])

        return attrs

    ###
    # Sequence data

    @docstring_parameter(__base_analysis__)
    def get_fastq(self, analysis=__default_basecall_2d_analysis__, section=__default_seq_section__, custom=None):
        """Get the fastq (sequence) data.

        :param analysis: Base analysis name (under {})
        :param section: (Probably) one of '2D', 'template', or 'complement'
        :param custom: Custom hdf path overriding all of the above.
        """

        err_msg = 'Could not retrieve sequence data from {}'

        if custom is not None:
            location = custom
        else:
            location = self._join_path(
                self.get_analysis_latest(analysis), __default_basecall_fastq__.format(section)
            )
        try:
            return self[location][()]
        except:
            # Did we get given section != 2D and no analysis, that's
            #    more than likely incorrect. Try alternative analysis
            if section != __default_seq_section__ and analysis == __default_basecall_2d_analysis__:
                location = self._join_path(
                    self.get_analysis_latest(__default_basecall_1d_analysis__),
                    __default_basecall_fastq__.format(section)
                )
                try:
                    return self[location][()]
                except:
                    raise ValueError(err_msg.format(location))
            else:
                raise ValueError(err_msg.format(location))

    @docstring_parameter(__base_analysis__)
    def get_sam(self, analysis=__default_alignment_analysis__, section=__default_seq_section__, custom=None):
        """Get SAM (alignment) data.

        :param analysis: Base analysis name (under {})
        :param section: (Probably) one of '2D', 'template', or 'complement'
        :param custom: Custom hdf path overriding all of the above.
        """

        if custom is not None:
            location = custom
        else:
            location = self._join_path(
                self.get_analysis_latest(analysis), 'Aligned_{}'.format(section), 'SAM'
            )
        try:
            return self[location][()]
        except:
            raise ValueError('Could not retrieve SAM data from {}'.format(location))

    @docstring_parameter(__base_analysis__)
    def get_reference_fasta(self, analysis=__default_alignment_analysis__, section=__default_seq_section__,
                            custom=None):
        """Get fasta sequence of known DNA fragment for the read.

        :param analysis: Base analysis name (under {})
        :param section: (Probably) one of '2D', 'template', or 'complement'
        :param custom: Custom hdf path overriding all of the above.
        """

        if custom is not None:
            location = custom
        else:
            location = self._join_path(
                self.get_analysis_latest(analysis), 'Aligned_{}'.format(section), 'Fasta'
            )
        try:
            return self[location][()]
        except:
            raise ValueError('Could not retrieve sequence data from {}'.format(location))


def iterate_fast5(path='Stream', strand_list=None, paths=False, limit=None):
    """Iterate over directory of fast5 files, optionally only returning those in list

    :param path: Directory in which single read fast5 are located or filename.
    :param strand_list: List of strands, can be a python list of delimited
        table. If the later and a filename field is present, this is used
        to locate files. If a file is given and a strand field is present,
        the directory index file is searched for and filenames built from that.
    :param paths: Yield file paths instead of fast5 objects.
    :param limit: Limit number of files to consider
    """
    if strand_list is None:
        #  Could make glob more specific to filename pattern expected
        if os.path.isdir(path):
            files = glob(os.path.join(path, '*.fast5'))
        else:
            files = [path]
    else:
        if isinstance(strand_list, list):
            files = [os.path.join(path, x) for x in strand_list]
        else:
            reads = readtsv(strand_list)
            if 'filename' in reads.dtype.names:
                #  Strand list contains a filename column
                if sys.version_info[0] < 3:
                    files = [os.path.join(path, x) for x in reads['filename']]
                else:
                    files = [os.path.join(path, x.decode('utf-8')) for x in reads['filename']]
            else:
                raise KeyError("Strand file does not contain required field 'filename'.\n")

    for f in files[:limit] :
        if not os.path.exists(f):
            sys.stderr.write('File {} does not exist, skipping\n'.format(f))
            continue
        if not paths:
            fh = Reader(f)
            yield fh
            fh.close()
        else:
            yield os.path.abspath(f)
