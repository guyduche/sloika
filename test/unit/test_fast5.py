import glob
from nose_parameterized import parameterized
import os
import unittest

from sloika import fast5


class IterationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']
        self.readdir = os.path.join(self.dataDir, 'reads')
        self.strand_list = os.path.join(self.dataDir, 'strands.txt')
        self.strands = set([os.path.join(self.readdir, r) for r in ['read03.fast5', 'read16.fast5']])

    def test_iterate_returns_all(self):
        fast5_files = set(fast5.iterate_fast5(self.readdir, paths=True))
        dir_list = set(glob.glob(os.path.join(self.readdir, '*.fast5')))
        self.assertTrue(fast5_files == dir_list)

    def test_iterate_respects_limits(self):
        _LIMIT = 2
        fast5_files = set(fast5.iterate_fast5(self.readdir, paths=True, limit=_LIMIT))
        self.assertTrue(len(fast5_files) == _LIMIT)

    def test_iterate_works_with_strandlist(self):
        fast5_files = set(fast5.iterate_fast5(self.readdir, paths=True,
                                              strand_list=self.strand_list))
        self.assertTrue(self.strands == fast5_files)


class GetAnyMappingDataTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    def test_unknown(self):
        filename = os.path.join(self.dataDir, 'reads', 'read03.fast5')

        with fast5.Reader(filename) as f5:
            ev, _ = f5.get_any_mapping_data('template')
            self.assertEqual(len(ev), 4491)


class ReaderAttributesTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    def test_filename_short(self):
        filename = os.path.join(self.dataDir, 'reads', 'read03.fast5')

        with fast5.Reader(filename) as f5:
            sn = f5.filename_short
            self.assertEqual(f5.filename_short, 'read03')


class GetSectionEventsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    @parameterized.expand([
        [os.path.join('reads', 'read03.fast5'),
         'Hairpin_Split', 4491],
        [os.path.join('test_tell_fast5', 'MINICOL163_20161013_FNFAB42424_MN16275_sequencing'
                                         '_throughput_HG_89618_ch100_read853_strand.fast5'),
         'Segment_Linear', 9946],
        [os.path.join('test_tell_fast5', 'MINICOL235_20161012_FNFAB42418_MN16250_sequencing_throughput_HG'
                                         '_77469_ch100_read7146_strand.fast5'),
         'Segment_Linear', 11145],
    ])
    def test(self, relative_file_path, analysis, number_of_events):
        filename = os.path.join(self.dataDir, relative_file_path)

        with fast5.Reader(filename) as f5:
            ev = f5.get_section_events('template', analysis=analysis)
            self.assertEqual(len(ev), number_of_events)

    @parameterized.expand([
        [os.path.join('test_tell_fast5', 'MINICOL082_20170405_FNFAE28492_MN15845_sequencing_throughput_AMW'
                                         '_Ecoli_yeast_human_R9_5_3_1_nafion_42604_ch127_read7313_strand.fast5')],
        [os.path.join('test_tell_fast5', 'MINICOL082_20170405_FNFAE28492_MN15845_sequencing_throughput_AMW'
                                         '_Ecoli_yeast_human_R9_5_3_1_nafion_42604_ch131_read411_strand.fast5')],
    ])
    def test_should_fail_when_events_section_is_missing(self, relative_file_path):
        '''Segmentation in these files is located at

        /Analyses/Segmentation_000/Summary/segmentation

        not at

        /Analyses/Segment_Linear_000/Summary/split_hairpin

        where we are looking. We initially attempted to look for it in the new place, but then discovered
        that not only the location has changed but also the structure. In particular, there doesn't appear
        to be an event index for the segmentation any more. Event index is required for get_section_events
        function to operate.'''

        filename = os.path.join(self.dataDir, relative_file_path)

        with fast5.Reader(filename) as f5:
            with self.assertRaises(ValueError) as context:
                ev = f5.get_section_events('template')

            # on precise and trusty the second part of this error message (the actual exception) is slightly
            # different from xenial:
            #
            #    KeyError("unable to open object (Symbol table: Can\\\'t open object)",)\',)'
            #    KeyError("Unable to open object (Object \\\'split_hairpin\\\' doesn\\\'t exist)",)\',)'
            #
            # so we compare only the first parts
            msg = repr(context.exception).split('\\n')[0]
            self.assertEqual(msg, 'ValueError(\'Could not retrieve template-complement split '
                                  'point data from attributes of /Analyses/Segmentation_000'
                                  '/Summary/split_hairpin')


class GetReadTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataDir = os.environ['DATA_DIR']

    @parameterized.expand([
        [os.path.join('reads', 'read03.fast5'),
         4617, False],
        [os.path.join('test_tell_fast5', 'MINICOL163_20161013_FNFAB42424_MN16275_sequencing'
                                         '_throughput_HG_89618_ch100_read853_strand.fast5'),
         51129, True],
        [os.path.join('test_tell_fast5', 'MINICOL235_20161012_FNFAB42418_MN16250_sequencing_throughput_HG'
                                         '_77469_ch100_read7146_strand.fast5'),
         55885, True],
        [os.path.join('test_tell_fast5', 'MINICOL082_20170405_FNFAE28492_MN15845_sequencing_throughput_AMW'
                                         '_Ecoli_yeast_human_R9_5_3_1_nafion_42604_ch131_read411_strand.fast5'),
         69443, True],
        [os.path.join('test_tell_fast5', 'MINICOL082_20170405_FNFAE28492_MN15845_sequencing_throughput_AMW'
                                         '_Ecoli_yeast_human_R9_5_3_1_nafion_42604_ch127_read7313_strand.fast5'),
         114400, True],
    ])
    def test(self, relative_file_path, number_of_events, raw):
        filename = os.path.join(self.dataDir, relative_file_path)

        with fast5.Reader(filename) as f5:
            ev = f5.get_read(raw=raw)
            self.assertEqual(len(ev), number_of_events)
