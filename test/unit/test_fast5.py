from __future__ import print_function
import glob
import os
import unittest
from untangled import fast5


class fast5Test(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('* Fast5')
        self.readdir = os.path.join(os.environ['DATA_DIR'], 'reads')
        self.filename = os.path.join(self.readdir, 'read03.fast5')
        self.section = 'template'
        self.strand_list = os.path.join(os.environ['DATA_DIR'],
                                        'strands.txt')
        self.strands = set(map(lambda r: os.path.join(self.readdir, r),
                               ['read03.fast5', 'read16.fast5']))

    def test_001_get_mapping_data(self):
        #  Interface used by batch.py
        with fast5.Reader(self.filename) as f5:
            ev, _ = f5.get_any_mapping_data(self.section)

    def test_002_get_event_data(self):
        #  Interface used by basecall_network.py
        with fast5.Reader(self.filename) as f5:
            ev = f5.get_section_events(self.section, analysis='Hairpin_Split')
            sn = f5.filename_short

    def test_003_iterate_returns_all(self):
        fast5_files = set(fast5.iterate_fast5(self.readdir, paths=True))
        dir_list = set(glob.glob(os.path.join(self.readdir, '*.fast5')))
        self.assertTrue(fast5_files == dir_list)

    def test_004_iterate_respects_limits(self):
        _LIMIT = 2
        fast5_files = set(fast5.iterate_fast5(self.readdir, paths=True, limit=_LIMIT))
        self.assertTrue(len(fast5_files) == _LIMIT)

    def test_005_iterate_works_with_strandlist(self):
        fast5_files = set(fast5.iterate_fast5(self.readdir, paths=True,
                                              strand_list=self.strand_list))
        self.assertTrue(self.strands == fast5_files)
