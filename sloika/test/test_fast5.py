from fast5_research import fast5
import unittest

class fast5Test(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print '* Fast5'
        self.readdir = os.path.join(os.path.dirname(__file__), 'data', 'reads')
        self.filename = os.path.join(self.readdir, 'read03.fast5')
        self.section ='template'

    def test_001_get_mapping_data(self):
        #  Interface used by batch.py
        with fast5.Fast5(self.filename) as f5:
            ev, _ = f5.get_any_mapping_data(self.section)

    def test_002_get_event_data(self):
        #  Interface used by basecall_network.py
        with fast5.Fast5(self.filename) as f5:
            ev = f5.get_section_events(self.section)
            sn = f5.filename_short

    """
    def test_003_iterate_fast5(self):
        #  Iteration used
        fast5_files = set(fast5.iterate_fast5(args.input_folder, paths=True,
                                              limit=args.limit,
                                              strand_list=args.strand_list))
    """
