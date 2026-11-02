"""Use the actual shell setup and Python multiprocessing socket allocator."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LimitedRuntimeChecks(unittest.TestCase):
    def test_multiprocessing_socket_fits_scratch_runtime(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='lt-runtime-') as directory:
            env=dict(os.environ,SCRATCH=directory,SLURM_JOB_ID='1344474')
            code='from multiprocessing.connection import Listener; import tempfile; s=Listener(); print(s.address); s.close()'
            result=subprocess.run(['bash','-c','source cluster/jean_zay/limited_runtime.sh\npython3 -c "$1"','check',code],
                cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(result.stdout.strip().startswith(directory+'/'))
            self.assertLess(len(result.stdout.strip().encode()),108)


if __name__=='__main__':unittest.main()
