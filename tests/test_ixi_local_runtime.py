"""Exercise the actual multiprocessing FD-sharing failure without GPU inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'cluster/jean_zay/ixi_local_tmp.sh'
PROBE = """
import json, os, tempfile
from multiprocessing.resource_sharer import DupFd
fd = os.open('/dev/null', os.O_RDONLY)
try:
    shared = DupFd(fd)
    copied = shared.detach()
    os.close(copied)
    print(json.dumps(dict(fd_transfer=True, tmpdir=tempfile.gettempdir())))
finally:
    os.close(fd)
"""


class IXILocalRuntime(unittest.TestCase):
    def test_short_scratch_path_fixes_fd_sharing(self):
        scratch = Path(os.environ['SCRATCH'])
        (scratch / 'tmp').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='n07-path-check-', dir=scratch / 'tmp') as staging:
            long_path = Path(staging) / ('original-long-runtime-' * 4)
            long_path.mkdir()
            env = dict(os.environ, TMPDIR=str(long_path), JOBSCRATCH=str(long_path))
            original = subprocess.run([os.sys.executable, '-c', PROBE], env=env,
                                      capture_output=True, text=True, timeout=15)
            self.assertNotEqual(original.returncode, 0)
            self.assertIn('AF_UNIX path too long', original.stderr)
            # Simulate env.sh's JOBSCRATCH override, then source the real fix.
            env['SLURM_JOB_ID'] = 'check-' + str(os.getpid())
            short_path = scratch / 'tmp' / ('icl-' + env['SLURM_JOB_ID'])
            self.assertFalse(short_path.exists(), 'Preserve another check directory')
            try:
                fixed = subprocess.run(
                    ['bash', '-c', 'set -e; export TMPDIR="$JOBSCRATCH"; '
                     'source "$1"; exec "$2" -c "$3"', 'n07-check',
                     str(HELPER), os.sys.executable, PROBE],
                    env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(fixed.returncode, 0, fixed.stderr)
                result = json.loads(fixed.stdout)
                self.assertTrue(result['fd_transfer'])
                self.assertEqual(Path(result['tmpdir']), short_path)
                self.assertTrue(long_path.is_dir(), 'Runtime staging must be preserved')
            finally:
                if short_path.exists():
                    shutil.rmtree(short_path)

    def test_wrappers_apply_fix_after_environment(self):
        for name in ['ixi_local_components_gpu.slurm', 'ixi_local_components_cpu.slurm']:
            body = (ROOT / 'cluster/jean_zay' / name).read_text()
            self.assertLess(body.index('source cluster/jean_zay/env.sh'),
                            body.index('source cluster/jean_zay/ixi_local_tmp.sh'))
            self.assertIn('IXI_LOCAL_TMPDIR', body)


if __name__ == '__main__':
    unittest.main()
