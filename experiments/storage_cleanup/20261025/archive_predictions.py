"""One-off, reviewed archival of dormant prediction arrays; no dataset/model removal."""
import hashlib
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STUDIES = ['fine_feature_architecture', 'multiscale_pretrained', 'node_verifier_feasibility']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    for study in STUDIES:
        root = REPO / 'experiments' / study
        destination = OUT / (study + '-predictions.zip')
        assert not destination.exists(), destination
        files = sorted(root.rglob('*.npz'))
        assert files and all(p.is_file() and not p.is_symlink() for p in files)
        entries = []
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as archive:
            for i, path in enumerate(files):
                st = path.stat()
                record = dict(path=str(path.relative_to(REPO)), bytes=st.st_size,
                              allocated_bytes=st.st_blocks * 512, mtime_ns=st.st_mtime_ns,
                              sha256=sha(path))
                archive.write(path, record['path'])
                entries.append(record)
                if i % 500 == 0:
                    print('archiving', study, i, '/', len(files), flush=True)
        with zipfile.ZipFile(destination) as archive:
            assert len(archive.infolist()) == len(entries)
            for record in entries:
                h = hashlib.sha256()
                with archive.open(record['path']) as f:
                    for chunk in iter(lambda: f.read(8 * 1024**2), b''):
                        h.update(chunk)
                assert h.hexdigest() == record['sha256'], record['path']
                st = (REPO / record['path']).stat()
                assert (st.st_size, st.st_mtime_ns) == (record['bytes'], record['mtime_ns'])
        manifest = dict(study=study, archive=str(destination.relative_to(REPO)),
                        archive_sha256=sha(destination), archive_bytes=destination.stat().st_size,
                        original_bytes=sum(e['bytes'] for e in entries),
                        original_allocated_bytes=sum(e['allocated_bytes'] for e in entries),
                        entries=entries, verified=True, removed=False)
        manifest_path = OUT / (study + '-manifest.json')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        # Delete originals only after every archive member and original stat is verified.
        for record in entries:
            (REPO / record['path']).unlink()
        manifest['removed'] = True
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        (root / 'ARCHIVED_PREDICTIONS.md').write_text(
            '# Prediction arrays archived 25 October 2026\n\n'
            'All original NPZ bytes are retained in a SHA256-verified ZIP. '
            'Checkpoints, configs, logs, metrics, records and reconstructions remain in place. '
            'Historical analysis scripts using loose NPZ paths require restoration first.\n\n'
            'From the repository root:\n\n```bash\n'
            f'unzip -n {destination.relative_to(REPO)} -d .\n```\n\n'
            f'Archive SHA256: `{manifest["archive_sha256"]}`. '
            f'Member checksums: `{manifest_path.relative_to(REPO)}`.\n'
            'Restore only needed members to avoid exhausting the project file quota again.\n')
        print('verified and consolidated', study, len(entries),
              'files, bytes', manifest['original_bytes'], '->', manifest['archive_bytes'], flush=True)


if __name__ == '__main__':
    main()
