# Prediction arrays archived 25 October 2026

All original NPZ bytes are retained in a SHA256-verified ZIP. Checkpoints, configs, logs, metrics, records and reconstructions remain in place. Historical analysis scripts using loose NPZ paths require restoration first.

From the repository root:

```bash
unzip -n experiments/storage_cleanup/20261025/multiscale_pretrained-predictions.zip -d .
```

Archive SHA256: `1020cd1cde622c39e6b2e6b59264f0155ea02358bb49cb287b2ab4ba38555aca`. Member checksums: `experiments/storage_cleanup/20261025/multiscale_pretrained-manifest.json`.
Restore only needed members to avoid exhausting the project file quota again.
