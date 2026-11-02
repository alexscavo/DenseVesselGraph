"""Print the completed R06 study and any separately collected geometry supplement."""
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'experiments/r06_relation_tokens_20261027'
print((out/'report.md').read_text())
supplement=out/'geometry_completion_20261027/report.md'
if supplement.exists():print('\n'+supplement.read_text())
else:print('\nBranch/Curve reporting supplement pending; no further training is running.')
