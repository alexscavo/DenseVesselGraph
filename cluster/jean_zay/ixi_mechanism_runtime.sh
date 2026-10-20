#!/usr/bin/env bash
# Source after stage_spline_runtime.sh; only repair the job-local import path.
# The archived SymPy is incomplete; use protected SymPy/mpmath without promoting
# obsolete standard-library backports in the module's site-packages directory.
mkdir -p "$spline_runtime_dir/ixi-protected"
for package in sympy mpmath; do
  ln -s "$spline_runtime_python_prefix/lib/python3.11/site-packages/$package" "$spline_runtime_dir/ixi-protected/$package"
done
export PYTHONPATH="$spline_runtime_dir/ixi-protected:$PYTHONPATH"
