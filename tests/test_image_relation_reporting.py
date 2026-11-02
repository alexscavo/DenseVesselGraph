"""Regression for empty-graph degree histograms in CPU result collection."""
import ast
from pathlib import Path
import unittest
import numpy as np


def reporter():
    # This arithmetic seam requires only NumPy, not CUDA/model dependencies.
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_ixi_image_relations.py'
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='degree_distribution')
    scope={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    return scope['degree_distribution']


class DegreeReportingChecks(unittest.TestCase):
    def test_empty_graph_mixed_with_isolates_and_edges(self):
        result=reporter()([{'degree_histogram':[]},{'degree_histogram':[3]},{'degree_histogram':[0,2]}])
        self.assertEqual(result['degree_histogram'],[3,2])
        self.assertAlmostEqual(result['mean_nodes'],5/3)
        self.assertAlmostEqual(result['mean_edges'],1/3)

    def test_all_empty_graphs(self):
        result=reporter()([{'degree_histogram':[]},{'degree_histogram':[]}])
        self.assertEqual(result['degree_histogram'],[])
        self.assertEqual(result['mean_nodes'],0.)
        self.assertEqual(result['mean_edges'],0.)


if __name__=='__main__':unittest.main()
