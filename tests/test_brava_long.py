"""Focused protocol tests: query identities, full exposure and global scheduling."""
import unittest
import numpy as np
import torch
from scripts.brava_long_experiment import expand_queries,epoch_order,EpochBatches,lr_scale,summarize_updates


class ProtocolTests(unittest.TestCase):
    def test_expansion_preserves_relation_tokens_and_new_scratch_rows(self):
        source={'query_embed.weight':torch.arange(122*4).reshape(122,4).float(),
                'classifier':torch.tensor([2.,3.])}
        target={'query_embed.weight':torch.full((258,4),-17.),'classifier':torch.zeros(2)}
        result=expand_queries(source,target,120)
        torch.testing.assert_close(result['query_embed.weight'][:120],source['query_embed.weight'][:120])
        torch.testing.assert_close(result['query_embed.weight'][256:258],source['query_embed.weight'][120:122])
        torch.testing.assert_close(result['query_embed.weight'][120:256],target['query_embed.weight'][120:256])
        torch.testing.assert_close(result['classifier'],source['classifier'])
        self.assertTrue(torch.all(target['query_embed.weight']==-17.))

    def test_nonquery_architecture_mismatch_is_rejected(self):
        with self.assertRaises(AssertionError):
            expand_queries({'query_embed.weight':torch.zeros(122,4),'head':torch.zeros(3)},
                           {'query_embed.weight':torch.zeros(258,4),'head':torch.zeros(4)},120)

    def test_both_batch_sizes_visit_every_patch_exactly_once(self):
        for size in [16,32]:
            sampler=EpochBatches(size);sampler.epoch=13
            items=[p for batch in sampler for p in batch]
            self.assertEqual(len(items),8768)
            np.testing.assert_array_equal([p[1] for p in items],epoch_order(13))
            self.assertEqual(len({p[1] for p in items}),8768)
            self.assertTrue(all(p[0]==13 for p in items))

    def test_schedule_is_continuous_across_job_boundaries_and_has_floor(self):
        steps=274
        self.assertAlmostEqual(lr_scale(5*steps,steps),1.)
        self.assertGreater(lr_scale(25*steps+1,steps),.9)
        self.assertAlmostEqual(7e-5*lr_scale(300*steps,steps),1e-6)
        self.assertLess(lr_scale(100*steps,steps),lr_scale(25*steps,steps))

    def test_first_update_diagnostics_do_not_break_epoch_average(self):
        result=summarize_updates([{'total':2.,'lr':.1,'step':1,'encoder.gradient_norm':4.},
                                  {'total':4.,'lr':.05,'step':2}])
        self.assertEqual(result['total'],3.)
        self.assertAlmostEqual(result['lr'],.075)
        self.assertNotIn('encoder.gradient_norm',result)
        self.assertNotIn('step',result)


if __name__=='__main__':unittest.main()
