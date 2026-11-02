"""Strict 1→2 feature-level migration; all compatible tensors are preserved."""
import math
import torch
from .ops.modules import MSDeformAttn


def migrate_single_level(model, source, fine_prior=1e-4):
    """Copy per-head rows, not a flat prefix (rows interleave levels per head).

    Fine attention starts with the same point distribution and a small nonzero
    mass; it can learn immediately. New fine offsets start locally within one
    fine-grid cell: copying the trained coarse offsets can put an entire new
    sampling branch outside the feature map and kill its offset gradients. The new
    projection keeps its seeded initialization; the new level embedding starts
    at the pretrained embedding. Existing coarse parameters are bitwise copied.
    """
    if not 0 < fine_prior < 1:
        raise ValueError('fine_prior must be strictly between zero and one')
    state = model.state_dict()
    assert source['decoder.level_embed'].shape[0] == 1
    assert state['decoder.level_embed'].shape[0] == 2
    allowed_new = {'fine_input_proj.weight', 'fine_input_proj.bias'}
    assert set(state) - set(source) == allowed_new
    assert not set(source) - set(state)
    expanded = {}
    for name, module in model.named_modules():
        if not isinstance(module, MSDeformAttn):
            continue
        assert module.n_levels == 2
        for field, width in [('sampling_offsets', 3), ('attention_weights', 1)]:
            for suffix in ['weight', 'bias']:
                key = name + '.' + field + '.' + suffix
                old = source[key].reshape(module.n_heads, 1, module.n_points, width, *source[key].shape[1:])
                new = old.repeat(1, 2, *([1] * (old.ndim - 2)))
                if field == 'sampling_offsets':
                    if suffix == 'weight':
                        new[:, 1].zero_()
                    else:
                        default=state[key].reshape(module.n_heads,2,module.n_points,3)
                        new[:, 1] = default[:, 1] / module.n_points
                elif suffix == 'bias':
                    new[:, 1] += math.log(fine_prior)
                expanded[key] = new.reshape_as(state[key])
    copied = []
    for key, value in source.items():
        if key in expanded:
            state[key] = expanded[key]
        elif key == 'decoder.level_embed':
            state[key] = value.repeat(2, 1)
        else:
            assert state[key].shape == value.shape, key
            state[key] = value.clone(); copied.append(key)
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(model.state_dict()[k].cpu(), source[k].cpu()) for k in copied)
    return dict(copied_tensors=len(copied),expanded_tensors=sorted(expanded),new_tensors=sorted(allowed_new),fine_prior=fine_prior,fine_offsets='zero query weights; radial biases at (1..P)/P fine cells')


def set_fine_enabled(model, enabled):
    for module in model.modules():
        if isinstance(module, MSDeformAttn):
            module.disable_fine = not enabled
