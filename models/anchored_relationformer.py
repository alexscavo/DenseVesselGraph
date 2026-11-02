"""Heatmap-anchored RelationFormer (30 Oct 2026).

The transformer, relation head, box/class heads and K8 spline head are the
unchanged RelationFormer modules. Only the object queries change: instead of
learned query embeddings with learned reference points, every object query is
spatially anchored at a peak of a frozen dense node heatmap (the WS2 U-Net
probe). The decoder reference point of a query is its anchor; its positional
query is an MLP of a sine embedding of the anchor; its content is a learned
vector plus a linear function of the anchor score. Node centres are predicted
as a bounded residual (at most +-3 voxels) around the anchor.

Coordinates: arrays are D/H/W, voxel i has its centre at normalized i/size
(``data.augmentations.denormalize_coordinates``). The deformable attention
samples with ``grid_sample(align_corners=False)`` in W/H/D order, where voxel i
is centred at (i+0.5)/size; ``anchors_to_reference_points`` converts.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .deformable_transformer import DeformableTransformer
from .relationformer import MLP, RelationFormer

PROBE_FEATURES = (32, 32, 64, 128, 256, 32)


def proposal_seed(key: str) -> int:
    """Stable per-sample RNG seed for padding anchors (independent of epoch/augmentation)."""
    digest = hashlib.blake2b(str(key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'little') % (2 ** 63 - 1)


def anchor_sine_embedding(anchors: torch.Tensor, width: int, temperature: float = 10_000.0) -> torch.Tensor:
    """DETR-style sine/cosine embedding of normalized D/H/W anchors -> ``[..., width]``."""
    if width % 6:
        raise ValueError('anchor embedding width must be divisible by 6')
    per_axis = width // 3
    index = torch.arange(per_axis, device=anchors.device, dtype=torch.float32)
    frequencies = temperature ** (2 * torch.div(index, 2, rounding_mode='floor') / per_axis)
    phase = anchors.float()[..., None] * (2 * math.pi) / frequencies  # [...,3,per_axis]
    embedding = torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), dim=-1).flatten(-2)
    return embedding.flatten(-2)


def anchors_to_reference_points(anchors: torch.Tensor, size: int | Sequence[int] = 64) -> torch.Tensor:
    """Normalized D/H/W voxel-centre anchors -> deformable-attention W/H/D reference points.

    Repository convention: voxel i <-> i/size. Deformable attention uses
    ``grid_sample(align_corners=False)``, i.e. voxel i <-> (i+0.5)/size, and
    stores reference points as W/H/D (x/y/z).
    """
    shape = anchors.new_tensor(size if isinstance(size, Sequence) else [size] * 3)
    return ((anchors * shape + 0.5) / shape).flip(-1)


class HeatmapProposer(nn.Module):
    """Frozen WS2 node-heatmap U-Net producing exactly ``count`` anchors per patch.

    Output channels: 0 heat logit, 1:4 sub-voxel offsets (voxels, D/H/W), 4
    centerline. Anchors are 3^3 max-pool local maxima of sigmoid(heat), ranked by
    score (stable: ties keep raster order), position = voxel index + offset.
    Missing anchors are padded with score 0 at uniformly random in-patch positions
    from a generator seeded per sample. Always in eval mode, never trainable.
    """

    def __init__(self, count: int, size: int = 64, autocast: bool = True):
        super().__init__()
        from monai.networks.nets import BasicUNet
        self.count, self.size, self.autocast = int(count), int(size), bool(autocast)
        self.unet = BasicUNet(spatial_dims=3, in_channels=1, out_channels=5, features=PROBE_FEATURES)
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True):  # noqa: D401 - frozen BN/dropout regardless of parent mode
        return super().train(False)

    def load_probe(self, path) -> dict:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.unet.load_state_dict(checkpoint['model'], strict=True)
        self.requires_grad_(False)
        return {k: v for k, v in checkpoint.items() if k != 'model'}

    @torch.no_grad()
    def forward(self, images: torch.Tensor, seeds: Sequence[int]):
        if len(seeds) != images.shape[0]:
            raise ValueError('one padding seed is required per sample')
        if tuple(images.shape[-3:]) != (self.size,) * 3:
            raise ValueError(f'proposer expects {self.size}^3 patches')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.autocast and images.is_cuda):
            out = self.unet(images)
        out = out.float()
        heat = out[:, 0].sigmoid()
        peak = heat == F.max_pool3d(heat[:, None], 3, 1, 1)[:, 0]
        score = torch.where(peak, heat, torch.full_like(heat, -1.0)).flatten(1)
        values, index = torch.sort(score, dim=1, descending=True, stable=True)
        values, index = values[:, :self.count], index[:, :self.count]
        valid = values >= 0
        s = self.size
        voxel = torch.stack((torch.div(index, s * s, rounding_mode='floor'),
                             torch.div(index, s, rounding_mode='floor') % s, index % s), -1).float()
        offsets = out[:, 1:4].flatten(2).gather(2, index[:, None].expand(-1, 3, -1)).transpose(1, 2)
        position = (voxel + offsets).clamp(-0.5, s - 0.5)
        scores = torch.where(valid, values, torch.zeros_like(values))
        counts = valid.sum(1)
        for b in range(images.shape[0]):
            missing = self.count - int(counts[b])
            if missing:
                generator = torch.Generator().manual_seed(int(seeds[b]))
                uniform = torch.rand(self.count, 3, generator=generator)[:missing] * s - 0.5
                position[b, ~valid[b]] = uniform.to(position)
        return position / s, scores, counts


def encode_memory(transformer: DeformableTransformer, srcs, masks, pos_embeds):
    """Exactly the encoder half of ``DeformableTransformer.forward`` (unchanged class)."""
    if torch.is_tensor(srcs):
        srcs, masks, pos_embeds = [srcs], [masks], [pos_embeds]
    if not len(srcs) == len(masks) == len(pos_embeds) == len(transformer.level_embed):
        raise ValueError('Feature sources, masks, positions and configured levels must agree')
    src_flatten, mask_flatten, lvl_pos_embed_flatten, pos_embeds_flatten, spatial_shapes = [], [], [], [], []
    for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
        _, _, d, h, w = src.shape
        spatial_shapes.append((d, h, w))
        src = src.flatten(2).transpose(1, 2)
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed_flatten.append(pos_embed + transformer.level_embed[lvl].view(1, 1, -1))
        src_flatten.append(src)
        mask_flatten.append(mask)
        pos_embeds_flatten.append(pos_embed)
    pos_embeds_flatten = torch.cat(pos_embeds_flatten, 1)
    src_flatten = torch.cat(src_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.stack([transformer.get_valid_ratio(m) for m in masks], 1)
    memory = transformer.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                 lvl_pos_embed_flatten, mask_flatten)
    return memory, spatial_shapes, level_start_index, valid_ratios, pos_embeds_flatten, mask_flatten


class AnchoredRelationFormer(RelationFormer):
    """RelationFormer whose object queries are anchored at frozen heatmap peaks."""

    def __init__(self, encoder: nn.Module, decoder: DeformableTransformer, config: Mapping,
                 proposer: HeatmapProposer):
        super().__init__(encoder, decoder, config)
        model = config['model']
        if self.domain_adaptation_enabled or self.node_refinement is not None or self.node_type_embed is not None:
            raise ValueError('anchored model supports the D08 architecture only (no DA/refinement/node types)')
        if self.fine_input_proj is not None:
            raise ValueError('anchored model supports one feature level')
        if self.spline_head is not None and self.spline_head.curve_evidence is not None:
            raise ValueError('curve evidence is not supported')
        decoder_config = model['decoder']
        if int(decoder_config['dummy_tokens']) != 0:
            raise ValueError('dummy tokens are not supported')
        anchored = model.get('anchored', {})
        self.residual_voxels = float(anchored.get('residual_voxels', 3.0))
        self.patch_size = int(config['data']['image_size'][0])
        if proposer.count != self.obj_token or proposer.size != self.patch_size:
            raise ValueError('proposer must return one anchor per object query')
        self.relation_tokens = int(decoder_config['relation_tokens'])
        hidden = self.hidden_dim
        with torch.random.fork_rng(devices=[]):
            # Learned object rows are replaced by anchors; keep only relation rows.
            self.query_embed = nn.Embedding(self.relation_tokens, hidden)
            self.anchor_pos = MLP(hidden, hidden, hidden, 2)
            self.content_embed = nn.Parameter(torch.zeros(hidden))
            self.score_proj = nn.Linear(1, hidden)
            self.offset_embed = MLP(hidden, hidden, 3, 3)
        nn.init.zeros_(self.offset_embed.layers[-1].weight)
        nn.init.zeros_(self.offset_embed.layers[-1].bias)
        self.proposer = proposer

    def query_inputs(self, anchors: torch.Tensor, scores: torch.Tensor):
        """Return decoder ``tgt``, ``query_pos`` and W/H/D reference points for all queries."""
        batch = anchors.shape[0]
        object_pos = self.anchor_pos(anchor_sine_embedding(anchors, self.hidden_dim))
        object_tgt = self.content_embed.view(1, 1, -1) + self.score_proj(scores[..., None].float())
        relation_pos = self.query_embed.weight[None].expand(batch, -1, -1)
        relation_ref = self.decoder.reference_points(relation_pos).sigmoid()
        tgt = torch.cat((object_tgt, torch.zeros_like(relation_pos)), 1)
        query_pos = torch.cat((object_pos, relation_pos), 1)
        reference = torch.cat((anchors_to_reference_points(anchors.float(), self.patch_size), relation_ref), 1)
        return tgt, query_pos, reference

    def forward(self, samples: torch.Tensor, seeds: Sequence[int] | None = None, proposals=None,
                domain_alpha: float = 1.0):
        del domain_alpha
        if samples.ndim != 5:
            raise ValueError('samples must have shape [B,C,D,H,W]')
        if proposals is None:
            if seeds is None:
                raise ValueError('per-sample padding seeds are required')
            anchors, scores, counts = self.proposer(samples, seeds)
        else:
            anchors, scores = proposals[:2]
            counts = None
        anchors, scores = anchors.detach().float(), scores.detach().float()
        encoded = self.encoder(samples)
        mask = torch.zeros(encoded.shape[0], *encoded.shape[2:], dtype=torch.bool, device=encoded.device)
        positions = self.position_embedding(mask)
        projected = self.input_proj(encoded)
        memory, shapes, starts, ratios, pos_flat, mask_flat = encode_memory(self.decoder, projected, mask, positions)
        tgt, query_pos, reference = self.query_inputs(anchors, scores)
        tokens, _ = self.decoder.decoder(tgt, reference, memory, shapes, starts, ratios, pos_flat, query_pos, mask_flat)
        object_tokens = tokens[..., :self.obj_token, :]
        boxes = self.coord_embed(object_tokens).sigmoid()
        residual = self.residual_voxels / self.patch_size * torch.tanh(self.offset_embed(object_tokens))
        predictions = {
            'pred_logits': self.class_embed(object_tokens),
            'pred_nodes': torch.cat((anchors + residual, boxes[..., 3:]), -1),
            'anchors': anchors,
            'anchor_scores': scores,
        }
        if counts is not None:
            predictions['anchor_peak_counts'] = counts
        if self.spline_head is not None and self.spline_head.local_geometry is not None:
            predictions['spline_local_features'] = self._spline_local_features
            predictions['spline_input_shape'] = tuple(samples.shape[-3:])
        return tokens, predictions, projected


def build_anchored_relationformer(config: Mapping, probe_checkpoint=None) -> AnchoredRelationFormer:
    from .deformable_transformer import build_def_detr_transformer
    from .seresnet import build_seresnet
    proposer = HeatmapProposer(int(config['model']['decoder']['object_queries']),
                               int(config['data']['image_size'][0]))
    if probe_checkpoint is not None:
        proposer.load_probe(probe_checkpoint)
    return AnchoredRelationFormer(build_seresnet(config), build_def_detr_transformer(config), config, proposer)


__all__ = ['AnchoredRelationFormer', 'HeatmapProposer', 'anchor_sine_embedding', 'anchors_to_reference_points',
           'build_anchored_relationformer', 'encode_memory', 'proposal_seed']
