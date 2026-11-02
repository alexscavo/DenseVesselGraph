"""Minimal 3D RelationFormer model used by the supported baseline."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .deformable_transformer import (
    DeformableTransformer,
    build_def_detr_transformer,
)
from .domain_adaptation import DomainDiscriminator
from .position_encoding import PositionEmbeddingSine3D


class MLP(nn.Module):
    """Baseline feed-forward head with checkpoint-compatible layer names."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.num_layers = int(num_layers)
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(source, destination)
            for source, destination in zip(
                [input_dim, *hidden], [*hidden, output_dim]
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for index, layer in enumerate(self.layers):
            output = (
                F.relu(layer(output))
                if index < self.num_layers - 1
                else layer(output)
            )
        return output


class RelationFormer(nn.Module):
    """Encode a 3D volume and predict graph-node tokens and coordinates."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: DeformableTransformer,
        config: Mapping,
    ) -> None:
        super().__init__()
        model = config["model"]
        decoder_config = model["decoder"]
        hidden_dim = int(decoder_config["hidden_dim"])
        object_queries = int(decoder_config["object_queries"])
        relation_tokens = int(decoder_config["relation_tokens"])
        dummy_tokens = int(decoder_config["dummy_tokens"])

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = object_queries + relation_tokens + dummy_tokens
        self.hidden_dim = hidden_dim
        self.position_embedding = PositionEmbeddingSine3D(channels=hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.input_proj = nn.Conv3d(encoder.num_features, hidden_dim, kernel_size=1)
        self.fine_input_proj = None
        self._multiscale_features = None
        self.fine_feature_mode = 'normal'  # Inference-only contribution diagnostics.
        if int(decoder_config['feature_levels']) == 2:
            fine_stage=decoder_config.get('fine_feature_stage','layer3')
            if fine_stage not in ('layer2','layer3'):
                raise ValueError('fine_feature_stage must be layer2 or layer3')
            fine_channels=encoder.num_features // (4 if fine_stage=='layer2' else 2)
            with torch.random.fork_rng(devices=[]):
                self.fine_input_proj = nn.Conv3d(fine_channels, hidden_dim, kernel_size=1)
            getattr(self.encoder,fine_stage).register_forward_hook(self._capture_multiscale_features)
        self.class_embed = nn.Linear(hidden_dim, int(model["num_classes"]))
        # Optional patch-graph degree class: isolated, terminal, degree-2, junction.
        # Keep disabled for exact compatibility with existing checkpoints.
        # Keep subsequent shared-parameter initialization identical for the
        # matched with/without-node-type ablation under the same random seed.
        if model.get("node_type", {}).get("enabled", False):
            with torch.random.fork_rng(devices=[]):
                self.node_type_embed = nn.Linear(hidden_dim, 4)
        else:
            self.node_type_embed = None
        self.coord_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.obj_token = object_queries

        relation_input_dim = hidden_dim * (2 + relation_tokens)
        self.relation_embed = MLP(relation_input_dim, hidden_dim, 2, 3)
        spline = model.get('spline', {})
        self.spline_head = None
        if spline.get('enabled', False):
            from .spline import SplineHead
            with torch.random.fork_rng(devices=[]):
                self.spline_head = SplineHead(relation_input_dim, hidden_dim,
                                              spline.get('controls', 8), spline.get('samples', 64),
                                              spline.get('residual_scale', 'mm'), spline.get('curve_evidence'),
                                              family=spline.get('family', 'bspline'),
                                              token_inputs=spline.get('token_inputs', 'full'),
                                              local_geometry=spline.get('local_geometry'))
        self.node_refinement = None
        node_refinement = model.get('node_refinement', {})
        if node_refinement.get('enabled', False):
            from .node_refinement import LocalNodeRefinement
            with torch.random.fork_rng(devices=[]):
                self.node_refinement = LocalNodeRefinement(hidden_dim,
                    radius_vox=node_refinement.get('radius_vox', 4.),
                    image_mode=node_refinement.get('image_mode', 'full'),
                    refine_classification=node_refinement.get('refine_classification', True),
                    detach_features=node_refinement.get('detach_features', True),
                    compact_output=node_refinement.get('compact_output', False))
        self._spline_local_features = None
        if self.node_refinement is not None or (self.spline_head is not None and self.spline_head.local_geometry is not None):
            # Capture early features before SENet max pooling; this adds no
            # CNN computation or trainable image encoder.
            assert hasattr(self.encoder.layer0, 'relu1') and self.encoder.layer0.conv1.stride == (2, 2, 2)
            assert self.encoder.layer0.conv1.kernel_size == (7, 7, 7) and self.encoder.layer0.conv1.padding == (3, 3, 3)
            assert self.encoder.layer0.bn1.num_features == spline.get('local_geometry', {}).get('channels', 64)
            self.encoder.layer0.relu1.register_forward_hook(self._capture_local_features)
        self.domain_adaptation_enabled = bool(
            config.get("domain_adaptation", {}).get("enabled", False)
        )
        if self.domain_adaptation_enabled:
            domain_width = int(config["domain_adaptation"]["hidden_size"])
            self.backbone_domain_discriminator = DomainDiscriminator(
                hidden_dim, hidden_size=domain_width
            )
            self.instance_domain_discriminator = DomainDiscriminator(
                hidden_dim * self.num_queries, hidden_size=domain_width
            )

    def _capture_local_features(self, module, inputs, output):
        self._spline_local_features = output

    def _capture_multiscale_features(self, module, inputs, output):
        self._multiscale_features = output  # Keep backbone gradients connected.

    def forward(
        self, samples: torch.Tensor, domain_alpha: float = 1.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Return all tokens, node predictions, and projected encoder features.

        `samples` has shape `[B,C,D,H,W]`. The returned token tensor has shape
        `[B,Q,H]`; `pred_logits` is `[B,object_queries,num_classes]` and
        `pred_nodes` is `[B,object_queries,6]` in normalized coordinates.
        """

        if samples.ndim != 5:
            raise ValueError("samples must have shape [B,C,D,H,W]")

        encoded = self.encoder(samples)
        mask = torch.zeros(
            encoded.shape[0],
            *encoded.shape[2:],
            dtype=torch.bool,
            device=encoded.device,
        )
        positions = self.position_embedding(mask)
        projected_features = self.input_proj(encoded)
        sources, masks, embeddings = projected_features, mask, positions
        if self.fine_input_proj is not None:
            fine = self._multiscale_features
            if self.fine_feature_mode == 'roll':
                fine = torch.roll(fine, shifts=(1, 1, 1), dims=(-3, -2, -1))
            elif self.fine_feature_mode != 'normal':
                raise ValueError(self.fine_feature_mode)
            fine = self.fine_input_proj(fine)
            fine_mask = torch.zeros(fine.shape[0], *fine.shape[2:], device=fine.device, dtype=torch.bool)
            # Preserve the original coarse level at index zero and its key names.
            sources = [projected_features, fine]
            masks = [mask, fine_mask]
            embeddings = [positions, self.position_embedding(fine_mask)]
        tokens = self.decoder(
            sources,
            masks,
            self.query_embed.weight,
            embeddings,
        )
        object_tokens = tokens[..., : self.obj_token, :]
        predictions = {
            "pred_logits": self.class_embed(object_tokens),
            "pred_nodes": self.coord_embed(object_tokens).sigmoid(),
        }
        if self.node_refinement is not None:
            predictions['pred_nodes'], predictions['pred_logits'] = self.node_refinement(
                object_tokens, predictions['pred_nodes'], predictions['pred_logits'],
                self._spline_local_features, samples.shape[-3:])
        if self.spline_head is not None and self.spline_head.curve_evidence is not None:
            predictions['curve_image_features'] = self.spline_head.curve_evidence.encode(samples)
        if self.spline_head is not None and self.spline_head.local_geometry is not None:
            predictions['spline_local_features'] = self._spline_local_features
            predictions['spline_input_shape'] = tuple(samples.shape[-3:])
        if self.node_type_embed is not None:
            predictions["pred_node_types"] = self.node_type_embed(object_tokens)
        if self.domain_adaptation_enabled:
            image_features = projected_features.permute(0, 2, 3, 4, 1).reshape(
                -1, self.hidden_dim
            )
            instance_features = tokens.flatten(start_dim=1)
            image_reversed, image_plain = self.backbone_domain_discriminator(
                image_features, domain_alpha
            )
            instance_reversed, instance_plain = self.instance_domain_discriminator(
                instance_features, domain_alpha
            )
            predictions.update(
                domain_image_log_probs=image_reversed,
                domain_image_plain_log_probs=image_plain,
                domain_instance_log_probs=instance_reversed,
                domain_instance_plain_log_probs=instance_plain,
            )
        return tokens, predictions, projected_features


def build_relationformer(config: Mapping) -> RelationFormer:
    """Build the supported checkpoint-compatible 3D model."""

    from .seresnet import build_seresnet

    return RelationFormer(
        encoder=build_seresnet(config),
        decoder=build_def_detr_transformer(config),
        config=config,
    )


__all__ = ["MLP", "RelationFormer", "build_relationformer"]
