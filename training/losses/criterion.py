"""Baseline graph losses with optional focal/HNM and Betti extensions."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from boxes import box_cxcyczwhd_to_xyxyzz, generalized_box_iou_3d
from models.matcher import build_matcher

from .betti_h0 import h0_betti_matching_loss
from .betti_h1 import cycle_space_matching_loss
from .focal import (
    build_unmatched_relation_pairs,
    linear_progress_schedule,
    scheduled_candidate_weight,
    select_active_unmatched_queries,
    select_hard_unmatched_relation_logits,
    softmax_focal_loss,
)


def _source_indices(assignments, device):
    batches = []
    sources = []
    for batch, (source, _) in enumerate(assignments):
        source = source.to(device=device, dtype=torch.long)
        batches.append(torch.full_like(source, batch))
        sources.append(source)
    if not sources:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.cat(batches), torch.cat(sources)


def _ratio_upsample(logits, labels, ratio, tolerance):
    """Preserve the baseline's deterministic minority-class duplication."""

    positives = logits[labels == 1]
    negatives = logits[labels == 0]
    if positives.shape[0] == 0 or negatives.shape[0] == 0:
        return logits, labels
    actual = positives.shape[0] / float(negatives.shape[0])
    if actual < ratio - tolerance:
        count = int(negatives.shape[0] * ratio - positives.shape[0])
        source, label = positives, 1
    elif ratio + tolerance < actual:
        count = int(positives.shape[0] / ratio - negatives.shape[0])
        source, label = negatives, 0
    else:
        return logits, labels
    if count <= 0:
        return logits, labels
    full_repeats, remainder = divmod(count, int(source.shape[0]))
    parts = []
    if full_repeats:
        parts.append(source.repeat(full_repeats, 1))
    if remainder:
        parts.append(source[:remainder])
    # count is positive, so at least one branch contributes. Avoiding
    # repeat(0, 1) is required by the cluster's legacy RepeatBackward.
    additions = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    new_labels = torch.full(
        (count,), label, dtype=torch.long, device=labels.device
    )
    return torch.cat((logits, additions)), torch.cat((labels, new_labels))


class GraphCriterion(nn.Module):
    """Compute the supported RelationFormer graph objective.

    Cross-entropy with ratio upsampling is the baseline. Focal/HNM and Betti
    terms are opt-in and share no execution path with a disabled baseline.
    """

    def __init__(
        self,
        config: Mapping,
        matcher: nn.Module,
        relation_embed: nn.Module,
        *,
        validation: bool = False,
        spline_head=None,
    ):
        super().__init__()
        self.config = config
        self.matcher = matcher
        self.relation_embed = relation_embed
        self.spline_head = spline_head
        self.dimensions = int(config["data"]["spatial_dims"])
        decoder = config["model"]["decoder"]
        self.object_queries = int(decoder["object_queries"])
        self.relation_tokens = int(decoder["relation_tokens"])

        loss = config["loss"]
        self.enabled_losses = tuple(loss["enabled"])
        self.weights = {key: float(value) for key, value in loss["weights"].items()}
        self.node_config = loss["node"]["classification"]
        self.node_type_config = loss["node_type"]["classification"]
        self.edge_config = loss["edge"]["classification"]
        self.candidates = loss["edge"]["candidates"]
        self.balancing = loss["edge"]["balancing"]
        self.degree = loss["degree"]
        self.topology = config["topology"]
        self.domain_adaptation = config.get("domain_adaptation", {})
        self.validation = bool(validation)
        self.epoch = 1
        self.progress_percent = 0.0

    def set_training_progress(self, epoch: int, progress_percent: float) -> None:
        self.epoch = max(1, int(epoch))
        self.progress_percent = min(100.0, max(0.0, float(progress_percent)))

    def _classification_gamma(self, configuration, progress_percent):
        gamma = float(configuration.get("focal_gamma", 2.0))
        curriculum = configuration.get("curriculum", {})
        if self.validation or not curriculum.get("enabled", False):
            return gamma
        return linear_progress_schedule(
            progress_percent,
            gamma,
            curriculum["start_percent"],
            curriculum["end_percent"],
        )

    def _classification_loss(self, logits, labels, configuration):
        weights = torch.as_tensor(
            configuration["class_weights"],
            dtype=logits.dtype,
            device=logits.device,
        )
        if configuration["name"] == "focal":
            return softmax_focal_loss(
                logits,
                labels,
                class_weights=weights,
                gamma=self._classification_gamma(
                    configuration, self.progress_percent
                ),
                reduction="mean",
            )
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            weight=weights,
            reduction="mean",
        )

    def loss_classification(self, logits, assignments):
        labels = torch.zeros(logits.shape[:2], dtype=torch.long, device=logits.device)
        labels[_source_indices(assignments, logits.device)] = 1
        return self._classification_loss(logits, labels, self.node_config)

    def loss_cardinality(self, logits, assignments):
        target_count = logits.new_tensor(
            [float(len(source)) for source, _ in assignments]
        )
        predicted_count = (logits.argmax(-1) == logits.shape[-1] - 1).sum(1)
        return F.l1_loss(
            predicted_count.float(), target_count, reduction="sum"
        ) / (logits.shape[0] * logits.shape[1])

    def loss_node_types(self, logits, target_nodes, target_edges, assignments):
        """Supervise patch-graph degree classes on Hungarian-matched queries."""
        parts = []
        labels = []
        for batch, (source, target) in enumerate(assignments):
            if not source.numel():
                continue
            count = len(target_nodes[batch])
            degree = torch.zeros(count, device=logits.device, dtype=torch.long)
            edges = torch.as_tensor(target_edges[batch], device=logits.device).long().reshape(-1, 2)
            if edges.numel():
                degree.scatter_add_(0, edges[:, 0], torch.ones_like(edges[:, 0]))
                degree.scatter_add_(0, edges[:, 1], torch.ones_like(edges[:, 1]))
            # 0=isolated, 1=patch terminal, 2=degree-2, 3=junction.
            parts.append(logits[batch, source.to(logits.device)])
            labels.append(degree.clamp(max=3)[target.to(logits.device)])
        if not parts:
            return logits.sum() * 0.0
        return self._classification_loss(
            torch.cat(parts), torch.cat(labels), self.node_type_config
        )

    def loss_domains(self, predictions, domains):
        """Reference image/instance NLL plus prediction consistency."""
        settings = self.domain_adaptation
        domains = domains.long()
        weights = torch.as_tensor(
            settings["class_weights"], device=domains.device, dtype=torch.float32
        )
        losses = []
        if settings["image_adversarial"]:
            image = predictions["domain_image_log_probs"]
            if image.shape[0] % len(domains):
                raise ValueError("image domain predictions must divide into samples")
            positions = image.shape[0] // len(domains)
            image_labels = domains.repeat_interleave(positions)
            losses.append(F.nll_loss(image, image_labels, weight=weights))
        if settings["graph_adversarial"]:
            instance = predictions["domain_instance_log_probs"]
            losses.append(F.nll_loss(instance, domains, weight=weights))
        if settings["consistency_regularization"]:
            image_plain = predictions["domain_image_plain_log_probs"].reshape(
                len(domains), -1, 2
            )
            instance_plain = predictions["domain_instance_plain_log_probs"]
            losses.append(F.mse_loss(image_plain.mean(dim=1), instance_plain))
        return torch.stack(losses).sum()

    def _matched_nodes(self, predicted_nodes, target_nodes, assignments):
        source_index = _source_indices(assignments, predicted_nodes.device)
        predicted = predicted_nodes[source_index]
        target_parts = [
            nodes[target.to(nodes.device)]
            for nodes, (_, target) in zip(target_nodes, assignments)
            if len(target)
        ]
        if not target_parts:
            return predicted, predicted.new_empty((0, self.dimensions))
        return predicted, torch.cat(target_parts, dim=0).to(predicted.device)

    def loss_nodes(self, predicted_nodes, target_nodes, assignments):
        predicted, target = self._matched_nodes(
            predicted_nodes[..., : self.dimensions], target_nodes, assignments
        )
        if target.numel() == 0:
            return predicted_nodes.sum() * 0.0
        return F.l1_loss(predicted, target, reduction="sum") / target.shape[0]

    def loss_boxes(self, predicted_nodes, target_nodes, assignments):
        predicted, centers = self._matched_nodes(
            predicted_nodes, target_nodes, assignments
        )
        if centers.numel() == 0:
            return predicted_nodes.sum() * 0.0
        target_boxes = torch.cat((centers, 0.2 * torch.ones_like(centers)), dim=-1)
        overlap = generalized_box_iou_3d(
            box_cxcyczwhd_to_xyxyzz(predicted),
            box_cxcyczwhd_to_xyxyzz(target_boxes),
        )
        return (1.0 - torch.diag(overlap)).sum() / centers.shape[0]

    def _relation_features(self, object_tokens, relation_tokens, pairs):
        if pairs.numel() == 0:
            width = object_tokens.shape[-1] * (2 + self.relation_tokens)
            return object_tokens.new_empty((0, width))
        parts = (object_tokens[pairs[:, 0]], object_tokens[pairs[:, 1]])
        if self.relation_tokens:
            relation = relation_tokens.reshape(1, -1).expand(pairs.shape[0], -1)
            parts = (*parts, relation)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _local_true_edges(edges, matched_targets, device):
        mapping = {
            int(target): local for local, target in enumerate(matched_targets.tolist())
        }
        result = []
        for left, right in torch.as_tensor(edges).reshape(-1, 2).tolist():
            if int(left) in mapping and int(right) in mapping:
                local_left, local_right = mapping[int(left)], mapping[int(right)]
                if local_left != local_right:
                    result.append((local_left, local_right))
        if not result:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.as_tensor(result, dtype=torch.long, device=device)

    def _matched_edge_pool(self, tokens, target_edges, assignments, pair_contexts=None):
        feature_parts = []
        label_parts = []
        degree_contexts = []
        offset = 0
        object_tokens = tokens[..., : self.object_queries, :]
        relation_tokens = tokens[
            ..., self.object_queries : self.object_queries + self.relation_tokens, :
        ]
        maximum = self.candidates.get("max_per_graph")
        for batch, (source, target) in enumerate(assignments):
            source = source.to(tokens.device)
            target = target.to(tokens.device)
            count = int(source.numel())
            if count < 2:
                continue
            positives = self._local_true_edges(target_edges[batch], target, tokens.device)
            positive_set = {
                tuple(sorted((int(left), int(right))))
                for left, right in positives.tolist()
            }
            all_pairs = torch.combinations(
                torch.arange(count, device=tokens.device), r=2
            )
            negative_mask = torch.as_tensor(
                [tuple(pair) not in positive_set for pair in all_pairs.tolist()],
                dtype=torch.bool,
                device=tokens.device,
            )
            negatives = all_pairs[negative_mask]
            negative_total = int(negatives.shape[0])
            positive_cap = self.candidates.get("positive_cap")
            if positive_cap is not None and positives.shape[0] > int(positive_cap):
                positives = positives[: int(positive_cap)]
            positive_count = positives.shape[0]
            negative_limit = negatives.shape[0]
            if maximum is not None:
                negative_limit = min(negative_limit, max(0, int(maximum) - positive_count))
            if negative_limit < negatives.shape[0]:
                selection = torch.randperm(negatives.shape[0], device=tokens.device)[
                    :negative_limit
                ]
                negatives = negatives[selection]
            positives = positives.clone()
            negatives = negatives.clone()
            for pairs in (positives, negatives):
                if pairs.numel():
                    reverse = torch.rand(pairs.shape[0], device=tokens.device) > 0.5
                    pairs[reverse] = pairs[reverse][:, [1, 0]]
            pairs = torch.cat((positives, negatives), dim=0)
            if pair_contexts is not None:
                pair_contexts.append((batch, source[pairs]))
            matched_tokens = object_tokens[batch, source]
            feature_parts.append(
                self._relation_features(matched_tokens, relation_tokens[batch], pairs)
            )
            label_parts.append(
                torch.cat(
                    (
                        torch.ones(positives.shape[0], dtype=torch.long, device=tokens.device),
                        torch.zeros(negatives.shape[0], dtype=torch.long, device=tokens.device),
                    )
                )
            )
            if self.degree["enabled"]:
                degree_contexts.append(
                    (offset, count, pairs, positive_count, negative_total)
                )
            offset += int(pairs.shape[0])
        return feature_parts, label_parts, degree_contexts

    @staticmethod
    def _degree_consistency_loss(relation_logits, contexts):
        """Match expected node degrees to GT, correcting sampled negatives."""
        losses = []
        for offset, node_count, pairs, positive_count, negative_total in contexts:
            pair_count = int(pairs.shape[0])
            probabilities = relation_logits[offset : offset + pair_count].softmax(-1)[:, 1]
            negative_count = pair_count - positive_count
            if negative_count:
                weights = torch.cat(
                    (
                        probabilities.new_ones(positive_count),
                        probabilities.new_full(
                            (negative_count,), negative_total / negative_count
                        ),
                    )
                )
            else:
                weights = torch.ones_like(probabilities)
            weighted = probabilities * weights
            expected = probabilities.new_zeros(node_count)
            expected = expected.index_add(0, pairs[:, 0], weighted)
            expected = expected.index_add(0, pairs[:, 1], weighted)
            target = probabilities.new_zeros(node_count)
            if positive_count:
                true_pairs = pairs[:positive_count]
                ones = probabilities.new_ones(positive_count)
                target = target.index_add(0, true_pairs[:, 0], ones)
                target = target.index_add(0, true_pairs[:, 1], ones)
            losses.append(F.smooth_l1_loss(expected, target, beta=1.0))
        if not losses:
            return relation_logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _unmatched_hard_negatives(self, tokens, logits, assignments, predictions=None, targets=None):
        if not self.candidates.get("include_unmatched", False):
            return []
        object_tokens = tokens[..., : self.object_queries, :]
        relation_tokens = tokens[
            ..., self.object_queries : self.object_queries + self.relation_tokens, :
        ]
        selected = []
        for batch, (matched, _) in enumerate(assignments):
            _, _, active, _ = select_active_unmatched_queries(
                logits[batch],
                matched,
                self.candidates["unmatched_object_threshold"],
                self.candidates["max_active_unmatched"],
            )
            pairs = build_unmatched_relation_pairs(
                matched.to(tokens.device), active
            )
            if not pairs.numel():
                continue
            forward_features = self._relation_features(
                object_tokens[batch], relation_tokens[batch], pairs
            )
            reverse_features = self._relation_features(
                object_tokens[batch], relation_tokens[batch], pairs[:, [1, 0]]
            )
            # Relations are undirected.  Average the two endpoint orderings at
            # logit level so hard selection and focal supervision see the same
            # symmetric candidate while gradients reach both evaluations.
            pool_logits = 0.5 * (
                self.relation_embed(forward_features)
                + self.relation_embed(reverse_features)
            )
            if self.spline_head is not None and self.spline_head.curve_evidence is not None:
                pool_logits = self._curve_logits(tokens, predictions, targets, batch, pairs)
            hard_logits, _ = select_hard_unmatched_relation_logits(
                pool_logits, self.candidates["max_unmatched_pairs_per_graph"]
            )
            if hard_logits.numel():
                selected.append(hard_logits)
        return selected

    def _curve_logits(self, tokens, predictions, targets, batch, pairs):
        from models.curve_evidence import curve_relation_logits
        return curve_relation_logits(self.relation_embed, self.spline_head,
            tokens[batch, :self.object_queries],
            tokens[batch, self.object_queries:self.object_queries+self.relation_tokens],
            predictions['pred_nodes'][batch, :, :3], pairs,
            targets['spline'][batch]['metric'].to(tokens), predictions['curve_image_features'][batch])

    def loss_edges(self, tokens, node_logits, target_edges, assignments, predictions=None, targets=None):
        pair_contexts = []
        feature_parts, label_parts, degree_contexts = self._matched_edge_pool(
            tokens, target_edges, assignments, pair_contexts
        )
        if feature_parts:
            if self.spline_head is not None and self.spline_head.curve_evidence is not None:
                relation_logits = torch.cat([self._curve_logits(tokens, predictions, targets, batch, pairs)
                                             for batch, pairs in pair_contexts])
            else:
                relation_logits = self.relation_embed(torch.cat(feature_parts, dim=0))
            labels = torch.cat(label_parts, dim=0)
        else:
            relation_logits = tokens.new_empty((0, 2))
            labels = torch.empty(0, dtype=torch.long, device=tokens.device)
        degree_loss = (
            self._degree_consistency_loss(relation_logits, degree_contexts)
            if self.degree["enabled"]
            else relation_logits.sum() * 0.0
        )
        if labels.numel() and self.balancing["mode"] == "ratio_upsample":
            relation_logits, labels = _ratio_upsample(
                relation_logits,
                labels,
                float(self.balancing["positive_to_negative_ratio"]),
                float(self.balancing["tolerance"]),
            )
        matched_loss = (
            self._classification_loss(relation_logits, labels, self.edge_config)
            if labels.numel()
            else tokens.sum() * 0.0
        )
        # Keep relation-head parameters in every DDP backward graph, including
        # the rare batch where no graph contains a valid node pair.
        matched_loss = matched_loss + sum(
            parameter.sum() * 0.0 for parameter in self.relation_embed.parameters()
        )

        hard_chunks = self._unmatched_hard_negatives(tokens, node_logits, assignments, predictions, targets)
        if not hard_chunks:
            return matched_loss, degree_loss
        hard_logits = torch.cat(hard_chunks, dim=0)
        hard_labels = torch.zeros(
            hard_logits.shape[0], dtype=torch.long, device=tokens.device
        )
        hard_loss = self._classification_loss(
            hard_logits, hard_labels, self.edge_config
        )
        weight = float(self.candidates["unmatched_weight"])
        if not self.validation:
            weight = scheduled_candidate_weight(
                self.epoch,
                weight,
                self.candidates["unmatched_warmup_epochs"],
                self.candidates["unmatched_ramp_epochs"],
            )
        matched_count = float(labels.numel())
        hard_count = float(hard_labels.numel())
        edge_loss = (
            matched_loss * matched_count + hard_loss * weight * hard_count
        ) / max(1.0, matched_count + weight * hard_count)
        return edge_loss, degree_loss

    def _symmetric_edge_probabilities(self, object_tokens, relation_tokens, pairs):
        forward = self._relation_features(object_tokens, relation_tokens, pairs)
        reverse = self._relation_features(
            object_tokens, relation_tokens, pairs[:, [1, 0]]
        )
        return 0.5 * (
            self.relation_embed(forward).softmax(-1)[:, 1]
            + self.relation_embed(reverse).softmax(-1)[:, 1]
        )

    @torch.no_grad()
    def _matching_structure(self, tokens, candidate_indices):
        """Score all query pairs for a structure-aware matcher without a graph."""

        batch_size = tokens.shape[0]
        object_tokens = tokens[..., : self.object_queries, :]
        relation_tokens = tokens[
            ..., self.object_queries : self.object_queries + self.relation_tokens, :
        ]
        chunk_size = int(
            self.config["model"]["matcher"].get("pair_chunk_size", 1024)
        )
        structures = []
        for batch in range(batch_size):
            candidates = candidate_indices[batch].to(tokens.device)
            count = int(candidates.numel())
            structure = tokens.new_zeros((count, count))
            pairs = torch.combinations(
                torch.arange(count, device=tokens.device), r=2
            )
            for chunk in pairs.split(chunk_size):
                probabilities = self._symmetric_edge_probabilities(
                    object_tokens[batch, candidates], relation_tokens[batch], chunk
                )
                structure[chunk[:, 0], chunk[:, 1]] = probabilities
                structure[chunk[:, 1], chunk[:, 0]] = probabilities
            structures.append(structure)
        return structures

    def loss_topology(self, tokens, target_edges, assignments):
        zero = tokens.sum() * 0.0
        metrics = {"betti_h0": zero, "betti_h1": zero}
        object_tokens = tokens[..., : self.object_queries, :]
        relation_tokens = tokens[
            ..., self.object_queries : self.object_queries + self.relation_tokens, :
        ]
        for name, minimum_nodes, loss_function in (
            ("betti_h0", 2, h0_betti_matching_loss),
            ("betti_h1", 3, cycle_space_matching_loss),
        ):
            configuration = self.topology[name]
            if not configuration["enabled"]:
                continue
            sample_losses = []
            for batch, (source, target) in enumerate(assignments):
                source = source.to(tokens.device)
                target = target.to(tokens.device)
                count = int(source.numel())
                if count < minimum_nodes:
                    continue
                pairs = torch.combinations(
                    torch.arange(count, device=tokens.device), r=2
                )
                probabilities = self._symmetric_edge_probabilities(
                    object_tokens[batch, source], relation_tokens[batch], pairs
                )
                truth = self._local_true_edges(
                    target_edges[batch], target, tokens.device
                )
                keywords = dict(
                    num_vertices=count,
                    diagonal_factor=float(configuration["diagonal_factor"]),
                    normalize=bool(configuration["normalize"]),
                )
                if name == "betti_h0":
                    keywords["unmatched_weight"] = float(
                        configuration["unmatched_weight"]
                    )
                else:
                    keywords["false_positive_weight"] = float(
                        configuration["false_positive_weight"]
                    )
                    keywords["false_negative_weight"] = float(
                        configuration["false_negative_weight"]
                    )
                sample_loss, _ = loss_function(probabilities, pairs, truth, **keywords)
                if torch.isfinite(sample_loss):
                    sample_losses.append(sample_loss)
            if sample_losses:
                metrics[name] = torch.stack(sample_losses).mean()
        return metrics

    def loss_spline(self, tokens, predictions, targets, assignments):
        from models.spline import canonical_reverse
        if self.spline_head is None or 'spline' not in targets:
            raise ValueError('Spline loss requires the spline head and full-path target cache')
        errors, control_errors, curve_errors, endpoint_errors, branch_weights = [], [], [], [], []
        for batch, (source, target) in enumerate(assignments):
            mapping = dict(zip(target.tolist(), source.tolist()))
            target_edges = targets['edges'][batch]
            selected = [i for i,(a,b) in enumerate(target_edges.tolist()) if a in mapping and b in mapping]
            if not selected:
                continue
            pairs = torch.tensor([[mapping[a],mapping[b]] for a,b in target_edges[selected].tolist()], device=tokens.device)
            geometry = targets['spline'][batch]
            metric = geometry['metric'].to(tokens)
            endpoints = predictions['pred_nodes'][batch, pairs, :3]
            reverse = canonical_reverse(endpoints @ metric.T)
            pairs[reverse] = pairs[reverse][:, [1,0]]
            endpoints = predictions['pred_nodes'][batch, pairs, :3]
            relation = tokens[batch, self.object_queries:self.object_queries+self.relation_tokens]
            features = self._relation_features(tokens[batch, :self.object_queries], relation, pairs)
            local = ({'local_features': predictions['spline_local_features'][batch],
                      'input_shape': predictions['spline_input_shape']}
                     if self.spline_head.local_geometry is not None else {})
            controls, curves = self.spline_head(features, endpoints, metric, **local)
            target_controls = geometry['controls'][selected].clone()
            target_curves = geometry['curves'][selected].clone()
            target_controls[reverse] = target_controls[reverse].flip(1)
            target_curves[reverse] = target_curves[reverse].flip(1)
            dc = (controls[:, 1:-1]-target_controls[:, 1:-1]) @ metric.T
            dx = (curves-target_curves) @ metric.T
            mode = self.config['loss']['spline']['mode']
            delta = dc if mode=='controls' else dx
            if self.config['loss']['spline'].get('target', 'absolute') == 'residual':
                # Replace absolute-control error with error relative to each
                # graph's own chord. Greville coordinates reproduce B(t)=t.
                g = self.spline_head.greville[None, 1:-1, None]
                endpoint_delta = (endpoints-target_controls[:, [0,-1]]) @ metric.T
                delta = dc - ((1-g)*endpoint_delta[:, :1] + g*endpoint_delta[:, 1:])
            errors.append(F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=float(self.config['loss']['spline'].get('beta_mm',1.)), reduction='none').mean((1,2)))
            branch_weights.append(geometry.get('branch_weights', torch.ones(len(target_edges), device=tokens.device))[selected].to(tokens))
            control_errors.append(dc.norm(dim=-1).mean(1))
            curve_errors.append(dx.norm(dim=-1).mean(1))
            endpoint_errors.append(((endpoints-target_controls[:, [0,-1]]) @ metric.T).norm(dim=-1).mean(1))
        # Empty graphs still touch every head parameter, compatible with DDP.
        zero = sum(p.sum()*0. for p in self.spline_head.parameters()) + predictions['pred_nodes'].sum()*0.
        mean = lambda parts: torch.cat(parts).mean() if parts else zero
        weighted = (torch.cat(errors)*torch.cat(branch_weights)).sum()/torch.cat(branch_weights).sum() if errors else zero
        return {'spline': weighted, 'spline_control_mm': mean(control_errors),
                'spline_curve_mm': mean(curve_errors), 'spline_endpoint_mm': mean(endpoint_errors),
                'spline_positive_edges': tokens.new_tensor(sum(len(p) for p in errors))}

    def forward(self, tokens, predictions, targets):
        overflowing = [
            (index, int(len(nodes)))
            for index, nodes in enumerate(targets["nodes"])
            if len(nodes) > self.object_queries
        ]
        if overflowing:
            raise ValueError(
                "ground-truth graph exceeds decoder object-query capacity: "
                f"capacity={self.object_queries}, samples={overflowing}. "
                "Increase model.decoder.object_queries; silently dropping graph "
                "nodes would corrupt node and edge supervision."
            )
        candidate_indices = None
        predicted_structure = None
        if self.matcher.requires_structure:
            candidate_indices = self.matcher.matching_candidates(
                predictions, targets
            )
            predicted_structure = self._matching_structure(tokens, candidate_indices)
        assignments = self.matcher(
            predictions,
            targets,
            predicted_structure=predicted_structure,
            candidate_indices=candidate_indices,
        )
        edge_loss, degree_loss = self.loss_edges(
            tokens, predictions["pred_logits"], targets["edges"], assignments, predictions, targets
        )
        losses = {
            "class": self.loss_classification(predictions["pred_logits"], assignments),
            "nodes": self.loss_nodes(predictions["pred_nodes"], targets["nodes"], assignments),
            "boxes": self.loss_boxes(predictions["pred_nodes"], targets["nodes"], assignments),
            "cardinality": self.loss_cardinality(predictions["pred_logits"], assignments),
            "edges": edge_loss,
            "degree": degree_loss,
        }
        if "node_types" in self.enabled_losses:
            if "pred_node_types" not in predictions:
                raise ValueError("node_types loss requires model.node_type.enabled=true")
            losses["node_types"] = self.loss_node_types(
                predictions["pred_node_types"], targets["nodes"],
                targets["edges"], assignments,
            )
        if 'spline' in self.enabled_losses:
            losses.update(self.loss_spline(tokens, predictions, targets, assignments))
        if "domain" in self.enabled_losses:
            if "domains" not in targets:
                raise ValueError("domain loss requires batch domain labels")
            losses["domain"] = self.loss_domains(predictions, targets["domains"])
            losses["domain_weighted"] = losses["domain"] * self.weights["domain"]
        topology_losses = self.loss_topology(tokens, targets["edges"], assignments)
        for name, value in topology_losses.items():
            configuration = self.topology[name]
            weight = float(configuration["weight"])
            if not self.validation:
                weight = scheduled_candidate_weight(
                    self.epoch,
                    weight,
                    configuration["warmup_epochs"],
                    configuration["ramp_epochs"],
                )
            losses[name] = value
            losses[name + "_weighted"] = value * weight

        losses["total"] = sum(
            losses[name] * self.weights[name]
            for name in self.enabled_losses
        )
        if self.degree["enabled"]:
            degree_weight = float(self.degree["weight"])
            if not self.validation:
                degree_weight = scheduled_candidate_weight(
                    self.epoch,
                    degree_weight,
                    self.degree["warmup_epochs"],
                    self.degree["ramp_epochs"],
                )
            losses["degree_weighted"] = degree_loss * degree_weight
            losses["total"] = losses["total"] + losses["degree_weighted"]
        for name in ("betti_h0", "betti_h1"):
            if self.topology[name]["enabled"] and not self.topology[name]["log_only"]:
                losses["total"] = losses["total"] + losses[name + "_weighted"]
        return losses


def build_criterion(
    config: Mapping, model: nn.Module, *, validation: bool = False
) -> GraphCriterion:
    return GraphCriterion(
        config,
        build_matcher(config),
        model.relation_embed,
        validation=validation,
        spline_head=getattr(model, 'spline_head', None),
    )


__all__ = ["GraphCriterion", "build_criterion"]
