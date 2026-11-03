Here is the shortlist of metrics I would study for your journal extension, keeping only those with an estimated relevance ≥ 3/5.

The scores reflect your specific objective: comparing straight-edge graphs against sparse, continuous spline-based vascular graphs, while measuring geometry, topology, and efficiency independently.

## 1. Geometric metrics

| Metric                                       | Relevance | Main idea                                                                                   | Reference paper                                                                                                   |
| -------------------------------------------- | --------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Chamfer Distance ↓                           | 5/5       | Average spatial distance between sampled predicted and GT centerlines.                      | [VesselTok (2026)](https://arxiv.org/abs/2603.18797)                                                              |
| Curve F1 / tolerance-based centerline Dice ↑ | 5/5       | Measures how much predicted and GT centerline geometry matches within a spatial tolerance.  | [Robust Vessel Segmentation and Centerline Extraction (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295992/) |
| Street Mover's Distance (SMD) ↓              | 4/5       | Optimal-transport distance between spatial distributions sampled from graph edges.          | [Belli & Kipf (2019)](https://arxiv.org/abs/1910.14388)                                                           |
| Hausdorff Distance (HD95) ↓                  | 4/5       | Measures large spatial deviations, using the 95th percentile to reduce outlier sensitivity. | [Robust Vessel Segmentation (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295992/)                           |
| Average Symmetric Surface Distance (ASSD) ↓  | 4/5       | Mean bidirectional nearest-neighbor distance between centerlines or surfaces.               | [Robust Vessel Segmentation (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295992/)                           |
| Tortuosity error ↓                           | 5/5       | Compares vessel bending, for example through arc-length/chord-length ratios.                | [CaravelMetrics (2026)](https://arxiv.org/abs/2512.03869)                                                         |
| Curvature error ↓                            | 4/5       | Compares local bending along corresponding vessel branches.                                 | [CaravelMetrics (2026)](https://arxiv.org/abs/2512.03869)                                                         |
| Vessel length error ↓                        | 3/5       | Compares total vessel length or matched branch lengths.                                     | [CaravelMetrics (2026)](https://arxiv.org/abs/2512.03869)                                                         |

Chamfer Distance and SMD are especially worth testing against duplicate parallel edges, given your concern about overprediction.

The CaravelMetrics measures are originally vessel descriptors. Turning them into prediction errors requires comparing their values against GT.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+3



## 2. Topological metrics

| Metric                                 | Relevance | Main idea                                                                                                               | Reference paper                                                                                                                                                              |
| -------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TOPO precision / recall / F1 ↑         | 5/5       | Evaluates local network connectivity through reachable graph neighborhoods.                                             | [RelationFormer (2022)](https://arxiv.org/abs/2203.10202)                                                                                                                    |
| APLS ↑                                 | 5/5       | Compares shortest-path lengths between corresponding graph locations. Broken connections receive substantial penalties. | [SpaceNet (2018)](https://arxiv.org/abs/1807.01232)                                                                                                                          |
| Edge precision / recall / mAP ↑        | 5/5       | Measures whether the correct connections are predicted, including false-positive edges.                                 | [RelationFormer (2022)](https://arxiv.org/abs/2203.10202)                                                                                                                    |
| Junction matching precision / recall ↑ | 5/5       | Matches junctions and evaluates their connections, penalizing missing or incorrect branches.                            | [RoadTracer (2018)](https://dspace.mit.edu/entities/publication/c372afa9-e634-4e09-9683-94fc67c0c1e4)                                                                        |
| clDice ↑                               | 4/5       | Compares centerline-to-mask overlap, emphasizing preservation of tubular structures.                                    | [Shit et al. (2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html) |
| Betti-0 error ↓                        | 4/5       | Difference in the number of connected components.                                                                       | [VesselTok (2026)](https://arxiv.org/abs/2603.18797)                                                                                                                         |
| Betti-1 error ↓                        | 3/5       | Difference in the number of independent cycles or loops.                                                                | [VesselTok (2026)](https://arxiv.org/abs/2603.18797)                                                                                                                         |
| Graph Edit Distance (GED) ↓            | 3/5       | Minimum cost of node/edge edits needed to transform one graph into another.                                             | [Sanfeliu & Fu (1983)](https://doi.org/10.1109/TSMC.1983.6313167)                                                                                                            |
| Connected Road Ratio (CRR) ↑           | 3/5       | Fraction of GT segments recovered without discontinuities; adaptable to vessels.                                        | [DeepRoadMapper (2017)](https://www.cs.toronto.edu/~urtasun/publications/mattyus_etal_iccv17.pdf)                                                                            |

Important: clDice is topology-aware but does not directly verify graph adjacency. Betti errors detect global topological discrepancies but can miss incorrect connections when component and loop counts remain unchanged.

For your project, TOPO, APLS, and junction/edge matching are especially interesting because they evaluate connectivity rather than just geometric overlap.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.dbs.ifi.lmu.de\&sz=32)

dbs.ifi.lmu.de

+5



## 3. Efficiency metrics

These are mostly direct measurements rather than specialized graph-comparison metrics.

| Metric                    | Relevance | Main idea                                                                     | Reference / motivation                               |
| ------------------------- | --------- | ----------------------------------------------------------------------------- | ---------------------------------------------------- |
| Number of graph objects ↓ | 5/5       | Counts nodes, edges, and spline control points needed to represent a network. | [VesselTok (2026)](https://arxiv.org/abs/2603.18797) |
| Representation size ↓     | 5/5       | Bytes or numerical parameters needed to store the graph.                      | [VesselTok (2026)](https://arxiv.org/abs/2603.18797) |
| Inference time ↓          | 4/5       | Measures reconstruction speed under identical hardware conditions.            | [CRESI (2019)](https://arxiv.org/abs/1904.09901)     |
| Peak memory usage ↓       | 4/5       | Measures GPU/CPU memory needed for inference.                                 | [VesselTok (2026)](https://arxiv.org/abs/2603.18797) |

The references motivate these measurements; they do not necessarily introduce or report each one in exactly this form.

I would particularly consider representation size at matched geometric and topological accuracy. Fewer objects are not useful if reconstruction quality deteriorates.

## 4. Detection metrics — additional class

| Metric           | Relevance | Main idea                                             | Reference paper                                                                         |
| ---------------- | --------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Node mAP / mAR ↑ | 4/5       | Evaluates detection and localization of graph nodes.  | [RelationFormer (2022)](https://arxiv.org/abs/2203.10202)                               |
| Node F1 ↑        | 4/5       | Precision–recall balance for correctly matched nodes. | [RelationFormer (2022), related detection evaluation](https://arxiv.org/abs/2203.10202) |

I separate these from topology because correctly locating nodes does not guarantee correct connectivity.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.dbs.ifi.lmu.de\&sz=32)

dbs.ifi.lmu.de

+1



## Which ones would I study first?

Priority shortlist

8 metrics

1

Curve F1 / centerline Dice

Geometric

2

Chamfer Distance

Geometric

3

TOPO F1

Topological

4

APLS

Topological

5

Edge precision / recall

Topological

6

Junction matching

Topological

7

Graph object count

Efficiency

8

Representation size

Efficiency

I would keep SMD, clDice, Betti errors, and tortuosity error as supporting metrics, especially since you already have related implementations.

### About CaravelMetrics

I found the actual paper: An Automated Framework for Large-Scale Graph-Based Cerebrovascular Analysis (Falcetta et al., ISBI 2026).

Its curvature, tortuosity, length, bifurcation, and connectivity measurements are applicable to your graphs, with adaptations. In particular, your spline representation could allow direct curvature and arc-length computation without first fitting another spline to a sampled centerline.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org

+1



One methodological rule I would adopt: evaluate both representations using the same physical geometry and matching conventions, while counting their original representation objects separately. Otherwise, differences in node sampling could be mistaken for differences in reconstruction quality.