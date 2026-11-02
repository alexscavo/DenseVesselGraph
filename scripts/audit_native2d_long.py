"""Source topology/provenance checks after preparation; never opens test arrays."""
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyvista as pv
from native2d.targets import from_dense_graph
from scripts.prepare_native2d_long import csv_graph,plant_edges,OUT,DATA,write

def betti(nodes,edges):
    parent=list(range(len(nodes)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in np.asarray(edges,int).reshape(-1,2):parent[root(int(a))]=root(int(b))
    components=len({root(i) for i in range(len(nodes))})
    return [components,len(edges)-len(nodes)+components]

def check(nodes,edges):
    graph=from_dense_graph(nodes,edges,provenance={})
    before=betti(nodes,edges);after=betti(graph.nodes,graph.edges)
    assert before==after,(before,after)
    for edge,path in zip(graph.edges,graph.paths):np.testing.assert_allclose(path[[0,-1]],graph.nodes[edge],atol=1e-8,rtol=0)
    return dict(source_nodes=len(nodes),contracted_nodes=len(graph.nodes),betti=before)

def main():
    report={};source=DATA/'octa-synth-packed';seen={};graphs={};rows=[]
    prepared=json.loads((OUT/'octasynth/audit.json').read_text())
    assert len(prepared['alignment'])==128 and all(r['xy_mean_distance']<2 for r in prepared['alignment'])
    for row in csv.DictReader((source/'splits.csv').open()):
        if row['split']=='test':continue
        name=row['patient_id'];parent=name.removesuffix('data.png');xyz,edges=csv_graph(source/'graphs'/parent)
        with (source/'graphs'/parent/'nodes.csv').open() as stream:node_ids=[int(r['id']) for r in csv.DictReader(stream)]
        assert len(set(node_ids))==len(node_ids)
        item=check(xyz[:,[1,0]]*1400,edges);item.update(parent=parent,split=row['split']);rows.append(item)
        for table,digest in [(seen,hashlib.sha256((source/'raw'/name).read_bytes()).hexdigest()),
                             (graphs,hashlib.sha256((source/'graphs'/parent/'nodes.csv').read_bytes()+(source/'graphs'/parent/'edges.csv').read_bytes()).hexdigest())]:
            assert digest not in table or table[digest]==row['split'],'Exact source duplication across training/validation'
            table[digest]=row['split']
    assert len(rows)==128
    report['octasynth']=dict(passed=True,all_128_train_validation_source_topologies=rows,exact_raw_and_graph_cross_split_duplicates=0,test_arrays_opened=0)
    if not (OUT/'plants/manifest.json').exists():
        assert not json.loads((OUT/'plants/readiness.json').read_text())['passed']
        write(OUT/'source-topology-audit.json',dict(passed=True,datasets=report,plant_source_blocked=True));return
    manifest=json.loads((OUT/'plants/manifest.json').read_text());samples=manifest['samples']
    plant_audit=json.loads((OUT/'plants/audit.json').read_text())
    assert all(r['xy_mean_distance']<2 for r in plant_audit['alignment'])
    plant_audit['retained_parents']={s:len({r['subject_id'] for r in samples if r['split']==s}) for s in ['train','validation']}
    write(OUT/'plants/audit.json',plant_audit)
    manifest['split_scope']='official source-image membership; all train/val/test parent IDs disjoint; plant-individual IDs unavailable'
    manifest['source_annotation_example']=dict(path=str(DATA/'plants_3d2cut/graphs/Set00_IMG_3283_annotation.json'),title='3D2Cut Grapevine Annotation',coordinate_system='TopLeft',annotation_tool='semi auto; supplied feature coordinates and ParentID adjacency')
    write(OUT/'plants/manifest.json',manifest)
    selected=np.unique(np.linspace(0,len(samples)-1,256,dtype=int));rows=[]
    for index in selected:
        row=samples[int(index)];mesh=pv.read(row['source_graph']);nodes=np.asarray(mesh.points)[:,:2]*128;edges,repeated=plant_edges(mesh.lines)
        item=check(nodes,edges);item.update(sample_id=row['sample_id'],repeated_identical_segments=repeated);rows.append(item)
    report['plants']=dict(passed=True,deterministic_256_sample_source_topology_check=rows,all_patches_verified_during_preparation=True,test_arrays_opened=0)
    write(OUT/'source-topology-audit.json',dict(passed=True,datasets=report))
    print('Source topology, path endpoints and OCTA cross-split exact-duplicate checks passed',flush=True)

if __name__=='__main__':main()
