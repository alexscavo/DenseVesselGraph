"""Legacy 2D architecture, modern graph supervision; no implicit pretrained download."""
from pathlib import Path
from types import SimpleNamespace
import copy
import yaml


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value


def load_config(path=None, *, smoke=False):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'configs/base.yaml').read_text())
    # This is an isolated 2D entry point. The 3D config validator is unchanged.
    cfg['data'].update(spatial_dims=2, image_size=[128,128], batch_size=2)
    cfg['model']['encoder'].update(backbone='resnet101', init_weights=None, dilation=False)
    cfg['model']['decoder'].update(hidden_dim=512, attention_heads=8, encoder_layers=3,
        decoder_layers=3, feature_levels=4, object_queries=80, relation_tokens=5,
        feedforward_dim=1024, dropout=0.1, relation_attention=False)
    cfg['loss']['node']['classification'].update(name='focal', class_weights=[1.,1.], focal_gamma=2.)
    cfg['loss']['edge']['classification'].update(name='cross_entropy', class_weights=[1.,1.])
    cfg['loss']['enabled'].append('spline')
    cfg['loss']['weights']['spline'] = 1.4739989612791453
    cfg['loss']['spline'] = dict(target='residual', mode='controls', beta_distance=1.)
    cfg['model']['node_refinement'] = dict(enabled=False, radius_pixels=4., image_mode='full')
    cfg['model']['spline'] = dict(enabled=True, controls=8, samples=64, gradient_route='graph_only')
    cfg['evaluation'].update(node_threshold=.5, edge_threshold=.5)
    if smoke:
        cfg['model']['encoder']['backbone']='resnet18'
        cfg['model']['decoder'].update(hidden_dim=64, attention_heads=4, encoder_layers=1,
            decoder_layers=1, feature_levels=3, object_queries=16, relation_tokens=2,
            feedforward_dim=128, dropout=0.)
        cfg['data']['image_size']=[64,64]
    if path:
        def merge(a,b):
            for k,v in b.items():
                if isinstance(v,dict) and isinstance(a.get(k),dict): merge(a[k],v)
                else: a[k]=copy.deepcopy(v)
        merge(cfg, yaml.safe_load(Path(path).read_text()))
    if cfg['data']['spatial_dims'] != 2: raise ValueError('native2d requires spatial_dims=2')
    if cfg['model']['spline']['gradient_route'] != 'graph_only':
        raise ValueError('This port supports the detached graph_only contract')
    return cfg


def legacy_architecture(cfg):
    e,d=cfg['model']['encoder'],cfg['model']['decoder']
    return namespace(dict(MODEL=dict(NUM_CLASSES=2,
        ENCODER=dict(BACKBONE=e['backbone'], BACKBONE_INIT_WEIGHTS=e['init_weights'], HIDDEN_DIM=d['hidden_dim'],
            DILATION=e['dilation'], MASKS=False, NUM_FEATURE_LEVELS=d['feature_levels'], POSITION_EMBEDDING='sine'),
        DECODER=dict(HIDDEN_DIM=d['hidden_dim'], NHEADS=d['attention_heads'], ENC_LAYERS=d['encoder_layers'],
            DEC_LAYERS=d['decoder_layers'], DIM_FEEDFORWARD=d['feedforward_dim'], DROPOUT=d['dropout'],
            ACTIVATION=d['activation'], NUM_FEATURE_LEVELS=d['feature_levels'], RLN_ATTN=False)),
        TRAIN=dict(TRAIN_ENCODER=True,TRAIN_DECODER=True,TRAIN_CNN_BACKBONE=True)))
