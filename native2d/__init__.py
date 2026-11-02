"""Native 2D RelationFormer; importing this package does not construct 3D models."""

def build_model(config):
    from .model import RelationFormer2D
    return RelationFormer2D(config)
