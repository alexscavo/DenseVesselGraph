"""Supported model construction API."""


def build_model(config):
    """Build native 2D when explicitly requested; preserve the 3D default."""

    if int(config['data'].get('spatial_dims', 3)) == 2:
        from native2d import build_model as build_native_2d
        return build_native_2d(config)

    from .relationformer import build_relationformer

    return build_relationformer(config)


__all__ = ["build_model"]
