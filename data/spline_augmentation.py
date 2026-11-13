"""Exact affine targets for quarter-turn image augmentation (index/size coordinates)."""
import torch
from data.augmentations import _rotate_voxel_coordinates_once


def rotation_affine(turns, shape, *, dtype, device):
    """Return x' = x @ A.T + b, including the voxel-centre offset."""
    shape = tuple(int(s) for s in shape)
    probes = torch.cat((torch.zeros(1, 3, dtype=dtype, device=device),
                        torch.eye(3, dtype=dtype, device=device)))
    points = probes * torch.tensor(shape, dtype=dtype, device=device)
    current_shape = shape
    for k, (a, b) in zip(turns, ((0, 1), (0, 2), (1, 2))):
        points, current_shape = _rotate_voxel_coordinates_once(points, current_shape, k, a, b)
    points = points / torch.tensor(current_shape, dtype=dtype, device=device)
    offset = points[0]
    return (points[1:] - offset).T, offset


def rotate_spline_geometry(geometry, turns, shape):
    """Transform all target points without clipping controls outside the image.

    M' = M A^-1 retains physical distances in the original physical basis.
    Absolute physical positions may differ by a common translation; all losses
    use differences. Endpoints and residual controls share this same mapping.
    """
    controls = geometry['controls']
    matrix, offset = rotation_affine(turns, shape, dtype=controls.dtype, device=controls.device)
    result = {key: value @ matrix.T + offset for key, value in geometry.items()
              if key in ('controls', 'curves', 'guides', 'full_paths')}
    metric = geometry['metric']
    result['metric'] = metric @ torch.linalg.inv(matrix.to(metric))
    return result


def flip_spline_geometry(geometry, axes, shape):
    """Targets for flipping the image along the axes marked True: x' = x A^T + b with A = diag(+-1) and
    b = (size-1)/size on flipped axes (voxel i -> size-1-i, normalized coordinates); M' = M A^-1 as for the quarter turns."""
    controls = geometry['controls']
    matrix = torch.diag(controls.new_tensor([-1.0 if f else 1.0 for f in axes]))
    offset = controls.new_tensor([(int(s) - 1) / int(s) if f else 0.0 for f, s in zip(axes, shape)])
    result = {key: value @ matrix.T + offset for key, value in geometry.items()
              if key in ('controls', 'curves', 'guides', 'full_paths')}
    metric = geometry['metric']
    result['metric'] = metric @ torch.linalg.inv(matrix.to(metric))
    return result


def project_plant_geometry(geometry, parameters, inner_size, size, padding, voxel_mm):
    """PNG(y,x,0) -> projected/augmented/padded volume; virtual units, not plant mm.

    Keep the physical basis in original PNG axis order to preserve cached edge
    orientation. Controls may legitimately lie outside the volume: never clamp.
    """
    if parameters.zoom_factor != 1.0:
        raise ValueError('Plant spline supervision does not support zoom')
    controls = geometry['controls']
    swap = controls.new_tensor([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    matrix, offset = rotation_affine(parameters.quarter_turns, (inner_size,)*3,
                                    dtype=controls.dtype, device=controls.device)
    for axis, flip in enumerate(parameters.flip_axes):
        if flip:
            matrix[axis] *= -1
            offset[axis] = (inner_size-1)/inner_size - offset[axis]
    scale = inner_size / size
    result = {}
    for key, value in geometry.items():
        if key in ('controls', 'curves', 'guides', 'full_paths'):
            projected = value @ swap.T + controls.new_tensor([0.,0.,.5])
            result[key] = (projected @ matrix.T + offset)*scale + padding/size
    # Before augmentation: inner_size*voxel_mm in original y,x,z basis.
    result['metric'] = (inner_size*voxel_mm*swap) @ torch.linalg.inv(matrix*scale)
    return result
