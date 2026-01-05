# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random


def flip(img):
    """
    Reverse the channel order of an image (e.g., BGR <-> RGB).
    Note: This is NOT a horizontal flip. For horizontal flip, use img[:, ::-1, :].
    """
    return img[:, :, ::-1].copy()


def transform_preds(coords, center, scale, output_size):
    """
    Transform predicted coordinates from network output space back to the original image space.

    Args:
        coords (ndarray): shape [N, ...], where coords[:, 0:2] are (x, y) points in output space.
        center (ndarray): center used in affine transform.
        scale (float or ndarray): scale used in affine transform.
        output_size (list/tuple): output resolution used in affine transform.

    Returns:
        ndarray: transformed coordinates in the original image coordinate system.
    """
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    """
    Compute a 2x3 affine transformation matrix given center/scale/rotation.

    This transform is commonly used to:
      - crop a region around "center" with size "scale"
      - optionally rotate by "rot"
      - warp the region into the fixed "output_size"

    Args:
        center (ndarray): (x, y) center in the source image.
        scale (float or ndarray/list): scaling factor(s). If float, treated as isotropic scale.
        rot (float): rotation angle in degrees.
        output_size (list/tuple): (dst_w, dst_h) of the target image.
        shift (ndarray): shift factor applied to center.
        inv (int): if 1, compute inverse transform (dst->src); else src->dst.

    Returns:
        ndarray: 2x3 affine matrix.
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180.0

    # Direction vectors defining the source and destination coordinate frames
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    # Three points define an affine transform uniquely
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    # The third point is chosen so that (p0, p1, p2) forms a right-handed coordinate system
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    # Compute affine transformation matrix
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """
    Apply a 2x3 affine transform to a point (x, y).
    """
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    """
    Given points a and b, compute a third point that forms a perpendicular direction from segment (a->b).
    This is used to construct an affine transform from 3 point correspondences.
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """
    Rotate a 2D point around the origin by rot_rad radians.
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def crop(img, center, scale, output_size, rot=0):
    """
    Crop and warp an image to the target output size with optional rotation.
    """
    trans = get_affine_transform(center, scale, rot, output_size)
    dst_img = cv2.warpAffine(
        img,
        trans,
        (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )
    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
    """
    Compute a Gaussian radius for heatmap generation following CornerNet/CenterNet style.

    Args:
        det_size (tuple): (height, width) of the object (or circle diameter-like box).
        min_overlap (float): required minimum overlap.

    Returns:
        float: radius value.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    """
    Generate a 2D Gaussian kernel.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draw a 2D Gaussian on a heatmap at the given center with the given radius.
    This is the "umich" implementation used widely in CenterNet.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[
        radius - top:radius + bottom,
        radius - left:radius + right
    ]

    # Only apply if valid region exists
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    """
    Draw dense regression targets around the center, masked by the Gaussian heatmap region.

    Args:
        regmap (ndarray): regression map, shape [dim, H, W].
        heatmap (ndarray): heatmap, shape [H, W].
        center (tuple/list): center point (x, y).
        value (array-like): regression target values with length = dim.
        radius (int): Gaussian radius.
        is_offset (bool): if True and dim==2, apply offset correction relative to center.

    Returns:
        ndarray: updated regmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]

    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value

    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[
        radius - top:radius + bottom,
        radius - left:radius + right
    ]
    masked_reg = reg[
        :,
        radius - top:radius + bottom,
        radius - left:radius + right
    ]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1]
        )
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg

    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    """
    Draw a Gaussian on heatmap using the MSRA implementation.
    """
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)

    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

    # If the gaussian is completely outside the image, skip
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)

    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    )
    return heatmap


def grayscale(image):
    """
    Convert BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    """
    PCA-based lighting noise, used in ImageNet-style color augmentation.
    """
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    """
    Blend image1 and image2 with weight alpha.
    """
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    """
    Randomly perturb image saturation.
    """
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    """
    Randomly perturb image brightness.
    """
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    """
    Randomly perturb image contrast.
    """
    alpha = 1.0 + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    """
    Apply random color augmentation: brightness, contrast, saturation, and lighting noise.
    """
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()

    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)

    lighting_(data_rng, image, 0.1, eig_val, eig_vec)
