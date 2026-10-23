"""Recover *outer* H corners from a complete segmentation, never from truth.

Only sufficiently large, untruncated convex hulls are eligible. The projected
three-bar H must agree with the whole mask, including its concavities. Heading
comes from the causal pose prior; a symmetric H alone cannot provide it.
"""
import cv2
import numpy as np
from navigation.vision.associate_keypoints import associate_keypoints


def outer_quad(mask, minimum_side=32, margin=3, diagnostics=None, adaptive=False):
    diagnostics = {} if diagnostics is None else diagnostics
    def reject(reason):
        diagnostics["rejection"] = reason
        return None
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if mask.ndim != 2 or not mask.size:
        return reject("invalid_mask")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return reject("empty_mask")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    quad = cv2.approxPolyDP(hull, .025 * cv2.arcLength(hull, True), True)
    diagnostics["hull_vertices"] = len(quad)
    # A small bevel may add one vertex. Simplify only that bounded case; never
    # substitute an arbitrary rectangle for an incomplete H silhouette.
    if adaptive and len(quad) == 5:
        candidate = cv2.approxPolyDP(hull, .035 * cv2.arcLength(hull, True), True)
        if len(candidate) == 4:
            quad = candidate
            diagnostics["hull_simplified"] = True
    if len(quad) != 4:
        return reject("hull_vertices")
    q = quad.reshape(4, 2).astype(float) + .5
    h, w = mask.shape
    if (q < margin).any() or (q >= [w-margin, h-margin]).any():
        return reject("truncated_mask")
    shortest = float(np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1).min())
    diagnostics.update(minimum_side_px=shortest, required_side_px=float(minimum_side),
                       noise_multiplier=max(1., 32/max(shortest, 1.)))
    if shortest < minimum_side:
        return reject("minimum_side")
    # Fit full outer edges, rather than picking an eroded/antialiased tip pixel.
    boundary = contour.reshape(-1, 2).astype(float) + .5
    lines = []
    center = q.mean(0)
    diagnostics["edge_support_pixels"] = []
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        edge = b-a
        length = np.linalg.norm(edge)
        direction = edge / length
        normal = np.array([-direction[1], direction[0]])
        near = (np.abs((boundary-a) @ normal) < 1.5)
        along = (boundary-a) @ direction
        near &= (along > 1) & (along < length-1)
        diagnostics["edge_support_pixels"].append(int(near.sum()))
        required = max(6, min(8, int(np.ceil(length*.25)))) if adaptive else 8
        if near.sum() < required:
            return reject("edge_support")
        vx, vy, x, y = cv2.fitLine(boundary[near].astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        n = np.array([-vy, vx], float)
        if n @ (center-[x,y]) > 0:
            n *= -1
        # Boundary pixels are inside the binary surface by roughly half a pixel.
        lines.append((n, float(n @ [x,y] + .5)))
    corners = []
    for i in range(4):
        n1, d1 = lines[i-1]
        n2, d2 = lines[i]
        A = np.array([n1, n2])
        if abs(np.linalg.det(A)) < .1:
            return reject("parallel_edges")
        corners.append(np.linalg.solve(A, [d1, d2]))
    corners = np.array(corners)
    if adaptive and np.max(np.linalg.norm(corners-q, axis=1)) > 3:
        return reject("corner_extrapolation")
    if (corners < margin).any() or (corners >= [w-margin, h-margin]).any():
        return reject("truncated_fit")
    diagnostics.pop("rejection", None)
    return corners


def silhouette_iou(mask, ordered_quad, stroke_fraction=(.2, .2)):
    canonical = np.float32([[0,0], [1,0], [1,1], [0,1]])
    H = cv2.getPerspectiveTransform(canonical, np.asarray(ordered_quad, np.float32))
    sx, sy = stroke_fraction
    boxes = [(0, 0, sx, 1), (1-sx, 0, 1, 1), (0, .5-sy/2, 1, .5+sy/2)]
    rendered = np.zeros(np.shape(mask), np.uint8)
    for x0,y0,x1,y1 in boxes:
        p = cv2.perspectiveTransform(np.float32([[[x0,y0], [x1,y0], [x1,y1], [x0,y1]]]), H)[0]
        # Subpixel raster coordinates: array indices refer to pixel centres.
        cv2.fillConvexPoly(rendered, np.rint((p-.5)*16).astype(np.int32), 1, shift=4)
    truth = np.asarray(mask) > 0
    return float((truth & (rendered > 0)).sum() / max(1, (truth | (rendered > 0)).sum()))


def recover_from_mask(detection, image_shape, marker_points, calibration, prior, config, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    mask = detection.network_mask
    if mask is None:
        diagnostics["rejection"] = "no_network_mask"
        return None
    quad = outer_quad(mask, config.get("mask_minimum_side_px", 32), diagnostics=diagnostics,
                      adaptive=config.get("mask_adaptive_edges", False))
    if quad is None:
        return None
    scale, origin = detection.mask_coordinates(image_shape)
    # First restrict orientation with actual H silhouette, then use the same
    # IPPE dual-solution + heading prior resolver as the learned landmarks.
    candidates = []
    max_iou = 0.
    passed_iou = passed_pnp = False
    for flip in (False, True):
        for k in range(4):
            q = np.roll(quad[::-1] if flip else quad, k, axis=0)
            iou = silhouette_iou(mask, q, config.get("marker_stroke_fraction", [.2, .2]))
            max_iou = max(max_iou, iou)
            if iou < config.get("mask_min_iou", .82):
                continue
            passed_iou = True
            match = associate_keypoints(marker_points, q * scale + origin, calibration, prior, ordered=True, refine=config.get("refine_pnp", True))
            if match is not None:
                passed_pnp = True
                from scipy.spatial.transform import Rotation
                T, error, pixels = match
                angle = np.linalg.norm(Rotation.from_matrix(prior[:3,:3].T @ T[:3,:3]).as_rotvec())
                if angle <= config.get("mask_max_prior_angle_rad", .6):
                    score = error + np.linalg.norm(T[:3,3]-prior[:3,3])/10 + angle/.5
                    candidates.append((score, pixels, iou, scale))
    if not candidates:
        diagnostics.update(max_iou=max_iou, rejection="silhouette_iou" if not passed_iou else
                           "pnp_failed" if not passed_pnp else "prior_angle")
        return None
    _, pixels, iou, scale = min(candidates, key=lambda r: r[0])
    difference = np.linalg.norm((pixels-detection.keypoints) / scale, axis=1).max()
    return pixels, iou, float(difference), scale
