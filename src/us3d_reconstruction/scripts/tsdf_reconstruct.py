#!/usr/bin/env python3
"""TSDF fusion 3D reconstruction using Open3D.

Treats each ultrasound frame as a pseudo-depth image and integrates
it into a TSDF volume using the calibrated probe pose.

Usage:
    python3 tsdf_reconstruct.py --scan_dir data/scans/scan_001 \
                                 --calib_dir data/calibration \
                                 --output data/reconstructions/tsdf_001.ply
"""

import argparse
import os
import numpy as np
import cv2
import open3d as o3d
import yaml
import transforms3d.quaternions as tq

from voxel_reconstruct import load_calibration, load_scan_data


def us_to_depth(image, max_depth=0.15):
    """
    Convert ultrasound intensity image to pseudo-depth image.
    Bright regions (high echo) -> closer to surface -> smaller depth.
    """
    normalized = image.astype(np.float32) / 255.0
    depth = max_depth * (1.0 - normalized)
    depth[image < 10] = 0  # mask very dark regions as invalid
    return depth


def build_intrinsic(width, height, pixel_size_x, pixel_size_y):
    """Build pseudo-intrinsic matrix for ultrasound image plane."""
    fx = 1.0 / pixel_size_x
    fy = 1.0 / pixel_size_y
    cx = 0.0
    cy = 0.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    return intrinsic


def tsdf_fusion(frames, poses, T_tool0_probe, T_probe_us,
                pixel_size_x, pixel_size_y, voxel_length=0.001,
                sdf_trunc=0.005, max_depth=0.15):
    """
    Integrate ultrasound frames into TSDF volume.
    """
    h, w = frames[0].shape
    intrinsic = build_intrinsic(w, h, pixel_size_x, pixel_size_y)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    T_tool0_us = T_tool0_probe @ T_probe_us

    for i, (img, T_base_tool0) in enumerate(zip(frames, poses)):
        T_base_us = T_base_tool0 @ T_tool0_us

        depth = us_to_depth(img, max_depth)

        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        color_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        color_o3d = o3d.geometry.Image(color_rgb)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=max_depth,
            convert_rgb_to_intensity=False)

        # Open3D expects camera extrinsic = world_to_camera = inv(T_base_us)
        extrinsic = np.linalg.inv(T_base_us)

        volume.integrate(rgbd, intrinsic, extrinsic)

        if (i + 1) % 50 == 0:
            print("  Integrated %d/%d frames" % (i + 1, len(frames)))

    return volume


def main():
    parser = argparse.ArgumentParser(description='TSDF fusion 3D reconstruction')
    parser.add_argument('--scan_dir', required=True)
    parser.add_argument('--calib_dir', required=True)
    parser.add_argument('--output', required=True, help='Output .ply mesh file')
    parser.add_argument('--voxel_length', type=float, default=0.001)
    parser.add_argument('--sdf_trunc', type=float, default=0.005)
    parser.add_argument('--max_depth', type=float, default=0.15)
    args = parser.parse_args()

    print("Loading calibration...")
    T_tool0_probe, T_probe_us, px_x, px_y, roi = load_calibration(args.calib_dir)
    print("  Pixel size: %.4f x %.4f mm" % (px_x * 1000, px_y * 1000))
    if roi:
        print("  ROI: x=%d y=%d %dx%d" % (roi['x'], roi['y'], roi['w'], roi['h']))

    print("Loading scan data...")
    frames, poses = load_scan_data(args.scan_dir, roi=roi)
    print("  Loaded %d frames" % len(frames))

    print("Running TSDF fusion (voxel=%.2fmm, trunc=%.2fmm)..." % (
        args.voxel_length * 1000, args.sdf_trunc * 1000))
    volume = tsdf_fusion(frames, poses, T_tool0_probe, T_probe_us,
                         px_x, px_y, args.voxel_length, args.sdf_trunc, args.max_depth)

    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    o3d.io.write_triangle_mesh(args.output, mesh)
    print("Saved mesh to %s (%d vertices, %d triangles)" % (
        args.output, len(mesh.vertices), len(mesh.triangles)))

    pcd = volume.extract_point_cloud()
    pcd_path = args.output.replace('.ply', '_pcd.ply')
    o3d.io.write_point_cloud(pcd_path, pcd)
    print("Saved point cloud to %s" % pcd_path)


if __name__ == '__main__':
    main()
