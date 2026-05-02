#!/usr/bin/env python3
"""Volume visualization tool.

Displays 3D ultrasound reconstruction results using Open3D and
Matplotlib slice views.

Usage:
    python3 visualize_volume.py --volume data/reconstructions/vol_001.npy
    python3 visualize_volume.py --mesh data/reconstructions/tsdf_001.ply
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def show_slices(volume, axis=2, n_slices=9):
    """Display evenly spaced slices through the volume."""
    n = volume.shape[axis]
    indices = np.linspace(0, n - 1, n_slices).astype(int)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Volume slices (axis=%d)' % axis, fontsize=14)

    for idx, ax in zip(indices, axes.ravel()):
        if axis == 0:
            sl = volume[idx, :, :]
        elif axis == 1:
            sl = volume[:, idx, :]
        else:
            sl = volume[:, :, idx]

        ax.imshow(sl.T, cmap='gray', origin='lower')
        ax.set_title('slice %d' % idx)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_3d_volume(volume, threshold=30):
    """Render non-zero voxels as an Open3D point cloud (grayscale).

    Each voxel above the intensity threshold is shown as a discrete
    POINT (not a filled cube), which makes the cloud look "discontinuous"
    when zoomed in. For a continuous SURFACE, see show_3d_isosurface.
    """
    coords = np.argwhere(volume > threshold)
    if len(coords) == 0:
        print("No voxels above threshold %d" % threshold)
        return

    values = volume[volume > threshold]
    values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(float))

    gray = np.column_stack([values_norm, values_norm, values_norm])
    pcd.colors = o3d.utility.Vector3dVector(gray)

    print("3D point-cloud view: %d points above threshold %d" %
          (len(coords), threshold))
    o3d.visualization.draw_geometries([pcd],
                                       window_name='US Volume (points)',
                                       width=1024, height=768)


def show_mip(volume):
    """Maximum Intensity Projection (MIP) along all 3 axes.

    For each axis, take np.max along that axis to get a 2D image.
    Shows the brightest features as if you're looking through the
    volume — like a crude X-ray. No new dependencies.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Maximum Intensity Projection', fontsize=14)

    for ax_idx, label in enumerate(['axis 0 (X)', 'axis 1 (Y)', 'axis 2 (Z)']):
        mip = np.max(volume, axis=ax_idx)
        axes[ax_idx].imshow(mip.T, cmap='gray', origin='lower')
        axes[ax_idx].set_title('MIP %s\n(viewed from %s direction)' %
                               (label, label.split()[0]))
        axes[ax_idx].axis('off')

    plt.tight_layout()
    plt.show()


def _detect_scan_dir_from_volume(volume_path):
    """Try to find the matching scan_dir for a reconstructed volume.

    Strategy: look for a sibling scan directory with metadata.csv and
    compute the dominant motion direction from the first/last positions.
    """
    import os
    import csv as _csv

    # Try to find scan dir from common locations
    base_dir = os.path.dirname(os.path.abspath(volume_path))
    repo_root = os.path.dirname(os.path.dirname(base_dir))
    candidates = []
    for root in [repo_root, os.path.expanduser('~/joe/us3dscan')]:
        scans_dir = os.path.join(root, 'data', 'scans')
        if os.path.isdir(scans_dir):
            for d in sorted(os.listdir(scans_dir), reverse=True):
                full = os.path.join(scans_dir, d, 'metadata.csv')
                if os.path.exists(full):
                    candidates.append(full)
                    break  # newest only

    for csv_path in candidates:
        try:
            xs, ys = [], []
            with open(csv_path) as f:
                for row in _csv.DictReader(f):
                    xs.append(float(row['px']))
                    ys.append(float(row['py']))
            if len(xs) < 5:
                continue
            n = len(xs) // 4
            head = np.array([np.mean(xs[:n]), np.mean(ys[:n])])
            tail = np.array([np.mean(xs[-n:]), np.mean(ys[-n:])])
            dxy = tail - head
            norm = np.linalg.norm(dxy)
            if norm < 0.01:
                continue
            scan_dir = np.array([dxy[0] / norm, dxy[1] / norm, 0.0])
            print("Detected scan from %s: dir=(%.3f, %.3f)" %
                  (csv_path, scan_dir[0], scan_dir[1]))
            return scan_dir
        except Exception:
            continue
    return None


def auto_rotation_from_scan(scan_dir):
    """Compute rotation angle (deg, around Z) needed to align scan
    direction with the +X axis.

    scan_dir: 3-vector in base frame (x, y, z components).
    Returns: angle in degrees to rotate around Z axis.
    """
    return float(np.degrees(np.arctan2(scan_dir[1], scan_dir[0])))


def rotate_volume_z(volume, angle_deg):
    """Rotate the volume around its Z axis by `angle_deg` degrees.
    Uses bilinear interpolation. Output volume may be larger than
    input due to rotation expansion.
    """
    if abs(angle_deg) < 0.1:
        return volume
    try:
        from scipy.ndimage import rotate
    except ImportError:
        print("scipy not available, skipping rotation")
        return volume
    # rotate in the XY plane (axes 0 and 1), reshape=True to fit
    # the rotated content
    rotated = rotate(volume, angle=-angle_deg, axes=(0, 1),
                     reshape=True, order=1, mode='constant', cval=0.0)
    print("Volume rotated %+.1f° around Z: shape %s → %s" %
          (-angle_deg, volume.shape, rotated.shape))
    return rotated


def show_napari(volume, bounds=None):
    """Interactive 3D volume viewer using napari.

    Far better than Open3D point cloud / iso-surface for ultrasound:
      - Volume rendering (translucent, MIP, ISO, attenuated_mip)
      - Three orthogonal slice planes (MPR view)
      - Interactive sliders to step through slices
      - Each "slice plane" displays the actual 2D ultrasound image
        — stacked along whichever axis you choose

    For scan-aligned slicing (so the slices look like the original
    2D ultrasound B-mode), set rendering='plane' in the layer
    controls and drag the plane to be perpendicular to the scan
    direction. Or pre-rotate with --rotate_z (see main).

    Install: pip install napari[all]
    """
    try:
        import napari
    except ImportError:
        print("napari not installed. Install with:")
        print("    pip install 'napari[all]'")
        return

    # napari expects (Z, Y, X) order; our volume is (X, Y, Z)
    vol_napari = np.transpose(volume, (2, 1, 0))

    # Apply axis flips if requested (napari Y axis is image convention,
    # +Y down on screen, which is opposite to base_link +Y left).
    if getattr(show_napari, '_flip_y', False):
        vol_napari = np.flip(vol_napari, axis=1)
    if getattr(show_napari, '_flip_z', False):
        vol_napari = np.flip(vol_napari, axis=0)
    if getattr(show_napari, '_flip_x', False):
        vol_napari = np.flip(vol_napari, axis=2)

    # Optional Gaussian smoothing for display only (doesn't change file)
    smooth_sigma = getattr(show_napari, '_smooth_sigma', 0.0)
    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            vol_napari = gaussian_filter(
                vol_napari.astype(np.float32), sigma=smooth_sigma)
            print(f"Display smoothing: Gaussian sigma={smooth_sigma:.1f} voxels")
        except ImportError:
            print("scipy missing, skipping display smooth")

    # Voxel size from bounds (for proper aspect ratio)
    if bounds is not None:
        vsize = (bounds[1] - bounds[0]) / np.array(volume.shape)
        # napari (Z, Y, X) order
        scale = (vsize[2], vsize[1], vsize[0])
    else:
        scale = (1.0, 1.0, 1.0)

    viewer = napari.Viewer(title='US Volume')
    viewer.add_image(
        vol_napari,
        name='ultrasound',
        scale=scale,
        rendering='attenuated_mip',     # most US-like rendering
        contrast_limits=[15, 200],      # like B-mode display window
        colormap='gray',
        interpolation3d='linear',
    )
    # Set 2D-view interpolation too (this is the "smooth or pixelated"
    # toggle when scrolling through 2D slices)
    try:
        viewer.layers[0].interpolation = 'bicubic'
    except Exception:
        pass
    print("napari viewer opened. Try the rendering dropdown (top-left):")
    print("  - 'attenuated_mip': brightest features, attenuated by depth (US-like)")
    print("  - 'mip':            maximum intensity projection (X-ray-like)")
    print("  - 'translucent':    transparent volume (good for inner structure)")
    print("  - 'iso':            iso-surface (similar to Open3D iso)")
    print("Use the bottom slider to scrub through slices.")
    print("Press 2 / 3 to toggle 2D / 3D view.")
    napari.run()


def show_3d_isosurface(volume, isolevel=80, smooth=2):
    """Extract a continuous iso-surface (mesh) at the given intensity
    level using marching cubes. Far prettier than point-cloud view —
    you see actual surfaces of bright structures (organ boundaries,
    tissue interfaces).

    isolevel : intensity at which to draw the surface (try 50-150).
    smooth   : number of Taubin smoothing iterations (0 = sharp).
    """
    try:
        from skimage import measure
    except ImportError:
        print("scikit-image not installed (pip install scikit-image) — "
              "falling back to point cloud")
        return show_3d_volume(volume, threshold=isolevel)

    if volume.max() < isolevel:
        print("Volume max=%d < isolevel=%d, lower the isolevel" %
              (volume.max(), isolevel))
        return

    print("Extracting iso-surface at level=%d (this may take 5-30s)..."
          % isolevel)
    verts, faces, normals, _ = measure.marching_cubes(
        volume, level=isolevel, step_size=1)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    if smooth > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth)
        mesh.compute_vertex_normals()

    mesh.paint_uniform_color([0.9, 0.85, 0.7])  # warm beige (tissue-like)
    print("Iso-surface mesh: %d vertices, %d triangles" %
          (len(mesh.vertices), len(mesh.triangles)))
    o3d.visualization.draw_geometries([mesh],
                                       window_name='US Volume (iso-surface)',
                                       width=1024, height=768,
                                       mesh_show_back_face=True)


def show_mesh(mesh_path):
    """Display a PLY mesh from TSDF reconstruction."""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    print("Mesh: %d vertices, %d triangles" % (len(mesh.vertices), len(mesh.triangles)))
    o3d.visualization.draw_geometries([mesh],
                                       window_name='US 3D Reconstruction',
                                       width=1024, height=768)


def export_nifti(volume, bounds, output_path):
    """Export volume as NIfTI (.nii) format if nibabel is available."""
    try:
        import nibabel as nib
        voxel_size = (bounds[1] - bounds[0]) / np.array(volume.shape)
        affine = np.diag([*voxel_size, 1.0])
        affine[:3, 3] = bounds[0]
        nii = nib.Nifti1Image(volume, affine)
        nib.save(nii, output_path)
        print("Exported NIfTI to %s" % output_path)
    except ImportError:
        print("nibabel not installed, skipping NIfTI export")


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D ultrasound reconstruction')
    parser.add_argument('--volume', help='Path to volume .npy file')
    parser.add_argument('--mesh', help='Path to mesh .ply file')
    parser.add_argument('--threshold', type=float, default=30,
                        help='Intensity threshold for 3D point-cloud view')
    parser.add_argument('--axis', type=int, default=2, choices=[0, 1, 2],
                        help='Slice axis')
    parser.add_argument('--mode', default='napari',
                        choices=['napari', 'points', 'isosurface',
                                 'slices', 'mip'],
                        help='Visualization mode: '
                             'napari (interactive 3D, recommended), '
                             'points (Open3D cloud), '
                             'isosurface (Open3D marching-cubes mesh), '
                             'slices (matplotlib 2D grid), '
                             'mip (max-intensity projection)')
    parser.add_argument('--rotate_z', type=float, default=0.0,
                        help='Rotate volume around Z by N degrees BEFORE '
                             'display. Use this to align the scan path with '
                             'an axis so napari slices look like 2D B-mode. '
                             'Typical: 30-60° (computed as '
                             'atan2(scan_dir.y, scan_dir.x))')
    parser.add_argument('--align_to_scan', action='store_true',
                        help='Auto-detect scan direction from metadata.csv '
                             '(needs --scan_dir or sibling scan_*) and '
                             'rotate volume so scan_dir aligns with +X.')
    parser.add_argument('--flip_y', action='store_true',
                        help='Flip the Y axis when displaying in napari '
                             '(napari uses image convention +Y down, '
                             'base_link uses robot convention +Y left).')
    parser.add_argument('--flip_z', action='store_true',
                        help='Flip the Z axis (slice scrub direction) '
                             'in napari display.')
    parser.add_argument('--flip_x', action='store_true',
                        help='Flip the X axis in napari display.')
    parser.add_argument('--smooth_display', type=float, default=0.0,
                        help='Apply Gaussian smoothing (sigma in voxels) '
                             'BEFORE display. Reduces blocky voxel edges. '
                             'Try 0.5-1.5 for typical 0.7mm voxels. Display only,'
                             ' does not modify the saved volume.')
    parser.add_argument('--isolevel', type=float, default=80,
                        help='Iso-surface intensity (50-150 typical)')
    parser.add_argument('--smooth', type=int, default=2,
                        help='Taubin smoothing iterations for iso-surface')
    parser.add_argument('--no_slices', action='store_true',
                        help='Skip the 2D slice grid')
    parser.add_argument('--export_nii', help='Export as NIfTI to this path')
    args = parser.parse_args()

    if args.mesh:
        show_mesh(args.mesh)
        return

    if args.volume:
        volume = np.load(args.volume)
        print("Volume shape: %s, range: [%.1f, %.1f]" % (
            volume.shape, volume.min(), volume.max()))

        # Optional Z-axis rotation (align scan direction with +X)
        if args.rotate_z != 0.0:
            volume = rotate_volume_z(volume, args.rotate_z)
        elif args.align_to_scan:
            # Auto-detect from metadata.csv next to volume
            scan_dir = _detect_scan_dir_from_volume(args.volume)
            if scan_dir is not None:
                angle = auto_rotation_from_scan(scan_dir)
                print("Auto-rotation: scan_dir=(%.3f, %.3f) → "
                      "rotating volume by %+.1f°" %
                      (scan_dir[0], scan_dir[1], -angle))
                volume = rotate_volume_z(volume, angle)

        # Always show slice grid first unless suppressed
        if not args.no_slices and args.mode != 'napari':
            show_slices(volume, axis=args.axis)

        bounds = None
        bounds_path = args.volume.replace('.npy', '_bounds.npy')
        if os.path.exists(bounds_path):
            bounds = np.load(bounds_path)

        if args.mode == 'napari':
            # Pass flip/smooth options via function attributes
            show_napari._flip_x = args.flip_x
            show_napari._flip_y = args.flip_y
            show_napari._flip_z = args.flip_z
            show_napari._smooth_sigma = args.smooth_display
            show_napari(volume, bounds=bounds)
        elif args.mode == 'isosurface':
            show_3d_isosurface(volume, isolevel=args.isolevel,
                               smooth=args.smooth)
        elif args.mode == 'points':
            show_3d_volume(volume, threshold=args.threshold)
        elif args.mode == 'mip':
            show_mip(volume)
        # mode == 'slices' → just slices, no 3D

        if args.export_nii:
            bounds_path = args.volume.replace('.npy', '_bounds.npy')
            if os.path.exists(bounds_path):
                bounds = np.load(bounds_path)
                export_nifti(volume, bounds, args.export_nii)
            else:
                print("Bounds file not found, cannot export NIfTI")
        return

    parser.print_help()


if __name__ == '__main__':
    main()
