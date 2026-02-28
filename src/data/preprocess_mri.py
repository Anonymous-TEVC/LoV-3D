#!/usr/bin/env python3
"""
MRI Preprocessing Pipeline for LoV-3D.

Converts raw T1w MRI (DICOM or NIfTI) to analysis-ready NIfTI through a
standardized 6-step pipeline. Output is NOT resized — downstream encoders
apply their own target resolution (e.g., 128³ for ResNet-50).

Pipeline:
  Step 1: DICOM → NIfTI               (dcm2niix)
  Step 2: Reorient → RAS+             (nibabel)
  Step 3: N4 Bias Field Correction    (ANTsPy, Tustison et al. 2010)
  Step 4: Skull Stripping             (HD-BET / ANTsPy fallback)
  Step 5: Affine Registration → MNI152 (ANTsPy)
  Step 6: Z-score Intensity Normalization (within brain mask, clip [-3, +3])

Output naming: {subject_id}_{visit_date}_T1w_final.nii.gz

Usage:
  # Full pipeline from DICOM
  python src/data/preprocess_mri.py \\
      --input_dir /path/to/dicoms \\
      --output_dir /path/to/output \\
      --mode dicom

  # Full pipeline from NIfTI (already converted)
  python src/data/preprocess_mri.py \\
      --input_dir /path/to/niftis \\
      --output_dir /path/to/output \\
      --mode nifti

  # Single steps (e.g., skull strip only)
  python src/data/preprocess_mri.py \\
      --input_dir /path/to/bias_corrected \\
      --output_dir /path/to/output \\
      --start_step 4 --end_step 4

  # Batch from manifest CSV
  python src/data/preprocess_mri.py \\
      --manifest /path/to/manifest.csv \\
      --output_dir /path/to/output

Dependencies:
  pip install antspyx nibabel tqdm
  pip install hd-bet  (optional, for GPU skull stripping)
  conda install -c conda-forge dcm2niix  (for DICOM conversion)

References:
  - N4ITK: Tustison et al. (2010). N4ITK: improved N3 bias correction.
    IEEE Trans Med Imaging, 29(6):1310-1320.
  - HD-BET: Isensee et al. (2019). Automated brain extraction of
    multisequence MRI using artificial neural networks. Human Brain Mapping.
  - MNI152: Fonov et al. (2011). Unbiased average age-appropriate atlases
    for pediatric studies. NeuroImage, 54(1):313-327.
"""

import os
import sys
import csv
import json
import glob
import shutil
import subprocess
import argparse
import re
import time
from pathlib import Path
from datetime import timedelta

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation
from tqdm import tqdm


# ============================================================================
# Step 1: DICOM → NIfTI
# ============================================================================

def step1_dcm2nifti(dcm_dir, output_path, overwrite=False):
    """Convert a DICOM series to NIfTI using dcm2niix.

    Args:
        dcm_dir: path to directory containing .dcm files
        output_path: desired output .nii.gz path
        overwrite: if True, re-convert even if output exists

    Returns:
        (status, output_path or error_message)
    """
    if os.path.exists(output_path) and not overwrite:
        return 'skipped', output_path

    out_dir = os.path.dirname(output_path)
    out_prefix = os.path.basename(output_path).replace('.nii.gz', '')
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        'dcm2niix',
        '-z', 'y',        # compress output
        '-f', out_prefix,  # output filename
        '-o', out_dir,     # output directory
        '-b', 'y',         # create BIDS sidecar JSON
        '-w', '1',         # overwrite existing
        dcm_dir
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return 'error', f"dcm2niix error: {result.stderr}"

        # dcm2niix may create multiple files — pick the largest (full 3D volume)
        nii_files = glob.glob(os.path.join(out_dir, f"{out_prefix}*.nii.gz"))
        if len(nii_files) == 0:
            return 'error', 'No NIfTI file created'

        if len(nii_files) > 1:
            nii_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            best = nii_files[0]
            for f in nii_files[1:]:
                os.remove(f)
                json_f = f.replace('.nii.gz', '.json')
                if os.path.exists(json_f):
                    os.remove(json_f)
            if best != output_path:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(best, output_path)
                json_best = best.replace('.nii.gz', '.json')
                json_out = output_path.replace('.nii.gz', '.json')
                if os.path.exists(json_best):
                    if os.path.exists(json_out):
                        os.remove(json_out)
                    os.rename(json_best, json_out)
        elif nii_files[0] != output_path:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(nii_files[0], output_path)

        if os.path.exists(output_path):
            return 'success', output_path
        else:
            return 'error', 'Output file not found after conversion'

    except subprocess.TimeoutExpired:
        return 'error', 'Timeout (300s)'
    except FileNotFoundError:
        return 'error', 'dcm2niix not found — install with: conda install -c conda-forge dcm2niix'


# ============================================================================
# Step 2: Reorient to RAS+
# ============================================================================

def step2_reorient(input_path, output_path, overwrite=False):
    """Reorient a NIfTI image to RAS+ standard orientation.

    Eliminates scanner-specific orientation differences so all images
    share a consistent Left-Right / Anterior-Posterior / Inferior-Superior
    axis mapping.

    Returns:
        (status, original_orientation)
    """
    if os.path.exists(output_path) and not overwrite:
        return 'skipped', None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = nib.load(input_path)

    current_axcodes = nib.aff2axcodes(img.affine)
    if current_axcodes == ('R', 'A', 'S'):
        nib.save(img, output_path)
        return 'already_ras', current_axcodes

    current_ornt = io_orientation(img.affine)
    target_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(current_ornt, target_ornt)
    reoriented = img.as_reoriented(transform)

    new_axcodes = nib.aff2axcodes(reoriented.affine)
    assert new_axcodes == ('R', 'A', 'S'), f"Reorientation failed: {new_axcodes}"

    nib.save(reoriented, output_path)
    return 'reoriented', current_axcodes


# ============================================================================
# Step 3: N4 Bias Field Correction
# ============================================================================

def step3_bias_correction(input_path, output_path, overwrite=False):
    """N4 Bias Field Correction using ANTsPy.

    Corrects intensity inhomogeneity caused by RF coil non-uniformity.
    Uses the N4ITK algorithm (Tustison et al., 2010).

    Parameters:
        shrink_factor=4: speed up computation (subsample for fitting)
        convergence: 4 levels × 50 iterations, tolerance 1e-7
        spline_param=200: B-spline control point spacing (mm)

    Returns:
        (status, message)
    """
    if os.path.exists(output_path) and not overwrite:
        return 'skipped', None

    import ants

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = ants.image_read(input_path)

    # Rough mask via Otsu thresholding to focus correction on tissue
    mask = ants.get_mask(img, low_thresh=img.numpy().mean() * 0.1)

    corrected = ants.n4_bias_field_correction(
        img,
        mask=mask,
        shrink_factor=4,
        convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7},
        spline_param=200,
        return_bias_field=False,
        verbose=False,
    )

    ants.image_write(corrected, output_path)
    return 'success', None


# ============================================================================
# Step 4: Skull Stripping (Brain Extraction)
# ============================================================================

def step4_skull_strip(input_path, brain_path, mask_path,
                      method='hdbet', device='0', overwrite=False):
    """Brain extraction using HD-BET (preferred) or ANTsPy (fallback).

    HD-BET (Isensee et al., 2019) uses a nnU-Net architecture trained on
    >2000 brain MRIs. Robust to pathology (atrophy, lesions).

    Args:
        input_path: bias-corrected NIfTI
        brain_path: output brain-extracted NIfTI
        mask_path: output binary brain mask NIfTI
        method: 'hdbet' or 'antspy'
        device: GPU device for HD-BET ('0', '1', ... or 'cpu')

    Returns:
        (status, message)
    """
    if os.path.exists(brain_path) and os.path.exists(mask_path) and not overwrite:
        return 'skipped', None

    os.makedirs(os.path.dirname(brain_path), exist_ok=True)

    if method == 'hdbet':
        try:
            return _skull_strip_hdbet(input_path, brain_path, mask_path, device)
        except Exception as e:
            print(f"  HD-BET failed ({e}), falling back to ANTsPy...")
            return _skull_strip_antspy(input_path, brain_path, mask_path)
    else:
        return _skull_strip_antspy(input_path, brain_path, mask_path)


def _skull_strip_hdbet(input_path, brain_path, mask_path, device='0'):
    """HD-BET brain extraction."""
    # HD-BET v2.0 uses 'cuda'/'cpu'/'mps' instead of device number
    if device.isdigit():
        device = 'cuda'
    cmd = [
        'hd-bet',
        '-i', input_path,
        '-o', brain_path,
        '-device', device,
        '--disable_tta',
        '--save_bet_mask',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"HD-BET failed: {result.stderr}")

    # HD-BET naming: {brain_path} (brain) and {brain_path}_bet.nii.gz (mask)
    for mask_suffix in ['_bet.nii.gz', '_mask.nii.gz']:
        hdbet_mask = brain_path.replace('.nii.gz', mask_suffix)
        if os.path.exists(hdbet_mask) and hdbet_mask != mask_path:
            os.rename(hdbet_mask, mask_path)
            break
    else:
        # Create mask from brain image
        brain_img = nib.load(brain_path)
        brain_data = brain_img.get_fdata()
        mask_data = (brain_data > 0).astype(np.float32)
        mask_img = nib.Nifti1Image(mask_data, brain_img.affine, brain_img.header)
        nib.save(mask_img, mask_path)

    return 'success_hdbet', None


def _skull_strip_antspy(input_path, brain_path, mask_path):
    """ANTsPy brain extraction (fallback using Atropos-based masking)."""
    import ants

    img = ants.image_read(input_path)

    # Use get_mask (Otsu + morphological ops) for rough brain mask
    mask = ants.get_mask(img)
    brain = ants.mask_image(img, mask)

    ants.image_write(brain, brain_path)
    ants.image_write(mask, mask_path)

    return 'success_antspy', None


# ============================================================================
# Step 5: Affine Registration to MNI152
# ============================================================================

def step5_registration(input_path, output_path, transform_path=None,
                       overwrite=False):
    """Affine registration to MNI152 standard space using ANTsPy.

    AFFINE-ONLY registration preserves brain atrophy patterns, which are
    the discriminative features for AD classification. Non-linear (SyN)
    registration would warp away the very signal we want to detect.

    Template: MNI152NLin2009cAsym (from ANTsPy built-in).

    Args:
        input_path: brain-extracted NIfTI
        output_path: output registered NIfTI
        transform_path: optional path to save affine matrix (.mat)

    Returns:
        (status, registration_result)
    """
    if os.path.exists(output_path) and not overwrite:
        return 'skipped', None

    import ants

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load MNI152 template
    template = ants.image_read(ants.get_ants_data('mni'))
    moving = ants.image_read(input_path)

    # Affine registration (rigid + affine stages)
    reg = ants.registration(
        fixed=template,
        moving=moving,
        type_of_transform='Affine',
        aff_metric='mattes',
        aff_sampling=32,
        verbose=False,
    )

    ants.image_write(reg['warpedmovout'], output_path)

    # Save transform matrix if requested
    if transform_path:
        fwd_transforms = reg['fwdtransforms']
        if fwd_transforms:
            shutil.copy2(fwd_transforms[0], transform_path)

    return 'success', reg


# ============================================================================
# Step 6: Z-score Intensity Normalization
# ============================================================================

def step6_intensity_normalize(registered_path, output_path,
                              clip_range=(-3.0, 3.0), overwrite=False):
    """Z-score normalization within brain mask, clipped to [-3, +3].

    Formula: x' = clip((x - mu_brain) / sigma_brain, -3, +3)

    Background voxels (non-brain) are set to 0. Statistics are computed
    ONLY within the brain mask (non-zero voxels after registration).

    Args:
        registered_path: MNI-registered NIfTI
        output_path: output normalized NIfTI
        clip_range: (min, max) after z-score, default [-3, +3]

    Returns:
        (status, normalization_stats)
    """
    if os.path.exists(output_path) and not overwrite:
        return 'skipped', None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = nib.load(registered_path)
    data = img.get_fdata().astype(np.float32)

    # Brain mask: non-zero voxels in the registered (skull-stripped) image
    mask = data > 0
    brain_voxels = data[mask]

    if len(brain_voxels) == 0:
        raise ValueError("No brain voxels found (all zeros)")

    mu = np.mean(brain_voxels)
    sigma = np.std(brain_voxels)

    if sigma < 1e-8:
        raise ValueError(f"Standard deviation too small: {sigma:.2e}")

    # Z-score normalize within brain mask
    normalized = np.zeros_like(data)
    normalized[mask] = (data[mask] - mu) / sigma

    # Clip to remove outliers
    normalized[mask] = np.clip(normalized[mask], clip_range[0], clip_range[1])

    # Ensure RAS+ orientation
    out_img = nib.Nifti1Image(normalized, img.affine, img.header)
    current_ornt = io_orientation(out_img.affine)
    target_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(current_ornt, target_ornt)
    out_img = out_img.as_reoriented(transform)

    nib.save(out_img, output_path)

    stats = {
        'original_mean': float(mu),
        'original_std': float(sigma),
        'brain_voxels': int(np.sum(mask)),
        'total_voxels': int(data.size),
        'brain_fraction': float(np.sum(mask) / data.size),
        'normalized_min': float(normalized[mask].min()),
        'normalized_max': float(normalized[mask].max()),
    }
    return 'success', stats


# ============================================================================
# Quality Control
# ============================================================================

def qc_check(nii_path):
    """Basic quality control checks on a NIfTI file.

    Returns dict with shape, spacing, orientation, intensity stats, and flags.
    """
    img = nib.load(nii_path)
    data = img.get_fdata()
    shape = img.shape
    spacing = img.header.get_zooms()[:3]
    orientation = nib.aff2axcodes(img.affine)

    has_nan = bool(np.any(np.isnan(data)))
    has_inf = bool(np.any(np.isinf(data)))

    nonzero = data[data != 0]
    intensity_stats = None
    if len(nonzero) > 0:
        intensity_stats = {
            'mean': float(np.mean(nonzero)),
            'std': float(np.std(nonzero)),
            'min': float(np.min(nonzero)),
            'max': float(np.max(nonzero)),
            'p01': float(np.percentile(nonzero, 1)),
            'p99': float(np.percentile(nonzero, 99)),
        }

    flags = []
    if has_nan:
        flags.append('has_nan')
    if has_inf:
        flags.append('has_inf')
    if len(nonzero) == 0:
        flags.append('all_zeros')
    if len(nonzero) < 100000:
        flags.append('low_brain_voxels')
    if orientation != ('R', 'A', 'S'):
        flags.append(f'non_ras_orientation:{"".join(orientation)}')

    return {
        'shape': list(shape),
        'spacing': [float(s) for s in spacing],
        'orientation': ''.join(orientation),
        'has_nan': has_nan,
        'has_inf': has_inf,
        'nonzero_voxels': int(np.sum(data != 0)),
        'total_voxels': int(data.size),
        'intensity_stats': intensity_stats,
        'flags': flags,
    }


# ============================================================================
# Full Pipeline
# ============================================================================

def preprocess_single(subject_id, visit_date, input_path, output_dir,
                      mode='nifti', start_step=1, end_step=6,
                      skull_strip_method='hdbet', device='0',
                      clip_range=(-3.0, 3.0), overwrite=False):
    """Run the full preprocessing pipeline for a single scan.

    Args:
        subject_id: subject identifier (e.g., '002_S_0295')
        visit_date: visit date string (e.g., '2011-06-02')
        input_path: path to raw DICOM directory or NIfTI file
        output_dir: root output directory (substeps stored in subdirs)
        mode: 'dicom' or 'nifti'
        start_step: first step to run (1-6)
        end_step: last step to run (1-6)
        skull_strip_method: 'hdbet' or 'antspy'
        device: GPU device for HD-BET
        clip_range: intensity clipping range for step 6
        overwrite: re-process existing files

    Returns:
        dict with status per step and final output path
    """
    base = f"{subject_id}_{visit_date}"
    out = Path(output_dir)
    result = {'subject_id': subject_id, 'visit_date': visit_date, 'steps': {}}

    # Define intermediate paths
    paths = {
        'nifti':     str(out / '01_nifti'          / subject_id / f"{base}_T1w.nii.gz"),
        'reoriented': str(out / '02_reoriented'     / subject_id / f"{base}_T1w_reoriented.nii.gz"),
        'bias':      str(out / '03_bias_corrected'  / subject_id / f"{base}_T1w_n4.nii.gz"),
        'brain':     str(out / '04_brain_extracted'  / subject_id / f"{base}_T1w_brain.nii.gz"),
        'mask':      str(out / '04_brain_extracted'  / subject_id / f"{base}_T1w_brain_mask.nii.gz"),
        'registered': str(out / '05_registered'      / subject_id / f"{base}_T1w_mni_affine.nii.gz"),
        'transform': str(out / '05_registered'      / subject_id / f"{base}_T1w_affine.mat"),
        'final':     str(out / '06_normalized'       / subject_id / f"{base}_T1w_final.nii.gz"),
    }

    try:
        # Step 1: DICOM → NIfTI
        if start_step <= 1 <= end_step:
            if mode == 'dicom':
                status, msg = step1_dcm2nifti(input_path, paths['nifti'], overwrite)
            else:
                # Input is already NIfTI — copy (compress .nii → .nii.gz if needed)
                os.makedirs(os.path.dirname(paths['nifti']), exist_ok=True)
                if not os.path.exists(paths['nifti']) or overwrite:
                    if input_path.endswith('.nii') and not input_path.endswith('.nii.gz'):
                        # Uncompressed .nii → load and save as .nii.gz
                        img = nib.load(input_path)
                        nib.save(img, paths['nifti'])
                    else:
                        shutil.copy2(input_path, paths['nifti'])
                status = 'copied'
            result['steps']['step1'] = status
        else:
            # If starting from a later step, input_path should point to
            # the appropriate intermediate file
            if start_step == 2:
                paths['nifti'] = input_path
            elif start_step == 3:
                paths['reoriented'] = input_path
            elif start_step == 4:
                paths['bias'] = input_path
            elif start_step == 5:
                paths['brain'] = input_path
            elif start_step == 6:
                paths['registered'] = input_path

        # Step 2: Reorient
        if start_step <= 2 <= end_step:
            status, ornt = step2_reorient(paths['nifti'], paths['reoriented'], overwrite)
            result['steps']['step2'] = status
            if ornt:
                result['original_orientation'] = ''.join(ornt)

        # Step 3: Bias correction
        if start_step <= 3 <= end_step:
            status, _ = step3_bias_correction(paths['reoriented'], paths['bias'], overwrite)
            result['steps']['step3'] = status

        # Step 4: Skull stripping
        if start_step <= 4 <= end_step:
            status, _ = step4_skull_strip(
                paths['bias'], paths['brain'], paths['mask'],
                method=skull_strip_method, device=device, overwrite=overwrite,
            )
            result['steps']['step4'] = status

        # Step 5: Registration
        if start_step <= 5 <= end_step:
            status, _ = step5_registration(
                paths['brain'], paths['registered'], paths['transform'], overwrite,
            )
            result['steps']['step5'] = status

        # Step 6: Normalization
        if start_step <= 6 <= end_step:
            status, norm_stats = step6_intensity_normalize(
                paths['registered'], paths['final'], clip_range, overwrite,
            )
            result['steps']['step6'] = status
            if norm_stats:
                result['norm_stats'] = norm_stats

        result['output_path'] = paths['final']
        result['status'] = 'success'

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)

    return result


# ============================================================================
# CLI
# ============================================================================

def infer_subject_visit(filename):
    """Infer subject_id and visit_date from ADNI-style filename.

    Patterns recognized:
      - {subject_id}_{visit_date}_T1w*.nii.gz
      - ADNI_{subject_id}_MR_*
      - {XXX_S_XXXX}_{YYYY-MM-DD}_*
    """
    stem = Path(filename).name
    for ext in ['.nii.gz', '.nii']:
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break

    m = re.match(r'(\d{3}_S_\d+)_(\d{4}-\d{2}-\d{2})', stem)
    if m:
        return m.group(1), m.group(2)

    m = re.match(r'ADNI_(\d{3}_S_\d+)', stem)
    if m:
        return m.group(1), 'unknown'

    return stem, 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description='LoV-3D MRI Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  DICOM → NIfTI           (requires dcm2niix)
  2  Reorient → RAS+         (nibabel)
  3  N4 Bias Field Correction (ANTsPy)
  4  Skull Stripping          (HD-BET / ANTsPy)
  5  Affine → MNI152          (ANTsPy)
  6  Z-score Normalization    (clip [-3, +3])

Examples:
  # From DICOM directory structure
  python src/data/preprocess_mri.py --input_dir /data/ADNI --output_dir /data/preprocessed --mode dicom

  # From NIfTI files
  python src/data/preprocess_mri.py --input_dir /data/niftis --output_dir /data/preprocessed --mode nifti

  # Single file
  python src/data/preprocess_mri.py --input /path/to/T1w.nii.gz --output_dir /data/preprocessed \\
      --subject_id 002_S_0295 --visit_date 2011-06-02

  # Resume from step 4 (skull stripping)
  python src/data/preprocess_mri.py --input_dir /data/preprocessed/03_bias_corrected \\
      --output_dir /data/preprocessed --start_step 4
"""
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                             help='Single input file (DICOM dir or NIfTI)')
    input_group.add_argument('--input_dir', type=str,
                             help='Directory of NIfTI files or DICOM subject dirs')
    input_group.add_argument('--manifest', type=str,
                             help='CSV manifest (columns: subject_id, visit_date, '
                                  'input_path)')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root output directory')
    parser.add_argument('--mode', type=str, default='nifti',
                        choices=['dicom', 'nifti'],
                        help='Input format (default: nifti)')
    parser.add_argument('--subject_id', type=str,
                        help='Subject ID (for single file mode)')
    parser.add_argument('--visit_date', type=str,
                        help='Visit date (for single file mode)')
    parser.add_argument('--start_step', type=int, default=1,
                        choices=[1, 2, 3, 4, 5, 6],
                        help='Start from this step (default: 1)')
    parser.add_argument('--end_step', type=int, default=6,
                        choices=[1, 2, 3, 4, 5, 6],
                        help='End at this step (default: 6)')
    parser.add_argument('--skull_strip', type=str, default='hdbet',
                        choices=['hdbet', 'antspy'],
                        help='Skull stripping method (default: hdbet)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device for HD-BET (default: 0)')
    parser.add_argument('--clip_min', type=float, default=-3.0,
                        help='Min clip after z-score (default: -3.0)')
    parser.add_argument('--clip_max', type=float, default=3.0,
                        help='Max clip after z-score (default: 3.0)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-process existing files')
    parser.add_argument('--qc', action='store_true',
                        help='Run QC checks on final output')
    args = parser.parse_args()

    clip_range = (args.clip_min, args.clip_max)

    print("=" * 60)
    print("LoV-3D MRI Preprocessing Pipeline")
    print(f"  Steps: {args.start_step} → {args.end_step}")
    print(f"  Mode: {args.mode}")
    print(f"  Skull stripping: {args.skull_strip}")
    print(f"  Clip range: [{args.clip_min}, {args.clip_max}]")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)

    # Build task list
    tasks = []

    if args.input:
        # Single file mode
        sid = args.subject_id or infer_subject_visit(args.input)[0]
        vdate = args.visit_date or infer_subject_visit(args.input)[1]
        tasks.append((sid, vdate, args.input))

    elif args.manifest:
        # CSV manifest mode
        with open(args.manifest) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row['subject_id']
                vdate = row['visit_date']
                ipath = row.get('input_path') or row.get('nifti_path') or row.get('dcm_dir')
                if ipath and os.path.exists(ipath):
                    tasks.append((sid, vdate, ipath))
                else:
                    print(f"  WARNING: {ipath} not found, skipping {sid}_{vdate}")

    elif args.input_dir:
        # Directory scan mode
        input_dir = Path(args.input_dir)
        if args.mode == 'nifti':
            nii_files = sorted(
                list(input_dir.rglob('*.nii.gz')) + list(input_dir.rglob('*.nii'))
            )
            for f in nii_files:
                sid, vdate = infer_subject_visit(f.name)
                tasks.append((sid, vdate, str(f)))
        else:
            # DICOM: each subject dir contains scan type / date / image dirs
            for subj_dir in sorted(input_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                sid = subj_dir.name
                if not re.match(r'\d{3}_S_\d+', sid):
                    continue
                # Find DICOM directories (look for .dcm files)
                for dcm_dir in sorted(subj_dir.rglob('*.dcm')):
                    parent = dcm_dir.parent
                    # Extract date from path
                    parts = str(parent).split('/')
                    vdate = 'unknown'
                    for p in parts:
                        m = re.match(r'(\d{4}-\d{2}-\d{2})', p)
                        if m:
                            vdate = m.group(1)
                            break
                    tasks.append((sid, vdate, str(parent)))
                    break  # one per subject for now

    print(f"\nProcessing {len(tasks)} scans...")
    start_time = time.time()

    results = []
    stats = {'success': 0, 'error': 0, 'skipped': 0}

    for i, (sid, vdate, ipath) in enumerate(tqdm(tasks, desc="Preprocessing")):
        result = preprocess_single(
            subject_id=sid,
            visit_date=vdate,
            input_path=ipath,
            output_dir=args.output_dir,
            mode=args.mode,
            start_step=args.start_step,
            end_step=args.end_step,
            skull_strip_method=args.skull_strip,
            device=args.device,
            clip_range=clip_range,
            overwrite=args.overwrite,
        )
        results.append(result)

        if result['status'] == 'success':
            stats['success'] += 1
        else:
            stats['error'] += 1
            print(f"\n  ERROR {sid}_{vdate}: {result.get('error', 'unknown')}")

    elapsed = time.time() - start_time

    # QC on final outputs
    if args.qc and args.end_step == 6:
        print(f"\nRunning QC checks...")
        qc_results = []
        for r in results:
            if r['status'] == 'success' and 'output_path' in r:
                try:
                    qc = qc_check(r['output_path'])
                    qc_results.append({
                        'subject_id': r['subject_id'],
                        'visit_date': r['visit_date'],
                        **qc,
                    })
                    if qc['flags']:
                        print(f"  QC flags for {r['subject_id']}: {qc['flags']}")
                except Exception as e:
                    print(f"  QC error for {r['subject_id']}: {e}")

        if qc_results:
            qc_path = os.path.join(args.output_dir, 'qc_report.json')
            with open(qc_path, 'w') as f:
                json.dump(qc_results, f, indent=2)
            print(f"  QC report saved to {qc_path}")

    # Save processing log
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'preprocessing_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'stats': stats,
            'elapsed_seconds': elapsed,
            'results': results,
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {timedelta(seconds=int(elapsed))}")
    print(f"  Success: {stats['success']}")
    print(f"  Errors:  {stats['error']}")
    print(f"  Log: {log_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
