# -*- coding: utf-8 -*-
"""Shared NODDI O/N/F file mapping for all dataloaders."""

# =============================================================================
#  LEGACY MODALITY MAPPING  (IMPORTANT!!!)
# =============================================================================
# Due to an early typo, the mapping of O/N/F do NOT match their actual
# NODDI modalities. Keep this assignment to stay compatible with the bundled
# pretrained weights. Downstream finetuning use the same mapping, so downstream
# metrics remain valid.
#
# However, when interpreting results, remember to follow this mapping - not the
# literal O/N/F names:
#     O/enc_O  →  ICVF        N/enc_N  →  ISOVF        F/enc_F  →  OD
#
# -----------------------------------------------------------------------------
# To use semantically correct files without re-pretraining, switch to the FILES
# block below and permute model inputs so each encoder still sees its pretrain
# modality (use the same permutation in finetuning):
#     model(N, F, O)   # when dataloader O=OD, N=ICVF, F=ISOVF
#
# Safer option: pretrain from scratch with semantic FILES and no need to permute:
#     model(O, N, F)   # when dataloader O=OD, N=ICVF, F=ISOVF
#
#     FILES = {
#         "O": "NODDI_OD.npy",      # Oriented Diffusion (ODI)
#         "N": "NODDI_ICVF.npy",    # Intracellular Volume Fraction (NDI)
#         "F": "NODDI_ISOVF.npy",   # Isotropic Volume Fraction (FWF)
#     }
# =============================================================================
FILES = {
    "O": "NODDI_ICVF.nii.gz",
    "N": "NODDI_ISOVF.nii.gz",
    "F": "NODDI_OD.nii.gz",
}
