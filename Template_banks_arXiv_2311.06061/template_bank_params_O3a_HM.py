"""
Choices of template bank parameters for the search with higher modes 
"""
import numpy as np
import utils
import os
# import template_bank_generator_HM as tg

# Total number of banks
nbanks = 17

# Number of sub-banks in each bank
nsubbanks = {
'BBH_0': 5,
 'BBH_1': 3,
 'BBH_2': 3,
 'BBH_3': 3,
 'BBH_4': 3,
 'BBH_5': 3,
 'BBH_6': 3,
 'BBH_7': 2,
 'BBH_8': 2,
 'BBH_9': 1,
 'BBH_10': 1,
 'BBH_11': 1,
 'BBH_12': 1,
 'BBH_13': 1,
 'BBH_14': 1,
 'BBH_15': 1,
 'BBH_16': 1}

# ------------------------------------------------------------------------------
# Directory where the template bank files are located
DIR = os.path.join(utils.TEMPLATE_DIR, 'O3_hm', 'Multibanks')

# Path to the ASD file
# (currently using the O3a file)
asd_filepath = '/data/bzackay/GW/templates/O3a/asd_o3a.npy'
# asd_filepath = os.path.join(DIR, 'asd.npy')

# directory where generated waveforms will be stored
wf_DIR = os.path.join(utils.TEMPLATE_DIR, 'O3_hm', 'wf_reservoir')

# Flag used in grid_range function in
# template_bank_generator_hm.py to center the calpha grid
force_zero = True

# Following will be changed later
mb_keys = [
    'BBH_0', 
    'BBH_1', 
    'BBH_2', 
    'BBH_3',
    'BBH_4',
    'BBH_5', 
    'BBH_6', 
    'BBH_7', 
    'BBH_8',
    'BBH_9',
    'BBH_10', 
    'BBH_11', 
    'BBH_12', 
    'BBH_13',
    'BBH_14',
    'BBH_15',
    'BBH_16',
    ]

all_mb_keys = mb_keys

fudge = {
 'BBH_0': 1.05,
 'BBH_1': 1.05,
 'BBH_2': 1.05,
 'BBH_3': 1.05,
 'BBH_4': 1.05,
 'BBH_5': 1.05,
 'BBH_6': 1.05,
 'BBH_7': 1.05,
 'BBH_8': 1.05,
 'BBH_9': 1.05,
 'BBH_10': 1.05,
 'BBH_11': 1.05,
 'BBH_12': 1.05,
 'BBH_13': 1.05,
 'BBH_14': 1.05,
 'BBH_15': 1.05,
 'BBH_16': 1.05}

delta_calpha = {
 'BBH_0': 0.55,
 'BBH_1': 0.5,
 'BBH_2': 0.45,
 'BBH_3': 0.45,
 'BBH_4': 0.4,
 'BBH_5': 0.35,
 'BBH_6': 0.3,
 'BBH_7': 0.25,
 'BBH_8': 0.25,
 'BBH_9': 0.35,
 'BBH_10': 0.3,
 'BBH_11': 0.3,
 'BBH_12': 0.3,
 'BBH_13': 0.3,
 'BBH_14': 0.3,
 'BBH_15': 0.25,
 'BBH_16': 0.2}

mb_dirs = {x: os.path.join(DIR, x+'/') for x in all_mb_keys}

# input_wf_dirs = {x: os.path.join(DIR, x + '_input_wfs/')
#                  for x in all_mb_keys}
# test_wf_dirs = {x: os.path.join(DIR, x + '_test_wfs/')
#                 for x in all_mb_keys}
# coverage_wf_dirs = {x: os.path.join(DIR, x + '_coverage_wfs/')
#                     for x in all_mb_keys}
# 
# # for effectualness testing with precession and HM
# approximants_aligned = {x: 'IMRPhenomD' for x in mb_keys_BBH}
# approximants_aligned_HM = {x: 'IMRPhenomHM' for x in mb_keys_BBH}
# approximants_precessing = {x: 'IMRPhenomPv2' for x in mb_keys_BBH}
# approximants_precessing_HM = {x: 'IMRPhenomXPHM' for x in mb_keys_BBH}
# 
# test_wf_dirs_aligned = {x: os.path.join(DIR, x + '_test_wfs_aligned/')
#                         for x in mb_keys_BBH}
# test_wf_dirs_aligned_HM = {x: os.path.join(DIR, x + '_test_wfs_aligned_HM/')
#                            for x in mb_keys_BBH}
# test_wf_dirs_precessing = {x: os.path.join(DIR, x + '_test_wfs_precessing/')
#                            for x in mb_keys_BBH}
# test_wf_dirs_precessing_HM = {x: os.path.join(DIR, x + '_test_wfs_precessing_HM/')
#                               for x in mb_keys_BBH}
# test_wf_dirs_aligned_angles = {x: os.path.join(DIR,
#                                 x + '_test_wfs_aligned_angles/') for x in mb_keys_BBH}
# test_wf_dirs_aligned_HM_angles = {x: os.path.join(DIR,
#                                 x + '_test_wfs_aligned_HM_angles/') for x in mb_keys_BBH}
# test_wf_dirs_precessing_angles = {x: os.path.join(DIR,
#                                 x + '_test_wfs_precessing_angles/') for x in mb_keys_BBH}
# test_wf_dirs_precessing_HM_angles = {x: os.path.join(DIR,
#                                 x + '_test_wfs_precessing_HM_angles/') for x in mb_keys_BBH}


# def load_multibanks(mb_keys=None):
#     """
#     If mb_keys are ints it assumes BBH.
#     """
#     mb_keys = mb_keys or mb_dirs.keys()
#     return {k: tg.MultiBank.from_json(
#         mb_dirs[f'BBH_{k}' if isinstance(k, int) else k] + 'metadata.json')
#             for k in mb_keys}

# m1rng = {
#     **{f'BNS_{i}': (1, 3) for i in range(3)},
#     **{f'NSBH_{i}': (3, 100) for i in range(3)},
#     **{f'BBH_{i}': (3, 100) for i in range(5)},
#     'BBH_5': (100, 200), 'BBH_6': (100, 200)
# }
# 
# m2rng = {
#     **{f'BNS_{i}': (1, 3) for i in range(3)},
#     **{f'NSBH_{i}': (1, 3) for i in range(3)},
#     **{f'BBH_{i}': (3, 100) for i in range(5)},
#     'BBH_5': (10, 100), 'BBH_6': (10, 200)
# }
# 
# mcrng = {
#     'BNS_0': (0, 1.1),
#     'BNS_1': (1.1, 1.3),
#     'BNS_2': (1.3, np.inf),
#     'NSBH_0': (0, 3),
#     'NSBH_1': (3, 6),
#     'NSBH_2': (6, np.inf),
#     'BBH_0': (0, 5),
#     'BBH_1': (5, 10),
#     'BBH_2': (10, 20),
#     'BBH_3': (20, 40),
#     'BBH_4': (40, np.inf),
#     'BBH_5': (20, 200), 'BBH_6': (20, 200)
# }
# 
# qmin = {
#     **{f'BNS_{i}': 1e-2 for i in range(3)},
#     **{f'NSBH_{i}': 1/50 for i in range(3)},
#     **{f'BBH_{i}': 1/18 for i in range(5)},
#     'BBH_5': 1/10, 'BBH_6': 1/10}
# 
# srng = {
#     **{f'BNS_{i}': (-.99, .99) for i in range(3)},
#     **{f'NSBH_{i}': (-.99, .99) for i in range(3)},
#     **{f'BBH_{i}': (-.99, .99) for i in range(7)}}
# 
# lrng = {
#     **{f'BNS_{i}': (0, 0) for i in range(3)},
#     **{f'NSBH_{i}': (0, 0) for i in range(3)},
#     **{f'BBH_{i}': (0, 0) for i in range(7)}}