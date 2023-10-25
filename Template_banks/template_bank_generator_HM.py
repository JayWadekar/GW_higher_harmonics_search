import itertools
import json
import numpy as np
import pickle
import joblib
import os
import scipy
import scipy.signal as signal
import scipy.interpolate as interpolate
import sys
import utils
import warnings
import params
# import interpolated_approximant as ia
# import template_bank_generator as tg

"""
# Note: The input files for this code have been generated using the 
# following notebook:
# .../gw_detection_ias/scratch_files/TemplateBank_HigherModes.ipynb
# The section 'using Template_bank_generator class' in the notebook
# outlines how to start using the TemplateBank class
"""

# TODO: After fixing everything, check that plots_publication will not require the ndims parameter in the old way
# TODO: Over 1% of the parameter space, we need extra phases for the higher harmonics

# Internal parameters
# -----------------------------------------------------------------------------
FS_BASIS_PATH = "f_phases.npy"
FS_UNWRAP_PATH = "f_unwrap_phases.npy"
WT_PATH = "weights_phases.npy"
BASIS_PATH = "svd_phase_basis.npy"
BASIS_HM_PATH = "svd_phase_basis_HM.npy"
BASIS_AMPS_PATH = "svd_amp_basis.npy"
AMP_PATH = "Af.npy"
AVG_PHASE_EVOLUTION_PATH = 'avg_phase_evolution.npy'
AVG_PHASE_EVOLUTION_HM_PATH = 'avg_phase_evolution_HM.npy'
ASD_PATH = "asd.npy"
RANDOMFOREST_PATH = 'RandomForest.pkl'
RANDOMFOREST_HM_PATH = 'RandomForest_HM.pkl'
RANDOMFOREST_AMPS_PATH = 'RandomForest_amps.pkl'
COEFFS_PATH = "coeffs.npy"
PAR_PATH = "params.npy"
HM_AMP_RATIO_PATH = "HM_amp_ratio_samples.npy"


def get_df(fs):
    if np.allclose(fs[1]-fs[0], np.diff(fs)):
        df = fs[1] - fs[0]
    else:
        df = np.r_[(fs[1] - fs[0]) / 2,
                   (fs[2:] - fs[:-2]) / 2,
                   (fs[-1] - fs[-2]) / 2]
    return df


def compute_overlap(wf1, wf2, fs, asds):
    """
    Overlap between two waveforms
    :param wf1: ... x len(fs) array with frequency domain waveforms
                (units of 1/Hz), can be vector if n_wf = 1
    :param wf2: ... x len(fs) array with frequency domain waveforms
                (units of 1/Hz), can be vector if n_wf = 1
    :param fs: Array with regularly spaced frequencies (units of Hz)
    :param asds: ASDs in units of 1/sqrt(Hz) at fs
    :return: Scalar or array of length n_wf with overlaps
    """
    df = get_df(fs)
    overlaps = 4.0 * np.sum(np.real(wf1 * wf2.conj()) / np.abs(asds) ** 2 * df,
                            axis=-1)

    return overlaps


def transform_pars(pars):
    """
    Transforms [m1, m2, s1z, s2z, lambda1, lambda2] to commonly used
    parameters [mchirp, eta, chieff, chia, tilde{lambda},
    delta tilde{lambda}]
    :param pars:
        return value of gen_and_save_temp_structure
        (can be vector for n_sample = 1)
    :return: n_sample x 6 array with mchirp, eta, chieff, chia,
             tilde{lambda} and delta tilde{lambda}
             (can be vector for n_wf = 1)
    """
    pars = np.asarray(pars)
    pars2 = np.zeros(pars.shape)

    scf = lambda n:  tuple([slice(None)] * (pars.ndim - 1) + [n])

    pars2[scf(0)] = (pars[scf(0)] * pars[scf(1)]) ** 0.6 / \
                    (pars[scf(0)] + pars[scf(1)]) ** 0.2
    pars2[scf(1)] = (pars[scf(0)] * pars[scf(1)]) / \
                    (pars[scf(0)] + pars[scf(1)]) ** 2
    pars2[scf(2)] = \
        (pars[scf(0)] * pars[scf(2)] + pars[scf(1)] * pars[scf(3)]) / \
        (pars[scf(0)] + pars[scf(1)])
    pars2[scf(3)] = \
        (pars[scf(0)] * pars[scf(2)] - pars[scf(1)] * pars[scf(3)]) / \
        (pars[scf(0)] + pars[scf(1)])
    pars2[scf(4)] = \
        (8. / 13.0 *
         ((1. + 7. * pars2[scf(1)] - 31. * pars2[scf(1)] ** 2) *
          (pars[scf(4)] + pars[scf(5)]) -
          np.sqrt(1. - 4. * pars2[scf(1)]) *
          (1. + 9. * pars2[scf(1)] - 11. * pars2[scf(1)] ** 2) *
          (pars[scf(4)] - pars[scf(5)])))
    pars2[scf(5)] = \
        ((pars[scf(0)] - pars[scf(1)]) / (pars[scf(0)] + pars[scf(1)]) *
         pars2[scf(4)])

    return pars2


def get_efficient_frequencies(fmin=24, fmid=128, fmax=512,
                              mchirp_min=2**-.2, delta_radians=1):
    """
    Return a grid of frequencies that allows to safely
    get unwrapped phase.
    At low frequencies (fmin < f < fmid) it guarantees that
    the post-Newtonian phase increases by constant amounts of
    <= delta_radians for mchirp >= mchirp_min.
    At high frequencies (fmid < f < fmax) it switches to a
    regular grid.
    """
    # PN phase = factor * f^(-5/3)
    # factor = 3/128 * (lal.G_SI*lal.MSUN_SI/lal.C_SI**3*np.pi*mchirp_min)**(-5/3)
    # lal.G_SI*lal.MSUN_SI/lal.C_SI**3
    cons = 4.925491025543575e-06
    factor = 3/128 * (cons*np.pi*mchirp_min)**(-5/3)
    # Index is a function of f that increases by one when the phase
    # increases by delta_radians for a binary with mchirp_min.
    def f_to_index(f):
        return -factor * f**(-5/3) / delta_radians
    def index_to_f(i):
        return (-delta_radians * i / factor)**(-3/5)
    imin = f_to_index(fmin)
    imid = f_to_index(fmid)
    f_low_end = index_to_f(np.linspace(imin, imid, int(np.ceil(imid-imin))))

    df = f_low_end[-1] - f_low_end[-2]
    f_high_end = np.linspace(fmid, fmax, int(np.ceil(fmax-fmid)/df))
    return np.concatenate([f_low_end, f_high_end[1:]])


def gen_random_pars(n_wf, mtrng, qrng, mrng, srng, lrng, flat_in_chieff=False, mcrng=None):
    """Generates n_wf random parameters. Applies either mrng (preferably) or
    mtrng, along with qrng
    :param n_wf: Number of random parameter choices
    :param mtrng: Array of length 2 with range of total masses (in M_sun)
    :param qrng: Array of length 2 with range of mass ratios (q<1)
    :param mrng: Array/tuple of length 2 with range of component mass (in M_sun)
    :param srng: Array of length 2 with range of spins
    :param lrng: Array of length 2 with range of tidal deformabilities
    :param flat_in_chieff : Bool, if to take s1z,s2z uniform from srng (False) or take
                            chi_eff = m1*s1z+m2*s2z/(m1+m2) uniform in snrg,
                            and take chia = s1z-s2z uniform in the allowed range
                            given chieff (True)
    """
    pars = np.zeros((n_wf, 6))  # Save parameters to examine
    for i in range(n_wf):
        if mrng is not None:
            keep_drawing = True
            while keep_drawing:
                keep_drawing = False
                m2, m1 = sorted(np.random.uniform(mrng[0], mrng[1], 2))
                if (mtrng is not None) and (not utils.is_within(m1 + m2, mtrng)):
                    keep_drawing = True
                if (qrng is not None) and (not utils.is_within(m2 / m1, qrng)):
                    keep_drawing = True
                if (mcrng is not None) and \
                        (not utils.is_within((m1*m2)**(3/5)/(m1+m2)**(1/5), mcrng)):
                    keep_drawing = True
        elif (mtrng is not None) and (qrng is not None):
            keep_drawing =True
            while keep_drawing:
                keep_drawing = False
                mt = np.random.uniform(mtrng[0], mtrng[1])
                q = np.random.uniform(qrng[0], qrng[1])
                m1, m2 = mt / (1.0 + q), mt * q / (1.0 + q)

                if (mcrng is not None) and \
                        (not utils.is_within((m1*m2)**(3/5)/(m1+m2)**(1/5), mcrng)):
                    keep_drawing = True
        else:
            raise RuntimeError("No mass range passed!")

        if not flat_in_chieff:
            s1z, s2z = np.random.uniform(srng[0], srng[1], 2)
        else:
            c1 = m2/(m1+m2)
            c2 = m1/(m1+m2)
            chieff = np.random.uniform(*srng, 1)
            min_chia = np.max(((-1-chieff)/c1, (-1+chieff)/c2))
            max_chia = np.min(((1+chieff)/c2, (1-chieff)/c1))
            chia = np.random.uniform(min_chia, max_chia)
            s1z = chieff+c1*chia
            s2z = chieff-c2*chia

        l1, l2 = np.random.uniform(lrng[0], lrng[1], 2)
        pars[i, :] = np.array((m1, m2, s1z, s2z, l1, l2))
    return pars


def upsample_lwfs(lwfs, fs_in, fs_out, phase_only=False):
    """
    Upsample given lwfs to an output frequency grid
    Note: Amplitudes are linearly interpolated, with zeros outside the range
          Pad phases with edge values to avoid discontinuities at the edges
    :param lwfs:
        ... x len(fs_in) array with log of waveforms in frequency domain
        (make sure to pass unwrapped phase)
    :param fs_in: Array of size(fs_in) with frequencies
    :param fs_out: Array of size(fs_out) with frequencies
    :param phase_only:
        Flag indicating whether to compute and return only the phases
    :return:
        ... x len(fs_out) array with log of upsampled waveforms
        (phases if phase_only is True)
    """
    phase_interp = lambda phase_arr: np.interp(fs_out, fs_in, phase_arr)
    phase_out = np.apply_along_axis(phase_interp, -1, lwfs.imag)

    if phase_only:
        return phase_out

    amp = np.exp(lwfs.real)
    lamp_interp = lambda amp_arr: \
        np.log(np.interp(fs_out, fs_in, amp_arr, left=0., right=0.))
    # ignore -np.inf error as we later take exp of this which makes this zero
    l_amp_out = np.apply_along_axis(lamp_interp, -1, amp)

    return l_amp_out + 1j * phase_out


def remove_linear_component(phases, fs, wts, return_coeffs=False):
    """
    Fits out a linear dependence of phase w.r.t frequency
    :param phases:
        n_wf x len(fs) array with angles (Note that it needs to be a 2d array)
    :param fs: Array with frequencies
    :param wts:
        Weights for linear fit, note that for Gaussian w/ variance sigma^2,
        weights are 1/sigma
    :param return_coeffs: Flag to return coefficients of the linear terms
    :return: 1. n_wf x len(fs) array with unweighted residuals of linear fit
             2. If return_coeffs, n_wf x 2 array with coefficients of linear
                and constant terms (in that order)
    """
    # Remove tc and constant phase
    lin_fit = np.polyfit(
        x=fs, y=np.transpose(np.atleast_2d(phases)), deg=1, w=wts)
    # Bestfit linear trend (n_wf x len(fs))
    # bestfit = np.dot(np.transpose(lin_fit), np.stack([fs, np.ones(len(fs))]))
    bestfit = np.array([np.poly1d(i)(fs) for i in np.transpose(lin_fit)])

    # Treat edges
    bestfit[:, wts == 0] = 0

    # Input to inner product (note, need to weight for SVD)
    residuals = (phases - bestfit)  # n_wf x len(fs)

    if return_coeffs:
        return residuals, lin_fit.T
    else:
        return residuals


def transform_basis(phases, avg_phase_evolution, basis, wts, fs):
    """Returns components of waveforms in the given basis
    :param phases:
        n_wf x len(fs) complex array with unwrapped phases of
        frequency domain waveforms (note it needs to be a 2d array)
    :param avg_phase_evolution:
        len(fs) array with average phase evolution * weights
    :param basis: n_basis x len(fs) array with orthonormal basis shapes
    :param wts:
        Array of size len(fs) with inner product weights to convert
        wfs into basis
    :param fs: Array with frequencies (Hz)
    :return:
        n_wf x n_basis array with components of each waveform along each basis
        element
    """
    svd_in = remove_linear_component(phases, fs, wts) - avg_phase_evolution / wts
    ips = np.dot(wts * svd_in, basis.T)  # n_wf x n_basis
    return ips


def compute_hplus_hcross(f, par_dic, approximant: str,
                         harmonic_modes=None, force_nnlo=True):
    """
    Generate frequency domain waveform using LAL.
    Return hplus, hcross evaluated at f.

    Parameters
    ----------
    approximant: String with the approximant name.
    f: Frequency array in Hz
    par_dic: Dictionary of source parameters. Needs to have these keys:
                 * m1, m2: component masses (Msun)
                 * d_luminosity: luminosity distance (Mpc)
                 * iota: inclination (rad)
                 * phi_ref: phase at reference frequency (rad)
                 * f_ref: reference frequency (Hz)
                 * s1x, s1y, s1z, s2x, s2y, s2z: dimensionless spins
                 * l1, l2: dimensionless tidal deformabilities
    harmonic_modes: Optional, list of 2-tuples with (l, m) pairs
                  specifying which (co-precessing frame) higher-order
                  modes to include.
    """
    
    lal = utils.load_module('lal')
    lalsimulation = utils.load_module('lalsimulation')
    
    par_dic = par_dic.copy()

    # Parameters ordered for lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_params = [
        'phi_ref', 'm1_kg', 'm2_kg', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
        'f_ref', 'd_luminosity_meters', 'iota', 'lal_dic', 'approximant', 'f']

    # SI unit conversions
    par_dic['d_luminosity_meters'] = par_dic['d_luminosity'] * 1e6 * lal.PC_SI
    par_dic['m1_kg'] = par_dic['m1'] * lal.MSUN_SI
    par_dic['m2_kg'] = par_dic['m2'] * lal.MSUN_SI

    lal_dic = lal.CreateDict()  # Contains tidal and higher-mode parameters
    if force_nnlo:
        lalsimulation.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lal_dic,102)
    # Tidal parameters
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
        lal_dic, par_dic['l1'])
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(
        lal_dic, par_dic['l2'])
    # Higher-mode parameters
    if harmonic_modes is not None:
        mode_array = lalsimulation.SimInspiralCreateModeArray()
        for l, m in harmonic_modes:
            lalsimulation.SimInspiralModeArrayActivateMode(mode_array, l, m)
        lalsimulation.SimInspiralWaveformParamsInsertModeArray(lal_dic,
                                                               mode_array)
    par_dic['lal_dic'] = lal_dic
    
    #par_dic['iota'] = 0.3

    par_dic['approximant'] = lalsimulation.GetApproximantFromString(
        approximant)

    par_dic['f'] = lal.CreateREAL8Sequence(len(f))
    par_dic['f'].data = f

    hplus, hcross = lalsimulation.SimInspiralChooseFDWaveformSequence(
        *[par_dic[par] for par in lal_params])
    hplus_hcross = np.stack([hplus.data.data, hcross.data.data])
    return hplus_hcross


def gen_waveform(
        fs_gen=None, sub_fac=1, m1=30, m2=30, s1x=0, s1y=0, s1z=0, s2x=0, s2y=0,
        s2z=0, l1=0, l2=0, approximant='IMRPhenomD_NRTidal', unwrap=True,
        phase="random", deltaF=None, f_min=None, f_max=None, inclination=None,
        phiRef=None, f_ref=None, distance=None, **kwargs_pars):
    """Generates waveforms with given binary parameters
    :param fs_gen: Set of frequencies (Hz) (alternatively, pass deltaF etc...)
    :param sub_fac:
        Factor we subsampled the frequencies by to avoid unwrap errors
    :param m1: First mass (in M_sun)
    :param m2: Second mass (in M_sun)
    :param s1x: X component of first spin (dimensionless)
    :param s1y: Y component of first spin (dimensionless)
    :param s1z: Z component of first spin (dimensionless)
    :param s2x: X component of second spin (dimensionless)
    :param s2y: Y component of second spin (dimensionless)
    :param s2z: Z component of second spin (dimensionless)
    :param l1: Tidal deformalbility of first mass
    :param l2: Tidal deformalbility of second mass
    :param approximant: LAL string for approximant
    :param unwrap: Flag indicating whether to unwrap and return amp and phase
    :param phase:
        If known, pass phase to use, else pass "random" to indicate
        a random phase
    :param deltaF: If on a regular grid, frequency spacing (Hz)
    :param f_min:
        If on a regular grid, minimum frequency to query the approximant at (Hz)
    :param f_max:
        If on a regular grid, maximum frequency (Hz, ~Nyquist frequency)
    :return:
        1. len(fs_gen / sub_fac) array with amplitude of frequency domain
           waveform (units of 1/Hz)
        2. len(fs_gen / sub_fac) array with phase of frequency domain waveform
    """
    if 'ComputeWF_farray' not in sys.modules:
        import ComputeWF_farray as WFgen
    else:
        WFgen = sys.modules['ComputeWF_farray']

    harr = WFgen.genwf_fd(
        frequencies=fs_gen, deltaF=deltaF, f_min=f_min, f_max=f_max,
        m1=m1, m2=m2,
        spin_1_x=s1x, spin_1_y=s1y, spin_1_z=s1z,
        spin_2_x=s2x, spin_2_y=s2y, spin_2_z=s2z,
        lambda_1=l1, lambda_2=l2, approximant=approximant,
        inclination=inclination, phiRef=phiRef,
        f_ref=f_ref, distance=distance, **kwargs_pars)

    if isinstance(phase, str):
        # Random phase
        alpha = np.random.uniform(0., 2.0 * np.pi)
        hrand = np.cos(alpha) * harr[:, 0] + np.sin(alpha) * harr[:, 1]
    else:
        hrand = np.cos(phase) * harr[:, 0] + np.sin(phase) * harr[:, 1]

    if unwrap:
        angle = np.unwrap(np.angle(hrand))
        amp = np.abs(hrand[::sub_fac])
        phase = angle[::sub_fac]
        return amp, phase
    else:
        return hrand


# Class for template banks
# -----------------------------------------------------------------------------
class TemplateBank(object):

    # IO for template bank info
    # -------------------------------------------------------------------------
    def __init__(
            self, fs_basis=None, wts=None, basis=None, basis_HM=None, 
            coeffs=None, ndims=None, avg_phase_evolution=None,
            avg_phase_evolution_HM=None, randomforest=None,
            randomforest_HM=None, amps_calpha_dependent=False,
            randomforest_amps=None, basis_amps=None,
            pars=None, amp=None, approximant=None, HM_amp_ratio_samples=None,
            min_shift_calpha_ind=None, glitch_threshold_wf_params=None,
            glitch_threshold_intervals_saved=None, asdf=None, fs_unwrap=None,
            config_fname=None, coverage_lwfs=None,
            coverage_coeffs=None, coverage_pars=None):
        """
        :param fs_basis:
            Array with basis frequencies (Hz), chosen as centers of bins such
            that the (2, 2) weights (\int df |A_22|^2/S(f)) are equal
        :param wts: len(fs_basis) array with weights to go from phases to basis
        :param basis:
            n_basis x len(fs_basis) array with basis functions for phase
        :param basis_HM:
            each is mode x n_basis x len(fs_basis) array with basis functions for
            (phase_33 - 3/2 * phase_22) or (phase_44 - 4/2 * phase_22)     
        :param amp:
            len(f_amp) x (1+ n_modes) array with frequencies (first column),
            and amplitudes at those frequencies (f_amp is different in general
            from fs_basis). Note f_amp goes below fs_basis[0]
        :param avg_phase_evolution:
            len(fs_basis) array with average evolution of (phase * weights)
        :param avg_phase_evolution_HM:
            same as avg_phase_evolution but corresponding to 
            (phase_33 - 3/2 * phase_22)
        :param ndims: Number of dimensions we use in the cartesian grid
        :param randomforest:
            Instance of sklearn.ensemble.RandomForestRegressor that takes an
            n_dims tuple of c_alphas, and returns a (n_basis - ndims) tuple of
            c_alphas to concatenate to get the full c_alpha tuple living in the
            physical space
        :param randomforest_HM:
            Takes an n_dims tuple of c_alphas of the 22 mode and returns 
            three-dimensional c_alphas corresponding to (phase_33 - 3/2 * phase_22)
        :param amps_calpha_dependent:
            Boolean flag for whether we are setting amps to be dependent on calphas
        :param basis_amps:
            if amps_calpha_dependent: 3 x 2 x len(f_amp) array.
            3 is for nmodes, 2 is for the first two SVD basis of residual amplitudes
             (i.e., amps - mean amps of banks)
        :param randomforest_amps:
            if amps_calpha_dependent: Takes an n_dims tuple of c_alphas of the 22 mode
            and returns 6-dimensional coefficients (2 dims each corresponding to 22,33,44)
            corresponding to SVDs of the residual amplitudes (i.e., the amplitudes
             after subtracting the mean amplitude of the bank)
        :param coeffs: n_wf x n_basis array with components of sample waveforms
        :param pars: n_wf x 6 array with [m1, m2, s1z, s2z, l1, l2] for each
                     waveform
        :param approximant: LAL string for approximant
        :param HM_amp_ratio_samples: 
            2 x n_samples x [|Z_33|/|Z_22|, |Z_44|/|Z_22|, weights]
            corresponding to physical wfs in the bank
            0th element: samples with random inclinations and orthogonalized modes
            1st element: samples with pi/2 inclinations and unorthogonalized modes
                         (needed for coherent score calculation)
        :param min_shift_calpha_ind: index in bank.coeffs which has
                                     minimum shift (used in conditioning the wf)
        :param glitch_threshold_wf_params: parameters for simulating wfs with HM (e.g.,
            calphas, phiref, mode ratios) which are used for calc. glitch thresholds
        :param glitch_threshold_intervals_saved: saved version of all ....INTERVALS
            entries in params.py corresponding to glitch_threshold_wf_params
        :param asdf:
            Function that takes frequencies (in Hz), returns ASD
            (in 1/sqrt(Hz)), to record the ASD that was used to generate the
            bank, distinct from the ASD of the actual data that is used to
            generate whitened waveforms
        :param fs_unwrap: Frequencies where the phase was unwrapped
        :param config_fname: json file with template information
        :param coverage_lwfs: n_wf x len(fs_lwf) array with log of frequency
                              domain coverage waveforms (make sure to pass
                              unwrapped phase). These are not used to build the
                              bank but to provide more complete coverage in
                              calpha space, to make get_approximate_params more
                              accurate.
        :param coverage_coeffs: n_wf x n_basis array with components of coverage
                                waveforms
        :param coverage_pars: n_wf x 6 array with [m1, m2, s1z, s2z, l1, l2]
                              for each coverage waveform
        """
        super().__init__()

        # Template bank information
        self.fs_basis = fs_basis
        self.basis = basis
        self.basis_HM = basis_HM
        self.wts = wts
        self.amp = amp
        self.avg_phase_evolution = avg_phase_evolution
        self.avg_phase_evolution_HM = avg_phase_evolution_HM
        self.ndims = ndims
        self.randomforest = randomforest
        self.randomforest_HM = randomforest_HM
        self.amps_calpha_dependent = amps_calpha_dependent
        self.randomforest_amps = randomforest_amps
        self.basis_amps = basis_amps
        self.fs_unwrap = fs_unwrap
        self.phys_grid = {}

        # Information of waveforms and ASD used to make the bank
        self.coeffs = coeffs
        # Bounds defines the dimensionality of the iterator
        # (the rest is determined by the random forest)
        self.bounds = np.c_[np.min(coeffs[:, :ndims], axis=0),
                            np.mean(coeffs[:, :ndims], axis=0),
                            np.max(coeffs[:, :ndims], axis=0)]
        self.pars = pars
        self.approximant = approximant
        self.HM_amp_ratio_samples = HM_amp_ratio_samples
        self.min_shift_calpha_ind = min_shift_calpha_ind
        self.glitch_threshold_wf_params = glitch_threshold_wf_params
        self.glitch_threshold_intervals_saved = glitch_threshold_intervals_saved
        self.asdf = asdf
        
        # Information of waveforms used to increase calpha coverage of the bank
        self.coverage_lwfs = coverage_lwfs
        self.coverage_coeffs = coverage_coeffs
        self.coverage_pars = coverage_pars
        if coverage_pars is not None and coverage_coeffs is not None:
            self.all_pars = np.r_[self.pars, self.coverage_pars]
            self.all_coeffs = np.r_[self.coeffs, self.coverage_coeffs]
        else:
            self.all_pars = self.pars
            self.all_coeffs = self.coeffs

        # File with metadata
        self.config_fname = config_fname

        # Parameters for matched filtering (start off uninitiated, these depend
        # on the detector and data analysis choices). Update as needed
        self.fftsize = None
        self.dt = None
        self.fs_fft = None
        self.analytic_wt_filter = False
        self.wt_filter_fd = None
        self.support_whitened_wf = None
        self.shift_whitened_wf = None
        self.normfac = None

        # Highpass filter parameters
        self.sos = None
        self.irl = None
        
        if not self.amps_calpha_dependent:
            # Make amplitudes smoothly go to zero
            # self.amp[0] is the freq. array for amplitudes (different from fs_basis)
            for mode in range(1, len(self.amp[0])):
                x = self.amp[:, 0] - self.fs_basis[0] * (mode + 1) / 2
                self.amp[:, mode] /= (1 + np.exp(-x * 10))
                x = self.fs_basis[-1] * (mode + 1) / 2 - self.amp[:, 0]
                mask = x>-5
                self.amp[:, mode] *= np.r_[1/(1 + np.exp(-x[mask] * 2)), 0*x[~mask]]
            
            # Normalizing the amplitudes
            df = get_df(self.amp[:, 0])
            asd = self.asdf(self.amp[:, 0])
            self.amp[:, 1:] /= np.sqrt(
                    np.sum((self.amp[:, 1:].T)**2 * 4 * df / asd**2, axis=-1))

        return

    @classmethod
    def from_json(cls, config_fname):
        """
        Instantiates class from json file created by previous run
        :param config_fname: json file with template information
        :return: Instance of TemplateBank
        """
        with open(config_fname, 'r') as fp:
            dic = json.load(fp)
        
        # Extract the directory name from config_fname and rely on
        # the file structure to figure out the other filenames.
        # This way we can move template bank directories around
        # and preserve their functionality:
        subbank_dir = os.path.dirname(config_fname)
        bank_dir = os.path.dirname(subbank_dir)
        multibank_dir = os.path.dirname(bank_dir)

        fname_fs_basis = os.path.join(bank_dir, FS_BASIS_PATH)
        fname_wts = os.path.join(bank_dir, WT_PATH)
        fname_basis = os.path.join(subbank_dir, BASIS_PATH)
        fname_basis_HM = os.path.join(subbank_dir,'HM',BASIS_HM_PATH)
        fname_basis_amps = os.path.join(subbank_dir,BASIS_AMPS_PATH)
        fname_amp = os.path.join(bank_dir, AMP_PATH)
        fname_avg_phase_evolution = os.path.join(
            subbank_dir, AVG_PHASE_EVOLUTION_PATH)
        fname_avg_phase_evolution_HM = os.path.join(
            subbank_dir,'HM',AVG_PHASE_EVOLUTION_HM_PATH)
        fname_random_forest = os.path.join(subbank_dir, RANDOMFOREST_PATH)
        fname_random_forest_HM = os.path.join(subbank_dir,'HM',RANDOMFOREST_HM_PATH)
        fname_random_forest_amps = os.path.join(subbank_dir,RANDOMFOREST_AMPS_PATH)
        fname_asds = os.path.join(multibank_dir, ASD_PATH)
        fname_coeffs = os.path.join(subbank_dir, COEFFS_PATH)
        fname_pars = os.path.join(subbank_dir, PAR_PATH)
        fname_fs_unwrap = os.path.join(bank_dir, FS_UNWRAP_PATH)
        fname_HM_amp_ratio_samples = os.path.join(subbank_dir,'HM', HM_AMP_RATIO_PATH)

        load_kwargs = dict()

        load_kwargs["fs_basis"] = np.load(fname_fs_basis)
        load_kwargs["wts"] = np.load(fname_wts)
        load_kwargs["basis"] = np.load(fname_basis)
        load_kwargs["basis_HM"] = np.load(fname_basis_HM)
        load_kwargs["amp"] = np.load(fname_amp)
        load_kwargs["avg_phase_evolution"] = np.load(fname_avg_phase_evolution)
        load_kwargs["avg_phase_evolution_HM"] = np.load(fname_avg_phase_evolution_HM)
        load_kwargs["ndims"] = dic.get("ndims")
        load_kwargs["randomforest"] = joblib.load(fname_random_forest)
        load_kwargs["randomforest_HM"] = joblib.load(fname_random_forest_HM)
        load_kwargs["coeffs"] = np.load(fname_coeffs)
        load_kwargs["pars"] = np.load(fname_pars)
        load_kwargs["fs_unwrap"] = np.load(fname_fs_unwrap)
        load_kwargs["approximant"] = dic.get('approximant', None)
        load_kwargs["min_shift_calpha_ind"] = dic.get('min_shift_calpha_ind', None)
        load_kwargs["HM_amp_ratio_samples"] = np.load(fname_HM_amp_ratio_samples)
        load_kwargs["glitch_threshold_wf_params"] = dic.get(
                                            'glitch_threshold_wf_params', None)
        load_kwargs["glitch_threshold_intervals_saved"] = dic.get(
                                            'glitch_threshold_intervals_saved', None)

        if os.path.isfile(fname_asds):
            load_kwargs["asdf"] = utils.asdf_fromfile(fname_asds)
        
        load_kwargs["amps_calpha_dependent"] = os.path.isfile(fname_basis_amps)
        if load_kwargs["amps_calpha_dependent"]:
            load_kwargs["basis_amps"] = np.load(fname_basis_amps)
            load_kwargs["randomforest_amps"] = joblib.load(fname_random_forest_amps)
            
        instance = cls(**load_kwargs)

        return instance

    def gen_amps_from_calpha(self, calpha=None, fs_out=None):
        """
        Generates amplitudes of all modes from given calphas
        :param calpha:
            n_wf x n_(basis elements needed) array with list of coefficients
            (can be vector for n_wf = 1). Defaults to the central waveform
        :param fs_out:
            Array of output frequencies. None indicates fs_out = fs_basis
        :return: n_wf x 3 x len(fs_out) real array with amplitudes at fs_out
        """
        if fs_out is None: fs_out = self.fs_basis
        
        if calpha is None: calpha = self.bounds[:, 1]
        
        calpha = np.asarray(calpha)
        if (calpha.ndim == 1): calpha=calpha.reshape(1,-1)
        
        calpha_amps = self.randomforest_amps.predict(calpha[..., :self.ndims])
        
        # These are residuals of amplitudes
        amp = np.array([np.dot(calpha_amps[...,:2], self.basis_amps[0]),
                              np.dot(calpha_amps[...,2:4], self.basis_amps[1]),
                              np.dot(calpha_amps[...,4:], self.basis_amps[2])])
        amp = np.moveaxis(amp,0,1)
            
        # Adding mean amplitudes of the bank
        amp += self.amp[:,1:].T
        amp[amp<0] = 0.
        
        # Make amplitudes smoothly go to zero
        # self.amp[0] is the freq. array for amplitudes (different from fs_basis)
        for mode in range(3): 
            x = self.amp[:, 0] - self.fs_basis[0] * (mode + 2) / 2
            amp[..., mode,:] /= (1 + np.exp(-x * 10))
            x = self.fs_basis[-1] * (mode + 2) / 2 - self.amp[:, 0]
            mask = x>-5
            amp[..., mode,:] *= np.r_[1/(1 + np.exp(-x[mask] * 2)), 0*x[~mask]]
            
        # Normalizing the amplitudes
        df = get_df(self.amp[:, 0])
        asd = self.asdf(self.amp[:, 0])
        amp = (amp.T/ np.sqrt(
                np.sum(amp**2 * 4 * df / asd**2, axis=-1)).T).T
        # Interpolating
        amp_intep = np.zeros(amp.shape[:-1] + (len(fs_out),))
        
        for j in range(len(amp_intep)):
            amp_intep[j] =  [np.interp(fs_out, self.amp[:, 0], amp[j,i],
                                         left=0, right=0) for i in range(3)]
        
        if len(amp_intep)==1: amp_intep = amp_intep[0]
        
        return(amp_intep)
    
    def gen_phases_from_calpha(self, calpha=None, fs_out=None):
        """
        Generates phases of all modes from given coefficients
        :param calpha:
            n_wf x n_(basis elements needed) array with list of coefficients
            (can be vector for n_wf = 1). Defaults to the central waveform
        :param fs_out:
            Array of output frequencies. None indicates fs_out = fs_basis
        :return: n_wf x 3 x len(fs_out) real array with phases at fs_out
        """
        if fs_out is None:
            fs_out = self.fs_basis
        
        if calpha is None:
            calpha = self.bounds[:, 1]

        calpha = np.asarray(calpha)
        if (calpha.ndim > 1):
            calpha = np.c_[calpha[..., :self.ndims],
                    self.randomforest.predict(calpha[..., :self.ndims])]
            calpha_HM = self.randomforest_HM.predict(calpha[..., :self.ndims])
            
        elif (calpha.ndim == 1):
            calpha = np.append(calpha[..., :self.ndims],
                        self.randomforest.predict([calpha[..., :self.ndims]])[0])
            calpha_HM = self.randomforest_HM.predict([calpha[..., :self.ndims]])[0]
            
        # n_wf x len(fs_basis) array with basis shapes
        # (can be len(fs_basis) for n_wf = 1)
        weighted_angles = np.dot(calpha, self.basis[:calpha.shape[-1], :])
        # Add the average weighted phase evolution
        weighted_angles += self.avg_phase_evolution
        # Unweight to go to angle space, properly treat zero weight
        angles = weighted_angles / self.wts
        
        # For HM
        angles_33 = 3 / 2 * angles
        angles_44 = 4 / 2 * angles
        
        angles_33 += (np.dot(calpha_HM[...,2:5], self.basis_HM[:3]) +
                         self.avg_phase_evolution_HM[0])/ self.wts
        angles_44 += (np.dot(calpha_HM[...,5:8], self.basis_HM[3:]) +
                         self.avg_phase_evolution_HM[1])/ self.wts
        
        # Adding the constant phase between the different modes
        angles_33 = (angles_33.T + calpha_HM[...,0].T).T
        angles_44 = (angles_44.T + calpha_HM[...,1].T).T
            
        # Resample to output frequency resolution if needed
        if angles.ndim == 1:
            ang_int = np.interp(fs_out, self.fs_basis, angles)
            ang_int_33 = np.interp(fs_out, self.fs_basis * 3 / 2, angles_33)
            ang_int_44 = np.interp(fs_out, self.fs_basis * 4 / 2, angles_44)
            ang_int = np.stack((ang_int, ang_int_33, ang_int_44), axis=0)

        else:
            ang_int = np.array(
                    [np.interp(fs_out, self.fs_basis, x) for x in angles])
            ang_int_33 = np.array(
                    [np.interp(fs_out, self.fs_basis * 3 / 2, x) for x in angles_33])
            ang_int_44 = np.array(
                    [np.interp(fs_out, self.fs_basis * 4 / 2, x) for x in angles_44])
            ang_int = np.stack((ang_int, ang_int_33, ang_int_44), axis=1)
                    
        # Return aligned memory
        phases = utils.FFTIN(ang_int.shape)
        phases[:] = ang_int[:]

        return phases
        
        
    def get_pdic_from_calpha(self, calpha, m1rng=(1, 200), m2rng=(1, 100),
                             add_pars=False, print_estimate=False):
        app = ia.InterpolatedApproximant.from_templatebank(self)
        m1, m2, s1z, s2z = app.get_params_from_calpha(calpha,
            m1rng=m1rng, m2rng=m2rng, print_estimate=print_estimate)
        pdic = {'m1': m1, 'm2': m2, 's1z': s1z, 's2z': s2z}
        pdic.update({'mchirp': (m1*m2)**.6 / (m1 + m2)**.2, 'q': m2/m1,
                     'eta': m1*m2 / (m1 + m2)**2,
                     'chieff': (m1*s1z + m2*s2z) / (m1 + m2)})
        if add_pars:
            pdic.update({'s1y': 0, 's2y': 0, 's1x': 0, 's2x': 0, 'l1': 0, 'l2': 0,
                         'd_luminosity': 1, 'iota': 0, 'vphi': 0,
                         'ra': 0, 'dec': 0, 'psi': 0, 't_geocenter': 0})
        pdic['calpha_in'] = calpha.copy()
        pdic['calpha_out'] = app.calpha_map([m1, m2, s1z, s2z, 0, 0])
        return pdic

    def get_linear_free_shift_from_wf_fd(self, wf_fd, freq=None):
        if freq is None:
            freq = self.fs_fft
            wtfilt = np.abs(self.wt_filter_fd)
        else:
            wtfilt = 1 / self.asdf(freq)
        df = get_df(freq)
        amp, phase = np.abs(wf_fd), np.unwrap(np.angle(wf_fd))
        amp_wt = amp * wtfilt
        phase_linfree = remove_linear_component(
            [phase], freq, amp_wt * np.sqrt(df))[0]
        i0 = np.argmax(amp_wt)  # Index of a relevant frequency
        t0 = (phase[i0 + 1] - phase[i0]
              - phase_linfree[i0 + 1] + phase_linfree[i0]) / (2 * np.pi * df)
        return t0

    def get_linear_free_shift_from_pars(self, **pars):
        if pars.get('approximant') is None:
            print('WARNING: Approximant not provided! Using IMRPhenomXAS')
            pars['approximant'] = 'IMRPhenomXAS'
        wf_fd = self.gen_wf_fd_from_pars(fs_out=self.fs_fft, linear_free=False,
                                         **pars)
        return self.get_linear_free_shift_from_wf_fd(wf_fd)

    def get_linear_free_shift_from_calpha(self, calpha):
        return self.get_linear_free_shift_from_pars(
            **self.get_pdic_from_calpha(calpha))
            
    def gen_wf_td_from_pars(
            self, dt=None, target_snr=None, highpass=False, phase=0, **pars):
        """
        WARNING: Not modified by Jay for the case of HM
        
        Generate time domain waveform for injection, let lal do the
        conditioning for us (works only for approximants with td waveforms)
        :param dt: Time spacing of samples (s). If None, uses self.dt
        The following variables need bank to be conditioned
        :param target_snr: Target SNR to achieve. Pass None to not apply this
        :param highpass: Flag indicating whether to high-pass filter waveforms
        :param phase: Phase of waveform
        :param pars:
            Desired subset of m1, m2, s1x, s1y, s1z, s2x, s2y,
            s2z, l1, l2, approximant. If approximant is None, uses
            self.approximant. See ComputeWF_farray.genwf_td for other defaults
        :return: Array with time-domain waveform (with spacing dt)
        """
        import ComputeWF_farray as WFgen

        # Define time spacing if not set
        if dt is None:
            if self.dt is not None:
                dt = self.dt
            else:
                raise RuntimeError("Need time spacing to generate waveform!")

        # Set approximant if not passed
        if pars.get("approximant") is None:
            if self.approximant is not None:
                pars["approximant"] = self.approximant
            else:
                raise RuntimeError("Need approximant to generate waveform!")

        # Generate h_+ and h_x and apply the linear combination from the phase
        harrs = WFgen.genwf_td(dt, params.FMIN, **pars)
        wf_td = np.cos(phase) * harrs[:, 0] + np.sin(phase) * harrs[:, 1]

        if (target_snr is not None) or highpass:
            if dt != self.dt:
                raise RuntimeError("I cannot resample bank parameters!")

        if target_snr is not None:
            # Pad to fftsize
            wf_td_padded = utils.FFTIN(self.fftsize)
            wf_td_padded[-len(wf_td):] = wf_td[:]

            # Frequency domain waveform (units of 1/Hz)
            wf_fd = utils.RFFT(wf_td_padded, axis=-1) * self.dt
            snrsq = compute_overlap(
                wf_fd, wf_fd, self.fs_fft, 1. / self.wt_filter_fd)

            # Renormalize to target SNR
            wf_td *= target_snr / np.sqrt(snrsq)

        if highpass:
            # Remove low frequency components
            wf_td = signal.sosfiltfilt(
                self.sos, wf_td, padlen=self.irl, axis=-1)

        return wf_td

    def gen_wf_fd_from_pars(
            self, fs_out=None, gen_domain='fd', phase=0, linear_free=True,
            **pars):
        """
        Convenience interface to gen_waveform that removes the linear
        component of the phase in the same way as was used to make the bank
        :param fs_out:
            If desired, generate waveform at these frequencies. Defaults to
            self.fs_fft (useful for, e.g., relative binning. In case we're
            passing fs_out, make sure that unwrap errors won't occur)
        :param gen_domain:
            String indicating whether the approximant is td or fd, use 'td'
            when LAL doesn't have fd version implemented
        :param phase: Phase of waveform
        :param linear_free: Flag whether to return the linear free waveform
        :param pars: Desired subset of m1, m2, s1x, s1y, s1z, s2x, s2y,
            s2z, l1, l2, approximant. If approximant is None, uses
            self.approximant. See ComputeWF_farray.genwf_td for other defaults
        :return:
            Array of length self.fs_fft with unconditioned frequency
            domain waveform
        """
        # Set approximant if it wasn't passed
        if pars.get("approximant") is None:
            #if self.approximant is not None:
            #    pars["approximant"] = self.approximant
            #else:
            raise RuntimeError("Need approximant to generate waveform!")

        if (gen_domain.lower() == 'td'):
            # Hack to make it work on laptop with IMRPhenomD, generates
            # time-domain waveforms using LAL's conditioning
            # Warning: Assumes SNR loss at low frequencies due to LAL's
            # conditioning is under control, check to make sure
            wf_td = self.gen_wf_td_from_pars(dt=self.dt, phase=phase, **pars)

            if len(wf_td) > self.fftsize:
                raise RuntimeError("The waveform is too long for our bank!")

            # Pad it to the right length
            wf_td_padded = utils.FFTIN(self.fftsize)
            wf_td_padded[-len(wf_td):] = wf_td[:]
            wf_fd_padded = utils.RFFT(wf_td_padded)

            # Array of frequencies
            fs_gen = self.fs_fft
            # All frequencies are valid here since the waveform was already
            # conditioned, avoid multiplying with a step
            fmask = np.ones_like(self.fs_fft, dtype=bool)
        else:
            # Generate frequency domain waveform
            if fs_out is not None:
                # Only consider frequencies above fmin
                fmask = fs_out >= params.FMIN
                # Generate waveform at the required frequencies
                wf_fd_padded = utils.FFTIN(len(fs_out), dtype=np.complex128)
                wf_fd_padded[fmask] = np.nan_to_num(gen_waveform(
                    fs_gen=fs_out[fmask], sub_fac=1, unwrap=False,
                    phase=phase, **pars))

                # Array of frequencies
                fs_gen = fs_out
            else:
                df = self.fs_fft[1] - self.fs_fft[0]
                wf_fd_padded = gen_waveform(
                    unwrap=False, phase=phase, deltaF=df, f_min=params.FMIN,
                    f_max=self.fs_fft[-1], **pars)

                # Array of frequencies
                fs_gen = self.fs_fft
                # All frequencies are valid here since the waveform was already
                # conditioned, avoid multiplying with a step
                fmask = np.ones_like(self.fs_fft, dtype=bool)

        if ((not linear_free) and ((fs_out is None) or
                                   np.allclose(fs_out, self.fs_fft))):
            # Just return the waveform as is
            return wf_fd_padded
        else:
            # Find the `linear-free' point with the fiducial ASD and amplitude
            # which are independent of the detector and signal in the bank
            # Record the A(f)
            amp = np.abs(wf_fd_padded[fmask])
            # Go to self.fs_basis and remove linear component
            phases = np.interp(
                self.fs_basis, fs_gen[fmask],
                np.unwrap(np.angle(wf_fd_padded[fmask])))
            phases_nolin = remove_linear_component(
                np.array([phases]), self.fs_basis, self.wts)[0]

            if fs_out is None:
                # Default to self.fs_fft
                phases_nolin = np.interp(
                    self.fs_fft[fmask], self.fs_basis, phases_nolin)
                wf_fd_padded[fmask] = amp * np.exp(1j * phases_nolin)
            else:
                # Interpolate lwfs on to desired frequency grid, as in
                # upsample_lwfs
                phases_nolin = np.interp(fs_out, self.fs_basis, phases_nolin)
                amp = np.interp(fs_out, fs_gen[fmask], amp, left=0., right=0.)
                wf_fd_padded = amp * np.exp(1j * phases_nolin)

            return wf_fd_padded


    # Functions to generate time-domain waveforms
    # -------------------------------------------------------------------------
    def gen_wfs_td_from_fd(
            self, wfs_fd_original, dt_extra=0, whiten=False, highpass=True,
            truncate=False, target_snr=None, fs_in=None, log=False,
            **conditioning_kwargs):
        """
        Generates conditioned time-domain waveforms given frequency domain
        waveforms.
        :param wfs_fd_original:
            ... x fs_in array with frequency domain waveforms
            (can be a vector for n_wf = 1)
        :param dt_extra: Increase roll by this time for safety (s)
        :param whiten: Flag whether to return the whitened waveform
        :param highpass:
            Flag indicating whether to high-pass filter the waveform
        :param truncate:
            Flag indicating whether to truncate the waveform
            IMP: The wf needs to be linear-free before truncating
        :param target_snr:
            Target SNR to enforce, pass None to not apply this
            If this is a scalar, then all waveforms in wfs_fd have this SNR
            If this is an array of shape wfs_fd.shape[:-1], then the individual
            waveforms have the desired SNRs
            If whiten == True, this is the norm of the truncated and possibly
            highpassed whitened waveform
            If whiten == False and truncate == True or highpass == True, there
            are losses associated with conditioning
        :param fs_in:
            Array of input frequencies. None indicates fs_in = self.fs_fft
        :param log:
            Flag indicating whether we passed the log of the waveforms in
        :param conditioning_kwargs:
            If known, pass conditioning parameters (useful when whitening
            different waveforms), defaults to those in the bank
            Can use multiple normfacs if needed
        :return:
            ... x fftsize array of TD waveforms (can be vector for n_wf=1)
        """
        wfs_fd = wfs_fd_original.copy() # To avoid changing the original array
        
        if self.fftsize is None:
            raise RuntimeError("Set waveform conditioning parameters first!")

        # Read off conditioning arguments
        support = conditioning_kwargs.get('support', self.support_whitened_wf)
        shift = conditioning_kwargs.get('shift', self.shift_whitened_wf)

        if not ((fs_in is None) or np.allclose(self.fs_fft, fs_in)):
            # We need to interpolate to self.fs_fft
            if not log:
                wfs_fd = np.log(wfs_fd)
                log = True
            wfs_fd = upsample_lwfs(wfs_fd, fs_in, self.fs_fft)

        if log:
            wfs_fd = np.exp(wfs_fd)

        if whiten:
            wfs_fd *= self.wt_filter_fd * np.sqrt(2 * self.dt)
        elif target_snr is not None:
            # We're not whitening the waveforms, so enforce SNR in FD
            snrsq = compute_overlap(
                wfs_fd, wfs_fd, self.fs_fft, 1/self.wt_filter_fd)
            if wfs_fd.ndim > 1:
                snrsq = snrsq[..., None]
            wfs_fd /= np.sqrt(snrsq)
            # Can avoid the last multiplication if target_snr = 1
            if not np.all(np.asarray(target_snr) == 1):
                wfs_fd *= target_snr

        # Define aligned output and load the whitened waveforms into it
        wfs_td = utils.FFTIN(wfs_fd.shape[:-1] + (self.fftsize,))
        wfs_td[:] = utils.IRFFT(wfs_fd, self.fftsize, axis=-1)
        wfs_td[:] /= self.dt

        # Apply roll with safety, note self.shift_whitened_wf is negative
        extra_shift_inds = int(np.ceil(dt_extra / self.dt))
        shift_wf = shift - extra_shift_inds
        wfs_td_out = utils.FFTIN(wfs_fd.shape[:-1] + (self.fftsize,))
        wfs_td_out[:] = np.roll(wfs_td, shift=shift_wf, axis=-1)

        if highpass:
            # Remove low frequency components
            wfs_td_out[:] = signal.sosfiltfilt(
                self.sos, wfs_td_out, padlen=self.irl, axis=-1)

        if truncate:
            truncate_length = support + extra_shift_inds
            wfs_td_out[..., :-truncate_length] = 0

        if whiten and (target_snr is not None):
            # We're whitening the waveforms, so enforce SNR in TD
            wfs_td_out[:] /= np.linalg.norm(wfs_td_out, axis=-1)[..., None]
            # Can avoid the last multiplication if target_snr = 1
            if not np.all(np.asarray(target_snr) == 1):
                wfs_td_out *= target_snr

        return wfs_td_out

    # Using the bank
    # --------------
    def gen_wfs_fd_from_calpha(self, calpha=None, fs_out=None, orthogonalize=False,
         return_cov=False, log=False):
        """
        Generates waveforms from given coefficients
        Note: At fs_out <= min(fs_basis) or >= max(fs_basis), the output
        amplitudes are zero
        :param calpha:
            n_wf x n_(basis elements needed) array with list of coefficients
            (can be a vector for n_wf = 1). Defaults to the central waveform
        :param fs_out:
            Array of output frequencies. None indicates fs_out = fs_basis
        :param log: FLag to return the log of the waveform instead
        :param orthogonalize: Only works when fs_out is self.fs_fft
            Note: better to orthogonalize wfs after whitening, rather than at this stage
        :return:
            n_wf x n_modes x len(fs_out) complex array with waveforms at fs_out
            (can be 2D for n_wf=1)
        """
        if fs_out is None:
            fs_out = self.fs_basis

        # Generate 22 phases (n_wf x 3 x len(fs_out))
        phases = self.gen_phases_from_calpha(calpha=calpha, fs_out=fs_out)

        # Sample amplitudes to output frequency resolution
        # 3 x len(fs_out) with A(f)
        
        if self.amps_calpha_dependent:
            amps = self.gen_amps_from_calpha(calpha=calpha, fs_out=fs_out)
        
        else: 
            amps = np.array(
            [np.interp(fs_out, self.amp[:, 0], self.amp[:, i], left=0, right=0)
             for i in range(1, len(self.amp[0]))])

        # Return aligned memory
        h = utils.FFTIN(phases.shape, dtype=np.complex128)

        if log:
            h[:] = 1j * phases + np.log(amps)
        else:
            h[:] = amps * np.exp(1j * phases)
        
        if orthogonalize or return_cov:
            if not np.allclose(fs_out, self.fs_fft):
                raise Exception("Orthogonalization works only \
                                 when fs_out is self.fs_fft")
            weights = 4 * get_df(self.fs_fft) * self.wt_filter_fd**2
        
            h_ort = np.zeros_like(h)
            if h.ndim == 2: # For a single wf (with 3 modes)
                h_ort, CovMat = self.orthogonalize_wfs(wfs=h, weights=weights)
            else:
                CovMat = np.zeros((len(h),3,3), dtype = 'complex128')
                for i in range(len(h)):
                    h_ort[i], CovMat[i] = self.orthogonalize_wfs(
                                        wfs=h[i], weights=weights)
           
            if orthogonalize: h = h_ort
        
            if return_cov is True:
                return h, CovMat

        return h

    def gen_wfs_td_from_calpha(self, calpha=None, orthogonalize=True,
         return_cov=False, truncate=True, **td_kwargs):
        """
        Generates time-domain waveforms given basis coefficients
        OK for FFT test but might not be for injection due to wraparound and
        truncation losses
        :param calpha:
            n_wf x n_(basis elements needed) array with list of coefficients
            (can be vector for n_wf = 1). Defaults to the central waveform
        :param td_kwargs: Extra arguments to gen_wfs_td_from_fd
        :param orthogonalize: Orthogonalizes the different mode wfs
        :param return_cov: Returns the covariance matrix between mode wfs
        :return:
            n_wf x n_modes x fftsize array of TD waveforms
            (can be 2D for n_wf=1)
        """
        
        if return_cov is True:
            wfs_fd, CovMat = self.gen_wfs_fd_from_calpha(calpha=calpha,fs_out=self.fs_fft,
                orthogonalize=orthogonalize, return_cov=True)
            return self.gen_wfs_td_from_fd(wfs_fd, truncate=truncate, **td_kwargs), CovMat
        else:
            wfs_fd = self.gen_wfs_fd_from_calpha(calpha=calpha, fs_out=self.fs_fft,
                orthogonalize=orthogonalize, return_cov=False) 
            return self.gen_wfs_td_from_fd(wfs_fd, truncate=truncate, **td_kwargs)            
        
    def orthogonalize_wfs(self, wfs, weights):
        """
        Orthogonalizes the different higher modes in the wf
        (useful for later appropriately calculating inner product
         of data with diff modes)
        :param wfs: n_modes (=3) x n_freq_basis wf in the Fourier domain
        :param weight: n_freq_basis for weights for different freq bins
        :return:
            n_modes x len(fs_out) wf and upper triangular elements of 
            the covariance matrix
        """
        
        full_CovMat = np.zeros((3,3),dtype = 'complex128')
        
        for (j,k) in np.array(np.triu_indices(3)).T:
            full_CovMat[j,k] = np.sum( weights* wfs[j]* np.conj(wfs[k]), axis=-1)
            
        full_CovMat[np.tril_indices(3)] = full_CovMat.T.conj()[np.tril_indices(3)]
        
        L = np.linalg.cholesky( np.linalg.inv(full_CovMat[::-1,::-1]) )[::-1,::-1]
        # changing the order of elements because cholesky of the inverse keeps the last
        # dimension fixed, i.e., L[-1,-1] = 1.
        
        wfs = np.dot( L.conj().T, wfs)    
        
        # returning only upper triangular elements
        return wfs, full_CovMat
        
        # Transforming scores from perp to original
        
        # L = np.linalg.cholesky(full_CovMat)
        # Z_original =np.dot( L.conj(), Z_perp)
        # where Zp are the scores for the perp wfs
        

    # Functions to make whitened waveforms
    # -------------------------------------------------------------------------
    # Functions to deal with conditioning
    # -------------------------------------------------------------------------
    @staticmethod
    def get_waveform_conditioning(fftsize, dt, whitened_wf_fd):
        """
        Useful function to get conditioning parameters for whitened waveforms
        :param fftsize: Number of time-domain indices
        :param dt: Time domain step size (s)
        :param whitened_wf_fd:
            len(rfftfreq(fftsize)) array with FD whitened waveform
            IMP note: The wf should be linear free.
        :return: 1. Support of waveform in units of indices
                 2. Shift to put waveform with weight toward the right side
                 3. Normalization factor
        """
        # Find TD envelope of waveform
        whitened_wf_td_cos = utils.IRFFT(whitened_wf_fd, n=fftsize) / dt
        whitened_wf_td_sin = utils.IRFFT(whitened_wf_fd * 1j, n=fftsize) / dt
        envelope_sq = whitened_wf_td_cos ** 2 + whitened_wf_td_sin ** 2

        # Look at how weight builds up
        lind = 0
        rind = len(envelope_sq)
        totalwt = np.sum(envelope_sq)
        currentwt = 0
        while currentwt < (1. - params.WFAC_WF) * totalwt:
            dlwt = envelope_sq[lind]
            drwt = envelope_sq[rind - 1]
            if dlwt > drwt:
                lind += 1
                currentwt += dlwt
            else:
                rind -= 1
                currentwt += drwt
        tight_support_wf = fftsize - rind + lind

        # Minimum roll + safety factor
        shift_whitened_wf = - lind - int(
            params.REL_SHIFT_SAFETY * tight_support_wf)

        # Fix support
        min_support = int(params.MIN_WFDURATION / dt)
        # If fftsize - rind is much larger than lind, this is inspiral-dominated
        # Different safety factors since orbital hangup is more important for BH
        if fftsize - rind > params.MERGERTYPE_THRESH * lind:
            support_whitened_wf = max(
                int(params.SUPPORT_SAFETY_BNS * tight_support_wf), min_support)
        else:
            support_whitened_wf = max(
                int(params.SUPPORT_SAFETY_BH * tight_support_wf), min_support)

        # Check that the fftsize is long enough
        assert fftsize >= support_whitened_wf, "fftsize is too short!"
        # if support_whitened_wf > fftsize:
        #     raise RuntimeError("fftsize too short!")

        # Normalization factor
        normfac = (totalwt / 2.) ** 0.5

        return support_whitened_wf, shift_whitened_wf, normfac

    def set_waveform_conditioning(self, fftsize, dt, wt_filter_fd=None, 
                                    min_support=None):
        """
        Finds normalization factor, support, and shift for whitened waveforms
        in bank if they aren't already defined
        *NOTE* for PSD estimation, need:  fftsize * dt > chunktime
        :param fftsize: (int) Number of samples in FFT for the template
        :param dt: (float) Sampling interval of template (in seconds)
        :param wt_filter_fd:
            Frequency domain whitening filter (~irfft of 1/ASD). Lives on
            rfftfreq(fftsize, dt). If None, defaults to version in bank
        :return: Updates
                 1. self.fftsize
                 2. self.dt
                 3. self.wt_filter_fd
                 4. self.support_whitened_wf:
                        TD support of waveform (in units of dt). Hardcoded to
                        params.DEF_MAX_WFDURATION
                 5. self.shift_wf:
                        Shift applied for weight on the right of filter
                        (in units of dt)
                 6. self.normfac:
                        Normalization factor to divide waveform *
                        whitening filter by for convolution with whitened data
                 and filter parameters
        """
        # Define conjugate frequency grid
        fs_fft = np.fft.rfftfreq(fftsize, d=dt)

        # If we have nothing to go on, first check if we had something earlier
        if wt_filter_fd is None:
            if self.wt_filter_fd is not None:
                wt_filter_fd = self.wt_filter_fd

        if wt_filter_fd is None:
            # We really have nothing, use the fiducial PSD
            self.analytic_wt_filter = True
            # Generate whitening filter from the default ASD
            wt_filter_fd_unconditioned = 1. / self.asdf(fs_fft)
            wt_filter_fd, _, _ = utils.condition_filter(
                wt_filter_fd_unconditioned, truncate=True, flen=fftsize)
        else:
            # Ensure the whitening filter is defined on the right domain
            wt_filter_fd = utils.change_filter_times_fd(
                wt_filter_fd, 2 * (len(wt_filter_fd) - 1), fftsize)

        # Default to the (2, 2) waveform with the lowest chirp mass in the bank
        mcminind = np.argmin(transform_pars(self.pars)[:, 0])
        wf_fd = self.gen_wfs_fd_from_calpha(
            self.coeffs[mcminind], fs_out=fs_fft)[0]
        wf_whitened_fd = wf_fd * wt_filter_fd * np.sqrt(2 * dt)
        support_whitened_wf, shift_whitened_wf, normfac = \
            self.get_waveform_conditioning(fftsize, dt, wf_whitened_fd)
        
        # Check if the index provided in the metadata gives an even lower shift   
        mcminind = self.min_shift_calpha_ind
        wf_fd = self.gen_wfs_fd_from_calpha(
            self.coeffs[mcminind], fs_out=fs_fft)[0]
        wf_whitened_fd = wf_fd * wt_filter_fd * np.sqrt(2 * dt)
        support_whitened_wf, _, normfac = \
            self.get_waveform_conditioning(fftsize, dt, wf_whitened_fd)
        if (_ < shift_whitened_wf): shift_whitened_wf = _
        
        if min_support is not None:
            support_whitened_wf = max(support_whitened_wf, min_support)

        # Other class variables
        self.fftsize = fftsize
        self.dt = dt
        self.fs_fft = fs_fft
        self.wt_filter_fd = wt_filter_fd
        self.support_whitened_wf = support_whitened_wf
        self.shift_whitened_wf = shift_whitened_wf
        self.normfac = normfac
        
        # Scaling HM_amp_ratio_samples approximately with the measured whitening filter
        amps = np.array(
            [np.interp(fs_fft, self.amp[:, 0], self.amp[:, i], left=0, right=0)
             for i in range(1, len(self.amp[0]))])
        zz = np.sqrt(4.0 * np.sum(amps**2 /np.abs(
                                self.asdf(fs_fft))** 2 * np.diff(fs_fft)[0], axis=-1))
        ratios_scale = np.array([zz[1]/zz[0], zz[2]/zz[0]])
        zz = np.sqrt(4.0 * np.sum(amps**2 * np.abs(
                                self.wt_filter_fd)** 2 * np.diff(fs_fft)[0], axis=-1))
        ratios_scale = np.array([zz[1]/zz[0], zz[2]/zz[0]])/ratios_scale
        self.HM_amp_ratio_samples[:,:,:2] *= ratios_scale
        

        # Initialize highpass filter that will be applied on data (doesn't
        # know about fftsize)
        # self.num, self.den, self.irl = utils.band_filter(
        #     dt, fmin=params.FMIN_PSD)
        self.sos, self.irl = utils.band_filter(dt, fmin=params.FMIN_PSD)

        return

    # Using the bank
    # --------------
    def gen_whitened_wfs_td(
            self, calpha=None, wfs_fd=None, orthogonalize=True,
             return_cov=False, **conditioning_kwargs):
        """
        Generates whitened and conditioned time-domain waveforms given basis
        coefficients, or input waveforms
        :param calpha:
            n_wf x n_(basis elements needed) array with list of coefficients
            (can be vector for n_wf = 1). Defaults to the central waveform
        :param wfs_fd:
            Frequency domain waveform(s) on self.fs_fft to whiten instead,
            using saved conditioning parameters (preferred over calpha)
        :param conditioning_kwargs:
            If known, pass conditioning parameters (useful when whitening
            different waveforms), defaults to those in the bank
            Can use multiple normfacs if needed
        :param orthogonalize: Orthogonalizes the different mode wfs for calpha case
        :param return_cov: Returns the covariance matrix between mode wfs
        :return: n_wf x n_modes x fftsize array of whitened TD waveforms
                 (can be 2D for n_wf=1)
        """
        if wfs_fd is not None:
            wfs_td = self.gen_wfs_td_from_fd(
                wfs_fd, whiten=True, highpass=True, truncate=False, target_snr=1,
                **conditioning_kwargs)
            # Truncate can be set to true only if wfs_fd is linear free
        else:
            if calpha is None:
                calpha = self.bounds[:, 1]
                
            wfs_td = self.gen_wfs_td_from_calpha(
                    calpha=calpha, orthogonalize=False, whiten=True,
                    highpass=True, truncate=True, target_snr=1, **conditioning_kwargs)
                    
        if orthogonalize or return_cov:
            wfs = utils.RFFT(wfs_td, axis=-1)
            
            if wfs.ndim==2:
                wfs, CovMat = self.orthogonalize_wfs(
                        wfs=wfs, weights=np.ones_like(self.fs_fft)/len(self.fs_fft))
            else:
                CovMat = np.zeros((len(wfs),3,3), dtype = 'complex128')
                for i in range(len(wfs)):
                    wfs[i], CovMat = self.orthogonalize_wfs(
                        wfs=wfs[i], weights=np.ones_like(self.fs_fft)/len(self.fs_fft))
            if orthogonalize:
                wfs_td = utils.IRFFT(wfs, axis=-1)
                
            if return_cov:
                return(wfs_td, CovMat)

        return(wfs_td)
    
    def gen_boundary_whitened_wfs_td(self, ncalpha=3):
        """
        Generates boundary whitened and conditioned time-domain waveforms
        :param ncalpha: Number of calphas for which to explore bounds
        :return: 3**n_calpha * fftsize array containing boundary whitened TD
                 waveforms
        """
        niter = min(ncalpha, len(self.bounds))
        coeff_array = utils.FFTIN((3 ** niter, len(self.bounds)))
        coeff_array[:, :niter] = np.array(list(itertools.product(*self.bounds[:niter, :])))
        coeff_array[:, niter:] = self.bounds[niter:, 1]
        return self.gen_whitened_wfs_td(coeff_array)
        
    @staticmethod
    def split_whitened_wf_td(whitened_wf_td, fractions=None, nchunk=1):
        """
        Splits whitened time domain waveform into chunks with desired
        fractions of SNR^2, useful for vetoes
        :param whitened_wf_td: Whitened time domain waveform
        :param fractions: Array with fraction of SNR^2 in each chunk, we create
                          len(array) + 1 chunks (last chunk makes it up to 1)
        :param nchunk: Number of equal SNR^2 chunks to split into, used only
                       if fractions=None
        :return: nchunk x len(whitened_wf_td) array with time domain waveforms
        """
        if fractions is None:
            divs = np.arange(1, nchunk + 1) / nchunk
        else:
            cfracs = utils.FFTIN(len(fractions) + 1)
            cfracs[:-1] = fractions
            cfracs[-1] = 1 - np.sum(fractions)
            if cfracs[-1] < 0:
                raise RuntimeError("Can't create chunk with negative snr^2!")
            divs = np.cumsum(cfracs)
        # Add epsilon to the last one to ensure we get the last index
        divs[-1] += params.HOLE_EPS
        if len(divs) == 1:
            return whitened_wf_td
        else:
            wfnorm = np.cumsum(whitened_wf_td ** 2)
            inds = np.searchsorted(wfnorm, wfnorm[-1] * divs)
            split_whitened_wf_td = utils.FFTIN((len(divs), len(whitened_wf_td)))
            for i in range(len(divs)):
                if i == 0:
                    split_whitened_wf_td[0, 0:inds[0]] = \
                        whitened_wf_td[0:inds[0]]
                else:
                    split_whitened_wf_td[i, inds[i - 1]:inds[i]] = \
                        whitened_wf_td[inds[i - 1]:inds[i]]
            return split_whitened_wf_td

    # Generator for whitened waveforms
    def wt_waveform_generator(
            self, delta_calpha=0.7, fudge=params.TEMPLATE_SAFETY,
            remove_nonphysical=True, force_zero=True, ncores=1, coreidx=0,
            orthogonalize=True, return_cov=False, random_order=False):
        """
        Generator that returns waveforms in bank and corresponding c_alphas
        :param delta_calpha: Grid point distance in terms of basis coefficients
        :param fudge:
            Safety factor to inflate the range of used values of each parameter
        :param remove_nonphysical:
            Flag indicating whether we want to keep only the gridpoints that
            are close to a physical waveform
        :param force_zero:
            Flag indicating whether we want to center the grid to get as much
            overlap as possible with the central region
        :param ncores:
        :param coreidx:
        :param orthogonalize: Orthogonalizes the different mode wfs
        :param return_cov: Returns the covariance matrix between mode wfs
        :param random_order: Flag indicating whether we want the waveforms in a
                             random order
        :return: Generator that returns whitened frequency domain templates
                 (lives on rfftfreq(fftsize, dt)) in units of 1/Hz, and
                 coefficients of basis functions
        """
        coords_iterator = self.get_coeff_grid(
            delta_calpha, fudge, remove_nonphysical, force_zero)
        bank_size = len(coords_iterator)
        jmp = max((bank_size + ncores) // ncores, 1)
        ind = coreidx * jmp
        # Safe even if coreidx * jmp > bank_size
        coords_iterator = coords_iterator[ind: min(ind + jmp, bank_size)]

        if random_order:
            np.random.shuffle(coords_iterator)

        # Should we normalize by len(self.fs_fft)
        for coords in coords_iterator:
            trunc_wf_whitened_td = self.gen_whitened_wfs_td(
                    np.array(coords), orthogonalize=False)
            trunc_wf_whitened_fd = utils.RFFT(trunc_wf_whitened_td, axis=-1)
            
            if orthogonalize:
                trunc_wf_whitened_fd, CovMat = self.orthogonalize_wfs(
                        wfs=trunc_wf_whitened_fd, 
                        weights=np.ones_like(self.fs_fft)/len(self.fs_fft))
            
            if return_cov:
                yield  trunc_wf_whitened_fd, coords, CovMat
            else:
                yield  trunc_wf_whitened_fd, coords
                

    # Functions to help in making a discrete grid
    # --------------------------------------------------
    # Properties of grid points
    @staticmethod
    def grid_range(min_val, max_val, d_val, force_zero=True):
        """
        Returns a grid between min_val and max_val
        If force_zero=True:
            0 is a gridpoint
            Spacing is <= d_val/2 at the edges
            Spacing is <= d_val in the bulk
        If force_zero=False:
            Spacing is <= d_val/2 at the edges
            Spacing is exactly d_val in the bulk
        """
        if force_zero:
            # < 0 side
            dl = -min_val / (round(-min_val / d_val) + .5)
            left = np.arange(0, min_val, -dl)[1:][::-1]
            # > 0 side
            dr = max_val / (round(max_val / d_val) + .5)
            right = np.arange(0, max_val, dr)
            return np.r_[left, right]

        elif (max_val - min_val) > d_val:
            return np.arange((min_val + max_val - np.floor(
                (max_val - min_val) / d_val) * d_val) / 2., max_val, d_val)
        else:
            return np.array([(max_val + min_val) / 2])

    def make_grid_axes(
            self, delta_calpha, fudge=params.TEMPLATE_SAFETY, force_zero=True):
        """
        Return a list of arrays with the axes of a grid that covers
        all the physical waveforms used to generate the bank.
        :param delta_calpha: float, grid spacing
        :param fudge: A safety factor to inflate the range of used values of
                      each parameter
        """
        return [self.grid_range(b_min - (b_max - b_min) * (fudge - 1.) / 2,
                                b_max + (b_max - b_min) * (fudge - 1.) / 2,
                                delta_calpha, force_zero)
                for (b_min, b_mean, b_max) in self.bounds]

    def define_important_grid(
            self, delta_calpha, fudge=params.TEMPLATE_SAFETY, force_zero=True):
        # Define calpha grid
        grid_axes = self.make_grid_axes(
            delta_calpha, fudge=fudge, force_zero=force_zero)
        # Important dimensions to iterate over in the grid
        important_dims = [(i, x) for i, x in enumerate(grid_axes) if len(x) > 1]
        # Banks such as (16, 0) might have less than ndims -> Always allow = ndims
        while len(important_dims) < self.ndims:
            important_dims.append((len(important_dims),
                np.array([-delta_calpha, 0, delta_calpha])))
        # Index limit of important dimensions
        max_ind = int(max([i for i, x in important_dims])) + 1
        # Define grid spacings
        spacings = np.array(
            [max(np.r_[np.diff(grid_axes[ind]), 0]) for ind in range(max_ind)])
        return grid_axes, important_dims, spacings

    def ntemplates(self, delta_calpha, fudge=params.TEMPLATE_SAFETY,
                   remove_nonphysical=True, force_zero=True):
        """Convenience function to output number of templates in the bank"""
        grid_axes = self.make_grid_axes(delta_calpha=delta_calpha, fudge=fudge,
                                        force_zero=force_zero)
        if remove_nonphysical:
            return len(self.remove_nonphysical_templates(
                grid_axes, fudge - 1, delta_calpha=delta_calpha))
        else:
            return np.prod([len(x) for x in grid_axes])

    @staticmethod
    def get_grid_index(grid_axes, point):
        """
        Get the multiindex of the gridpoint that most closely
        matches a point. Assumes that the grid is rectangular, but not
        necessarily regular.
        :param grid_axes: list of arrays with the coordinates of the
                          gridpoints, one for each grid dimension.
        :param point: array with the coordinates of the point we want to match
        """
        point = np.array(point)
        if all(len(a) == 1 for a in grid_axes):
            grid_dims = 1  # Allow 1 even if the bank is a single template.
        else:
            grid_dims = max(np.nonzero([len(a)>1 for a in grid_axes])[0]) + 1
        # grid_dims = next((i for i, a in enumerate(grid_axes) if len(a) == 1),
        #                  len(grid_axes))  # Old, fails if there's a dim 1 before the last
        grid_axes = np.array(grid_axes[:grid_dims],dtype=object)
        indices = np.array([np.argmin(np.abs(a - p))
                            for a, p in zip(grid_axes, point)])
        return indices

    def remove_nonphysical_templates(
            self, grid_axes, rel_blob_extent=(params.TEMPLATE_SAFETY - 1),
            ret_keep=False, delta_calpha=None, coeffs=None):
        """
        Return the set of points from a rectangular grid in component
        space that are close to a physical waveform.
        Assumes that irrelevant c_alphas are zero (in the extra dimensions).
        :param grid_axes: list of arrays that define the grid, ordered
                          with decreasing number of components.
        :param rel_blob_extent: float, size of the blob of grid-space to
                                keep around each physical template,
                                relative to the total extent of the grid
                                in each dimension.
                                Should be rel_blob_extent <= fudge-1
        :param rel_blob_extent: Flag indicating whether to return the boolean
                                array indicating which gridpoints have been kept
        :returns: array with the components of the templates in the grid
                  that are close to some physical waveform
        """
        if delta_calpha is None:  # Guess it from grid_axes
            delta_calpha = max(grid_axes[0][1] - grid_axes[0][0],
                               grid_axes[0][-1] - grid_axes[0][-2])
        total_dims = len(grid_axes)
        if all(len(a) == 0 for a in grid_axes):
            return np.array([[0] * total_dims])

        # If we have one point, return it
        if all(len(a) == 1 for a in grid_axes):
            return np.array([[a[0] for a in grid_axes]])

        grid_dims = max(np.nonzero([len(a)>1 for a in grid_axes])[0]) + 1
        grid_axes = grid_axes[:grid_dims]
        if coeffs is None:
            coeffs = self.coeffs
        ips = coeffs[:, :grid_dims]
        abs_blob_extent = (rel_blob_extent / 2 / delta_calpha * np.array(
            [a[-1] - a[0] for a in grid_axes])).astype(int)
        blob = np.array(list(itertools.product(
            *[range(-ext, ext + 1) for ext in abs_blob_extent])))
        grid = np.meshgrid(*grid_axes, indexing='ij')
        occurrences = np.zeros_like(grid[0])
        for components in ips:
            # Don't want to add gridpoints in small dimensions
            # if the points are already closer than dca/2
            trimmed_components = components * (
                np.abs(components) > delta_calpha/2/(1+rel_blob_extent))
            indices = self.get_grid_index(grid_axes, trimmed_components)
            occurrences[tuple(indices)] += 1
        keep = np.full_like(grid[0], False, dtype='bool')
        keep[occurrences.nonzero()] = True
        # Add blobs in low-density areas
        for indices in np.array(np.where(occurrences == 1)).T:
            for neighbor in blob:
                if all(indices + neighbor >= 0):
                    try:
                        keep.itemset(tuple(indices + neighbor), True)
                    except IndexError:
                        pass
        phys_temps = np.pad(np.array([g[keep] for g in grid]).T,
                            ((0, 0), (0, total_dims - grid_dims)),
                            'constant', constant_values=0)
        if ret_keep:
            return phys_temps, keep
        else:
            return phys_temps

    def get_coeff_grid(
            self, delta_calpha=.7, fudge=params.TEMPLATE_SAFETY,
            remove_nonphysical=True, force_zero=True, ndims=None):
        """
        Return the set of points on which to place templates, an array of
        dimension ntemplates x ndims
        """
        grid_axes = self.make_grid_axes(delta_calpha, fudge, force_zero)

        if ndims is not None:
            grid_axes = grid_axes[:ndims]

        # Define iterator to return waveforms
        if remove_nonphysical:
            coeff_grid =  self.remove_nonphysical_templates(
                grid_axes, fudge - 1., delta_calpha=delta_calpha)
        else:
            coeff_grid =  np.array(list(itertools.product(*grid_axes)))
        
        # Predicting the calphas after ndims by random forest
        return np.c_[coeff_grid, self.randomforest.predict(coeff_grid)]

    # Functions to test bank quality
    # -------------------------------------------------------------------------
    def gen_physical_grid(self, delta_calpha, fudge, force_zero):
        """
        Generate a list of physical templates and save it
        inside a dictionary as a class attribute. The key of
        the dictionary is a tuple (delta_calpha, fudge, force_zero)
        """
        grid_axes = self.make_grid_axes(delta_calpha, fudge, force_zero)
        if all(len(a) == 1 for a in grid_axes):
            grid_dims = 1
        else:
            grid_dims = max(np.nonzero([len(a)>1 for a in grid_axes])[0]) + 1

        physical_grid = self.remove_nonphysical_templates(
            grid_axes, rel_blob_extent=fudge - 1,
            delta_calpha=delta_calpha)[:, :grid_dims]
        self.phys_grid.update(
            {(delta_calpha, fudge, force_zero): physical_grid})
        return physical_grid

    def is_physical(self, calpha, delta_calpha, fudge, force_zero,
                    override_physical_grid=None):
        """
        Checks if the given calphas are physical, i.e. that they are in
        the trimmed grid that remove_nonphysical_templates outputs
        
        Note that in order to return True the calphas must lie exactly on the
        grid generated by (delta_calpha, fudge, force_zero), to within a small
        tolerance ~ 1e-5 << delta_calpha.

        :param calpha: An array of template bank components, with shape either
                       (n_components) or (n_trial_points, n_components). 
        :param delta_calpha: Resolution of calphas
        :param fudge: A safety factor to inflate the range of used values of
                      each parameter
        :param force_zero: Flag indicating whether to force 0 to be a gridpoint
        :param override_physical_grid:
            Pass a precomputed physical grid if available, speeds up background
            collection on multiple files
        :returns: Bool if a single set of calphas is passed, or array of bools
                  of len(n_trial_points) if many are passed.
        """
        phys_grid = override_physical_grid

        if phys_grid is None:
            # No grid given, query the dictionary
            phys_grid = self.phys_grid.get(
                (delta_calpha, fudge, force_zero), None)

            if phys_grid is None:
                # Grid not present in dictionary, generate it
                phys_grid = self.gen_physical_grid(
                    delta_calpha, fudge, force_zero)

        calpha = np.array(calpha)
        if calpha.ndim == 1:  # If there is only one calpha set to try
            return np.any([
                np.allclose(calpha[:len(c)], c[:len(calpha)])
                for c in phys_grid])
        else:
            return np.array([np.any([
                np.allclose(c1[:len(c2)], c2[:len(c1)])
                for c2 in phys_grid])
                for c1 in calpha])

    @staticmethod
    def optimize_calpha(calpha, calpha_coarse, delta_calpha, fine_axis):
        """
        Return the calpha on the fine axis that:
          - was under the responsibility of calpha_coarse
          - is closest to calpha
        """
        # Points in fine_axis that are optimizable from calpha:
        fine_cands = fine_axis[np.abs(fine_axis-calpha_coarse) < delta_calpha/2]
        
        return fine_cands[np.argmin(np.abs(fine_cands - calpha))]

    @staticmethod
    def closest_coeff(coeffs, coeff_grid):
        """Return the values of the coefficients in coeff_grid that most
        closely match some input coefficients. Similar to rounding to the
        grid spacing, but accounts for offset and the grid needn't be uniform
        or rectangular, e.g. if nonphysical templates have been removed.
        :param coeffs:
            n_wfs x n_dims array with the coefficients of a number of waveforms
        :param coeff_grid:
            n_templates x n_dims array with the coefficients of the templates in
            the bank. Doesn't need to be a regular grid
        """
        inds = [np.argmin(np.linalg.norm(coeff - coeff_grid, axis=-1), axis=-1)
                for coeff in coeffs]
        return coeff_grid[inds]

    def test_faithfulness(
            self, wfs_fd, delta_calpha, HM=True, wfs_22_fd=None, n_sinc_interp=2,
            highpass=True, truncate=True, ret_overlaps_guess=False, coeff_grid=None,
            do_optimization=True, fudge=params.TEMPLATE_SAFETY,
            force_zero=True, fs_in=None, search_full_bank=False):
        """ Computes the mismatches between original astrophysical waveforms and
        the bank's representation of them. Used to measure losses in sensitivity
        :param wfs_fd:
            n_wf x fs_in array with LOG of frequency domain waveforms on fs_in,
            If using 2,2 modes only, ensure that the phases are unwrapped so
            that we can linearly interpolate
            If using 2,2 + higher modes, interpolation won't be accurate so
            provide them on self.fs_fft
        :param HM: Flag indicating whether we are using higher mode template
                   bank or 22 only
        :param wfs_22_fd:
            If HM is true, provide a 22 version of the full waveform.
            These 22 wfs are only used to find the best-fit calphas (ensure that the
             phases are unwrapped so that we can linearly interpolate)
            Format: n_wf x fs_in array with LOG of frequency domain waveforms on fs_in
        :param delta_calpha:
            Resolution of calphas used for coeff_grid
            (the finer grid will have 0.5 * delta_calpha)
        :param coeff_grid:
            n_templates x n_dims array with the coefficients of the templates
            that are in the bank
        :param n_sinc_interp: Number of times to sinc-interpolate the overlaps
        :param highpass:
            Flag indicating whether to high-pass filter the waveforms
        :param fs_in: Frequency grid for wfs_fd (None = self.fs_fft)
        :return if ret_overlaps_guess = False,
                    Array of size n_wf with maximum overlaps at current
                    resolution
                else:
                    Array of size n_wf x 2 with
                    1. Guess of maximum overlaps at current resolution
                    2. Maximum overlaps at current resolution
                without angle information
        """
        if search_full_bank is True:
            search_full_bank = 0.999
        if search_full_bank is False:
            search_full_bank = -1
        if fs_in is None:
            fs_in = self.fs_fft
            
        if HM is False:
            wfs_22_fd = wfs_fd.copy()
        
        wfs_fd = np.exp(wfs_fd) * self.wt_filter_fd * np.sqrt(2 * self.dt) # Whitening
        wfs_whitened_td = np.zeros((len(wfs_fd), self.fftsize))
        wfs_whitened_td = self.gen_wfs_td_from_fd(
                wfs_fd, whiten=False, highpass=highpass, truncate=False,
                fs_in=fs_in, target_snr=1)
        
        # Give waveforms an arbitrary (-8000) shift, is this needed?
        wfs_whitened_td = np.roll(wfs_whitened_td, shift=-8000, axis=-1)
        whitened_data_no_noise_fd = utils.RFFT(wfs_whitened_td, axis=-1)
            
        # Project the 22 waveforms into calpha space
        phases_22 = upsample_lwfs(
            wfs_22_fd, fs_in, self.fs_basis, phase_only=True)
        coeffs_exact = transform_basis(
            phases_22, self.avg_phase_evolution, self.basis, self.wts,
            self.fs_basis)
            
        if coeff_grid is None:
            coeff_grid = self.get_coeff_grid(delta_calpha=delta_calpha)

        # Find the coeffs that we will likely assign to the waveforms
        coeffs_on_grid = self.closest_coeff(coeffs_exact, coeff_grid)
        if do_optimization:
            fine_dca = delta_calpha / 2
            fine_axes = self.make_grid_axes(fine_dca, fudge, force_zero)

            # Give up vectorization in favor of sanity:
            fine_coeff = np.array(
                [[self.optimize_calpha(
                    calpha, calpha_coarse, delta_calpha, fine_axis)
                  for calpha, calpha_coarse, fine_axis
                  in zip(calphas, calphas_coarse, fine_axes)]  # Dims
                    for calphas, calphas_coarse in
                    zip(coeffs_exact, coeffs_on_grid)])  # Events
            fine_coeff =  np.c_[fine_coeff,
                             self.randomforest.predict(fine_coeff)] 
            coeffs_on_grid = np.where(
                (np.linalg.norm(coeffs_exact - fine_coeff, axis=-1)
                 < np.linalg.norm(coeffs_exact - coeffs_on_grid, axis=-1))[
                    ..., np.newaxis], fine_coeff, coeffs_on_grid)

        # Distance to the wf in calpha space
        overlaps_guess = 1 - np.linalg.norm(
            coeffs_on_grid - coeffs_exact, axis=-1)**2 / 2
        
        # Generate templates corresponding to best-fit calpha
        # n_wf x nmodes x len(rfftfreq(self.fftsize))
        wfs_fd = self.gen_wfs_fd_from_calpha(
                calpha=np.array(coeffs_on_grid), fs_out=self.fs_fft)
        
        trunc_wf_whitened_fd = np.zeros(
                        (len(wfs_fd), 3, len(self.fs_fft)), dtype=complex)
                
        for i in range(len(wfs_fd)):
            wfs_whitened_td = self.gen_wfs_td_from_fd(
                wfs_fd[i], whiten=True, highpass=highpass, truncate=truncate,
                fs_in=fs_in, target_snr=1)
            
            trunc_wf_whitened_fd[i], _ = self.orthogonalize_wfs(
                        wfs=utils.RFFT(wfs_whitened_td, axis=-1),
                        weights=np.ones_like(self.fs_fft)/len(self.fs_fft))
        
        if HM is True:
            trunc_wf_whitened_fd = np.moveaxis(trunc_wf_whitened_fd,0,1)
            # becomes n_modes x n_wf x len(rfftfreq(self.fftsize))
        else:
            trunc_wf_whitened_fd = trunc_wf_whitened_fd[:,0]
            # becomes n_wf x len(rfftfreq(self.fftsize))

        # Compute overlaps, n_wf x self.fftsize
        mf_cos = utils.IRFFT(
            trunc_wf_whitened_fd * whitened_data_no_noise_fd.conj(),
            n=self.fftsize, axis=-1)
        mf_sin = utils.IRFFT(
            -1j * trunc_wf_whitened_fd * whitened_data_no_noise_fd.conj(),
            n=self.fftsize, axis=-1)
        
        # Normalization factors
        a_fac = np.sqrt(np.sum(2 * np.abs(whitened_data_no_noise_fd) ** 2,
                               axis=-1) / self.fftsize)[:, np.newaxis]
                               
        overlaps_complex = (mf_cos + 1j * mf_sin) / a_fac

        # Remove angle information
        # n_wf x self.fftsize
        overlaps_sq = (mf_cos ** 2 + mf_sin ** 2) / a_fac ** 2
                       
        if HM:
            overlaps_sq = np.sum(overlaps_sq, axis=0)
        
        # n_wf
        maxoverlaps = np.sqrt(np.max(overlaps_sq, axis=-1))

        inds_full_search = np.where(maxoverlaps < search_full_bank)[0]
        if len(inds_full_search) > 0:
            # This needs to be rewritten for HM
            # Generate templates, n_wf x len(rfftfreq(self.fftsize))
            trunc_wf_whitened_td = self.gen_whitened_wfs_td(np.array(coeff_grid))
            trunc_wf_whitened_fd = utils.RFFT(trunc_wf_whitened_td, axis=-1)
            for iwf_in in inds_full_search:
                # Compute overlaps, n_templates x self.fftsize
                d = whitened_data_no_noise_fd[iwf_in]
                dmf_cos = utils.IRFFT(d * trunc_wf_whitened_fd.conj(),
                                      n=self.fftsize, axis=-1)
                dmf_sin = utils.IRFFT(-1j * d * trunc_wf_whitened_fd.conj(),
                                      n=self.fftsize, axis=-1)
                coverlaps = dmf_cos + 1j * dmf_sin
                # Remove angle information
                # n_templates x self.fftsize
                overlapsq = (dmf_cos ** 2 + dmf_sin ** 2) / a_fac[iwf_in] ** 2
                ibest = np.unravel_index(np.argmax(overlapsq), overlapsq.shape)
                maxoverlaps[iwf_in] = np.sqrt(overlapsq[ibest[0], ibest[1]])
                overlaps_complex[iwf_in] = coverlaps[ibest[0]].copy()

        if n_sinc_interp < 1:
            if ret_overlaps_guess:
                return np.c_[overlaps_guess, maxoverlaps]  # Prepend guess
            else:
                return maxoverlaps
        else:
            # Sinc interpolate overlaps around the best time
            inds = np.argmax(overlaps_sq, axis=-1)

            # loop over waveforms
            result = np.zeros((len(inds), 2))
            #inds_not_interpolated = []
            for i, ind in enumerate(inds):
                if ((ind < params.SUPPORT_EDGE_DATA) or
                        (ind + params.SUPPORT_EDGE_DATA > self.fftsize)):
                    # TODO: Do something so that we really estimate the
                    #  effectualness with sinc-interpolation
                    #print("Overlap too close to edge, cannot sinc-interpolate!")
                    #inds_not_interpolated.append([i, ind])
                    result[i, 0] = maxoverlaps[i]
                    result[i, 1] = maxoverlaps[i]
                else:
                    fake_t = np.arange(0, self.fftsize) * self.dt
                    nind = params.SUPPORT_EDGE_DATA
                    
                    if HM:
                        x_sq = np.zeros(2*nind)
                        # Summing over modes
                        for x in overlaps_complex[:,i, ind - nind:ind + nind]:
                            t = fake_t[ind - nind:ind + nind]
                            for j in range(n_sinc_interp):
                                t, x = utils.sinc_interp_by_factor_of_2(t, x)
                            # Best sinc-interpolated overlaps, without angle information
                            x_sq += (x.real ** 2 + x.imag ** 2)
                    else:
                        x = overlaps_complex[i, ind - nind:ind + nind]
                        t = fake_t[ind - nind:ind + nind]
                        for j in range(n_sinc_interp):
                            t, x = utils.sinc_interp_by_factor_of_2(t, x)
                        # Best sinc-interpolated overlaps, without angle information
                        x_sq = (x.real ** 2 + x.imag ** 2)

                    result[i, 0] = maxoverlaps[i]
                    result[i, 1] = np.sqrt(np.max(x_sq))
            # put this interpolation failure print at the end
            #print("Got warning 'Overlap too close to edge, cannot sinc-interpolate!'",
            #      '\nfor the following inds (row in maxoverlaps, best template ind):')
            #print(inds_not_interpolated)
            if ret_overlaps_guess:
                return np.c_[overlaps_guess, result]  # Prepend guess
            else:
                return result
                
    # Define frequency grid for relative binning
    def def_relative_bins(self, dcalphas, dt=params.DT_OPT, delta=0.1):
        """
        Returns bin edges at which we evaluate relative waveforms
        :param dcalphas:
            Array with calpha spacing in the first few desired directions
        :param dt: Time shift to allow
        :param delta: Phase allowed to accumulate within each bin
        :return: Array with bin edges within fs_basis
        """
        dcalphas = np.asarray(dcalphas)

        # Allowed phase functions due to mismatch in calphas and time
        calpha_phases = (self.basis[:len(dcalphas), :] / self.wts) * \
            dcalphas[:, np.newaxis]
        shift_phase = (2. * np.pi * self.fs_basis * dt)[np.newaxis, :]
        phase_functions = np.r_[calpha_phases, shift_phase]

        if np.min(np.abs(np.diff(phase_functions, axis=-1))) > delta:
            raise RuntimeError(
                "Delta requirement too stringent, the phase is not " +
                "defined precisely enough")

        # Go through the bins and identify where to break
        bin_edges = [0]
        for ind in range(1, len(self.fs_basis)):
            max_phase_shift = np.sum(
                np.abs(phase_functions[:, ind] -
                       phase_functions[:, bin_edges[-1]]))
            if (max_phase_shift > delta) or \
                    (ind == (len(self.fs_basis) - 1)):
                bin_edges.append(ind)

        return self.fs_basis[bin_edges]
        
    def modes_ratios_physical(self, Z, marginalized=False):
        """
        The scores obtained from triggering can sometimes be unphysical
        (e.g., |Z_33|>>|Z_22|)
        This function outputs the ratio of 33/22 and 44/22 in the physical space
        closest to the input waveform
        :param Z: array with n_triggers x [Z22, Z33, Z44] (Z are complex)
        :param marginalized: Marginalizing over inclination and mass ratio for HMs
        :return: if marginalized: return square of marginalized score
                 else: array of len 3 similar to Z array,
                         but with the closest physical point
        """
        
        [r_33_samples, r_44_samples, weights_samples] = self.HM_amp_ratio_samples[0].T
        
        def output_scores(Z, marginalized):
            z22, z33, z44 = np.abs(Z)
            # phase_term is basically arg(z44) + arg(z22) - 2*arg(z33)
            if z33 > 1e-3:
                phase_term = Z[2] * Z[0] / (Z[1]**2)
                phase_term = phase_term / abs(phase_term)
            else:
                phase_term = 1+0j
            z22_proj = (z22 + z33*r_33_samples + z44*r_44_samples)\
                            /(1 + r_33_samples**2 + r_44_samples**2)
            scores_sampled = z22_proj * (z22 + z33 * r_33_samples+\
                                z44 * r_44_samples * np.real(phase_term))\
                            - 0.5 * z22_proj**2 * (1+r_33_samples**2+r_44_samples**2)
            i_max = np.argmax(scores_sampled)
                     
            if not marginalized:
                output = z22_proj[i_max] * Z / np.abs(Z) * np.array(
                    [1, r_33_samples[i_max], np.conj(phase_term) * r_44_samples[i_max]])
            else:
                max_score = scores_sampled[i_max]
                scores_sampled -= max_score   
                output = 2 * max_score + 2 * np.log(np.sum(
                                     np.exp(scores_sampled) * weights_samples))
            return(output)
        
        if Z.ndim==1: # single trigger
            Z_out = output_scores(Z, marginalized)
        else:
            if marginalized:
                Z_out = np.zeros(len(Z))
                for i in range(len(Z)):
                    Z_out[i] = output_scores(Z[i], True)
            else:
                Z_out = np.zeros((len(Z),3), dtype=complex)
                for i in range(len(Z)):
                    Z_out[i] = output_scores(Z[i], False)                
                                 
        return(Z_out)

    
    def get_approximate_params(self, calpha):
        """
        Return a guess of m1, m2, s1z, s2z, l1, l2 from
        a template's coefficients
        We have currently only used 22, not HM (for which, breaking the q-Chieff
         degeneracy might help)
        """
        i_closest_wf = np.argmin(np.linalg.norm(
            calpha - self.all_coeffs[:, :len(calpha)], axis=-1))
        return self.all_pars[i_closest_wf]
    
    def ntemplates(self, delta_calpha, fudge=params.TEMPLATE_SAFETY,
                   remove_nonphysical=True, force_zero=True):
        """Convenience function to output number of templates in the bank"""
        grid_axes = self.make_grid_axes(delta_calpha=delta_calpha, fudge=fudge,
                                        force_zero=force_zero)
        if remove_nonphysical:
            return len(self.remove_nonphysical_templates(
                grid_axes, fudge - 1, delta_calpha=delta_calpha))
        else:
            return np.prod([len(x) for x in grid_axes])
            
    @staticmethod
    def transform_pars(pars):
        """
        Transforms [m1, m2, s1z, s2z, lambda1, lambda2] to commonly used
        parameters [mchirp, eta, chieff, chia, tilde{lambda},
        delta tilde{lambda}]
        :param pars:
            return value of gen_and_save_temp_structure
            (can be vector for n_sample = 1)
        :return: n_sample x 6 array with mchirp, eta, chieff, chia,
                 tilde{lambda} and delta tilde{lambda}
                 (can be vector for n_wf = 1)
        """
        pars = np.asarray(pars)
        pars2 = np.zeros(pars.shape)

        scf = lambda n:  tuple([slice(None)] * (pars.ndim - 1) + [n])

        pars2[scf(0)] = (pars[scf(0)] * pars[scf(1)]) ** 0.6 / \
                        (pars[scf(0)] + pars[scf(1)]) ** 0.2
        pars2[scf(1)] = (pars[scf(0)] * pars[scf(1)]) / \
                        (pars[scf(0)] + pars[scf(1)]) ** 2
        pars2[scf(2)] = \
            (pars[scf(0)] * pars[scf(2)] + pars[scf(1)] * pars[scf(3)]) / \
            (pars[scf(0)] + pars[scf(1)])
        pars2[scf(3)] = \
            (pars[scf(0)] * pars[scf(2)] - pars[scf(1)] * pars[scf(3)]) / \
            (pars[scf(0)] + pars[scf(1)])
        pars2[scf(4)] = \
            (8. / 13.0 *
             ((1. + 7. * pars2[scf(1)] - 31. * pars2[scf(1)] ** 2) *
              (pars[scf(4)] + pars[scf(5)]) -
              np.sqrt(1. - 4. * pars2[scf(1)]) *
              (1. + 9. * pars2[scf(1)] - 11. * pars2[scf(1)] ** 2) *
              (pars[scf(4)] - pars[scf(5)])))
        pars2[scf(5)] = \
            ((pars[scf(0)] - pars[scf(1)]) / (pars[scf(0)] + pars[scf(1)]) *
             pars2[scf(4)])

        return pars2
            
    def gen_random_wfs_td(self, n_wf=1, highpass=True):
        """
        Has not been updated by Jay.
        Generates random time-domain waveforms, OK for FFT test but not
        for injection due to wraparound
        :param n_wf: Number of random waveforms to generate
        :param highpass: Flag indicating whether to lowpass filter the waveform
        :return: n_wf x fftsize array of random TD waveforms
                 (vector for n_wf=1)
        """
        if n_wf == 1:
            rvs = np.random.random(len(self.bounds))
        else:
            rvs = np.random.random((n_wf, len(self.bounds)))
        random_calpha = self.bounds[:, 0] + (
                (self.bounds[:, -1] - self.bounds[:, 0]) * rvs)
        return self.gen_wfs_td_from_calpha(random_calpha, highpass=highpass)

    def gen_snr_degrade_dt(self, n_wf=100):
        """
        Has not been updated by Jay.
        Computes SNR degradation for a number of random waveforms shifted by a
        single TD index, used in deciding the time resolution to use for the
        matched filtering
        :param n_wf: Number of random waveforms
        :return: Degradation factor for n_wf random waveforms
        """
        random_wfs_td = self.gen_random_wfs_td(n_wf)
        random_wfs_td_shifted = np.roll(random_wfs_td, 1, axis=-1)

        random_wfs_fd = utils.RFFT(random_wfs_td)
        random_wfs_fd_shifted = utils.RFFT(random_wfs_td_shifted)

        snr_degrade = tg.compute_snr_efficiency(
            random_wfs_fd, random_wfs_fd_shifted, self.fs_fft,
            1. / self.wt_filter_fd)

        return snr_degrade

    def gen_phase_mismatch(self, nwf=None):
        """ 
        Has not been updated by Jay.
        Saves SNR degradation as a function of frequency for waveforms in
        bank. Note: Requires that bank was loaded with load_lwfs=True
        :param nwf: Number of waveforms to get degradation for
        :return: n_wf x len(fs_basis) array with weight * (angle - angle_calpha)
        """
        if nwf is not None:
            nwf = min(nwf, len(self.pars))
        angle_nolin = remove_linear_component(
            self.lwfs[:nwf, :].imag, self.fs_basis, self.wts)
        calpha = transform_basis(
            self.lwfs[:nwf, :].imag, self.avg_phase_evolution, self.basis,
            self.wts, self.fs_basis)
        angle_calpha = self.gen_phases_from_calpha(
            calpha, fs_out=self.fs_basis)
        mismatch = self.wts * (angle_calpha - angle_nolin)
        return mismatch


pass


class MultiBank(object):

    def __init__(self, subbanks, fftsize=None, dt=None, wt_filter_fd=None):
        """Initialize from list of subbanks
        # NOTE: All subbanks live on the same basis grid!
        :param subbanks: List of TemplateBank objects
        :param fftsize: Size of FFT for the template
        :param dt: Sampling interval of template (in seconds)
        :param wt_filter_fd: Frequency domain whitening filter (~irfft of 1/ASD)
                             Lives on rfftfreq(fftsize, dt). If None is passed,
                             defaults to version in bank
        """
        self.subbanks = subbanks

        self.fftsize = fftsize
        self.dt = dt
        self.wt_filter_fd = wt_filter_fd

        return
        
    @classmethod
    def from_json(cls, bank_dir, nsubbanks):

        multibank_dir = os.path.dirname(bank_dir)

        subbanks = []
        for i in range(nsubbanks):
            path = os.path.join('bank_'+str(i),'metadata.json')
            subbanks.append(
                TemplateBank.from_json(os.path.join(bank_dir, path)))

        instance = cls(subbanks)
        
        return instance
        
    def ntemplates(self, delta_calpha, fudge=params.TEMPLATE_SAFETY,
                   remove_nonphysical=True, force_zero=True):
        """Convenience function to output number of templates in the multibank
        """
        return sum(bank.ntemplates(
            delta_calpha, fudge, remove_nonphysical, force_zero)
                   for bank in self.subbanks)

# The code for template bank prior has not been included
pass
