"""
Define class ``SearchCoherentScoreHMAS`` that computes the coherent
score (marginalized likelihood over extrinsic parameters) for
aligned-spin, (l,m) = [(2, 2), (3, 3), (4, 4)] waveforms.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

path_to_cogwheel = Path('/data/jayw/IAS/GW/cogwheel')
sys.path.append(path_to_cogwheel.as_posix())

import cogwheel.utils
from cogwheel.likelihood.marginalization.base import (BaseCoherentScoreHM,
                                                      MarginalizationInfoHM)


class SearchCoherentScoreHMAS(BaseCoherentScoreHM):
    """
    Class to marginalize the likelihood over extrinsic parameters,
    intended for the search pipeline.

    Assumptions:
        * Quasicircular
        * Aligned spins,
        * (l, m) = [(2, 2), (3, 3), (4, 4)] harmonics.
    """
    M_ARR = np.array([2, 3, 4])  # Also assume l = m

    def __init__(self, *, sky_dict, m_arr=M_ARR, lookup_table=None,
                 log2n_qmc: int = 11, nphi=128, seed=0,
                 beta_temperature=.1, n_qmc_sequences=128,
                 min_n_effective=50, max_log2n_qmc: int = 15):
        if not np.array_equal(m_arr, self.M_ARR):
            raise ValueError(f'`m_arr` must be {self.M_ARR} in this class.')

        super().__init__(m_arr=self.M_ARR,
                         sky_dict=sky_dict,
                         lookup_table=lookup_table,
                         log2n_qmc=log2n_qmc,
                         nphi=nphi,
                         seed=seed,
                         beta_temperature=beta_temperature,
                         n_qmc_sequences=n_qmc_sequences,
                         min_n_effective=min_n_effective,
                         max_log2n_qmc=max_log2n_qmc)

    @property
    def _qmc_range_dic(self):
        """
        Parameter ranges for the QMC sequence.
        The sequence explores the cumulatives of the single-detector
        (incoherent) likelihood of arrival times, the polarization, the
        fine (subpixel) time of arrival and the cosine inclination.
        """
        return super()._qmc_range_dic | {'cosiota': (-1, 1)}

    def _create_qmc_sequence(self):
        """
        Return a dictionary whose values are arrays corresponding to a
        Quasi Monte Carlo sequence that explores parameters per
        ``._qmc_range_dic``.
        The arrival time cumulatives are packed in a single entry
        'u_tdet'. An entry 'rot_psi' has the rotation matrices to
        transform the antenna factors between psi=0 and psi=psi_qmc.
        Also, entries for 'response' are provided. The response is defined
        so that:

          total_response :=
            := (1+cosiota**2)/2*fplus - 1j*cosiota*fcross
            = ((1+cosiota**2)/2, - 1j*cosiota) @ (fplus, fcross)
            = ((1+cosiota**2)/2, - 1j*cosiota) @ rot @ (fplus0, fcross0)
            = response @ (fplus0, fcross0)
        for the (2, 2) mode; the (3, 3) mode has an extra siniota; and
        the (4, 4) a siniota^2.
        """
        qmc_sequence = super()._create_qmc_sequence()
        siniota = np.sin(np.arccos(qmc_sequence['cosiota']))
        qmc_sequence['response'] = np.einsum(
            'Pq,qPp,qm->qpm',
            ((1 + qmc_sequence['cosiota']**2) / 2,
             - 1j * qmc_sequence['cosiota']),
            qmc_sequence['rot_psi'],
            np.power.outer(siniota, np.arange(3)))
        return qmc_sequence

    def get_marginalization_info(self, dh_mtd, hh_md, times,
                                 incoherent_lnprob_td, mode_ratios_qm):
        """
        Return a MarginalizationInfoHM object with extrinsic parameter
        integration results, ensuring that one of three conditions
        regarding the effective sample size holds:
            * n_effective >= .min_n_effective; or
            * n_qmc == 2 ** .max_log2n_qmc; or
            * n_effective = 0 (if the first proposal only gave
                               unphysical samples)
        """
        self.sky_dict.set_generators()  # For reproducible output

        # Resample to match sky_dict's dt:
        dh_mtd, _ = self.sky_dict.resample_timeseries(
            dh_mtd, times, axis=-2)
        t_arrival_lnprob, times = self.sky_dict.resample_timeseries(
            incoherent_lnprob_td.T, times, axis=-1)

        self.sky_dict.apply_tdet_prior(t_arrival_lnprob)
        t_arrival_prob = cogwheel.utils.exp_normalize(t_arrival_lnprob, axis=1)

        return self._get_marginalization_info(
            dh_mtd, hh_md, times, t_arrival_prob,
            mode_ratios_qm=mode_ratios_qm)

    def _get_marginalization_info_chunk(self, dh_mtd, hh_md, times,
                                        t_arrival_prob, i_chunk,
                                        mode_ratios_qm):
        q_inds = self._qmc_ind_chunks[i_chunk]  # Will update along the way
        n_qmc = len(q_inds)
        tdet_inds = self._get_tdet_inds(t_arrival_prob, q_inds)

        sky_inds, sky_prior, physical_mask \
            = self.sky_dict.get_sky_inds_and_prior(
                tdet_inds[1:] - tdet_inds[0])  # q, q, q

        # Apply physical mask (sensible time delays):
        q_inds = q_inds[physical_mask]
        tdet_inds = tdet_inds[:, physical_mask]

        if not any(physical_mask):
            return MarginalizationInfoHM(
                qmc_sequence_id=self._current_qmc_sequence_id,
                ln_numerators=np.array([]),
                q_inds=np.array([], int),
                o_inds=np.array([], int),
                sky_inds=np.array([], int),
                t_first_det=np.array([]),
                d_h=np.array([]),
                h_h=np.array([]),
                tdet_inds=tdet_inds,
                proposals_n_qmc=[n_qmc],
                proposals=[t_arrival_prob],
                flip_psi=np.array([], bool)
                )

        t_first_det = (times[tdet_inds[0]]
                       + self._qmc_sequence['t_fine'][q_inds])

        dh_qo, hh_qo = self._get_dh_hh_qo(sky_inds, q_inds, t_first_det,
                                          times, dh_mtd, hh_md, mode_ratios_qm)

        ln_numerators, important, flip_psi \
            = self._get_lnnumerators_important_flippsi(dh_qo, hh_qo, sky_prior)

        # Keep important samples (lnl above threshold):
        q_inds = q_inds[important[0]]
        sky_inds = sky_inds[important[0]]
        t_first_det = t_first_det[important[0]]
        tdet_inds = tdet_inds[:, important[0]]

        return MarginalizationInfoHM(
            qmc_sequence_id=self._current_qmc_sequence_id,
            ln_numerators=ln_numerators,
            q_inds=q_inds,
            o_inds=important[1],
            sky_inds=sky_inds,
            t_first_det=t_first_det,
            d_h=dh_qo[important],
            h_h=hh_qo[important],
            tdet_inds=tdet_inds,
            proposals_n_qmc=[n_qmc],
            proposals=[t_arrival_prob],
            flip_psi=flip_psi,
            )

    def lnlike_marginalized(self, dh_mtd, hh_md, times,
                            incoherent_lnprob_td, mode_ratios_qm):
        """
        Return log of marginalized likelihood over inclination, sky
        location, orbital phase, polarization, time of arrival and
        distance.

        Parameters
        ----------
        dh_mtd: (n_modes, n_times, n_det) complex array
            Timeseries of the inner product (d|h) between data and
            template, where the template is evaluated at a distance
            ``self.lookup_table.REFERENCE_DISTANCE`` Mpc.
            The convention in the inner product is that the second
            factor (i.e. h, not d) is conjugated.

        hh_md: (n_modes*(n_modes-1)/2, n_det) complex array
            Covariance between the different modes, i.e. (h_m|h_m'),
            with the off-diagonal entries (m != m') multiplied by 2.
            The ordering of the modes can be found with
            `self.m_arr[self.m_inds], self.m_arr[self.mprime_inds]`.
            The same template normalization and inner product convention
            as for ``dh_mtd`` apply.

        times: (n_times,) float array
            Times corresponding to the (d|h) timeseries (s).

        incoherent_lnprob_td: (n_times, n_det) float array
            Incoherent proposal for log probability of arrival times at
            each detector.

        mode_ratios_qm: (2**self.max_log2n_qmc, n_modes-1) float array
            Samples of mode amplitude ratio to the first mode (the part
            independent of inclination). These samples are used to
            marginalize over intrinsic parameters, mainly mass ratio.
        """
        marg_info = self.get_marginalization_info(
            dh_mtd, hh_md, times, incoherent_lnprob_td,
            mode_ratios_qm=mode_ratios_qm)
        return marg_info.lnl_marginalized

    def _get_dh_hh_qo(self, sky_inds, q_inds, t_first_det, times,
                      dh_mtd, hh_md, mode_ratios_qm):
        t_det = np.vstack((t_first_det,
                           t_first_det + self.sky_dict.delays[:, sky_inds]))
        dh_dmq = np.array(
            [self._interp_locally(times, dh_mtd[..., i_det], t_det[i_det])
             for i_det in range(len(self.sky_dict.detector_names))])

        # h_qdm = factor_qdm * h0_qm
        factor_qdm = (self.sky_dict.fplus_fcross_0[sky_inds, ]
                      @ self._qmc_sequence['response'][q_inds]
                      )  # qdp @ qpm -> qdm
        factor_qdm[..., 1:] *= mode_ratios_qm[q_inds, np.newaxis, :]

        dh_qm = np.einsum('dmq,qdm->qm', dh_dmq, factor_qdm.conj())  # qm
        hh_qm = np.einsum('md,qdm,qdm->qm',
                          hh_md,
                          factor_qdm[..., self.m_inds],
                          factor_qdm.conj()[..., self.mprime_inds])

        dh_qo = cogwheel.utils.real_matmul(dh_qm, self._dh_phasor)  # qo
        hh_qo = cogwheel.utils.real_matmul(hh_qm, self._hh_phasor)  # qo
        return dh_qo, hh_qo