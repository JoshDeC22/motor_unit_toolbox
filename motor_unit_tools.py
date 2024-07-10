import numpy as np
from scipy import signal
from copy import copy
import motor_unit_comparison as mu_comp
import itertools


def _check_mu_format(data):
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)
    return data


def get_discharge_rate(spike_train, timestamps):
    """Function to compute the discharge rate from motor unit firings in Hz.

    Args:
        spike_train (np.array): Binary array with 1s representing motor unit
            action potential firings
        timestamps (np.array): Timestamps of the firings

    Returns:
        float: mean discharge rate of a set of motor units in Hz
    """

    # Get number of motor units and initialise dr
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    dr = np.zeros(units)

    for unit in range(units):
        # Compute the total number of firings
        n_spikes = np.sum(spike_train[:, unit].astype(int))

        if n_spikes == 0:
            continue

        # Get firing times
        times_spikes = timestamps[spike_train[:, unit]]

        # Get total active period
        total_period = times_spikes[-1] - times_spikes[0]

        if total_period == 0:
            continue

        # Calculate interspike interval (ISI)
        isi = np.diff(times_spikes)

        # Find silent periods (i.e. where the ISI is larger than the minimum
        # motor unit discharge rate). This is 4 Hz according to "Negro F (2016)
        # Multi-channel intramuscular and surface EMG decomposition by
        # convolutive blind source separation."
        silent_period = np.sum(isi[isi > 0.25])

        # Calculate the actual active period of the motor unit
        active_period = total_period - silent_period

        # Return mean discharge rate
        dr[unit] = n_spikes / active_period

    return dr


def get_number_of_spikes(spike_train):
    """Function to compute the number of spikes.

    Args:
        spike_train (np.array): Binary array with 1s representing motor unit
            action potential firings

    Returns:
        int: number of spikes for each motor unit
    """

    # Compute the number of spikes
    n_spikes = np.sum(spike_train.astype(int), axis=0)

    return n_spikes


def get_inst_discharge_rate(spike_train, fs=2048):

    # Get number of motor units and initialise ints_DR
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    inst_dr = np.zeros(spike_train.shape)

    # Define hanning window
    dur = 1  # (s) for the moving average
    hann_win = np.hanning(np.round(dur * fs))

    for unit in range(units):
        # Convolve the hanning window and the binary spikes
        inst_dr[:, unit] = np.convolve(
            spike_train[:, unit], hann_win, mode='same'
            ) * 2

    return inst_dr


def get_coefficient_of_variation(spike_train, timestamps):

    # Get number of motor units and initialise cov
    spike_train = _check_mu_format(spike_train.astype(bool))
    units = spike_train.shape[-1]
    cov = np.zeros(units)
    cov[:] = np.nan

    for unit in range(units):

        # Get firing times
        if not np.any(spike_train[:, unit]):
            continue

        times_spikes = timestamps[spike_train[:, unit]]

        # Calculate interspike interval (isi)
        isi = np.diff(times_spikes)

        # Discard isi > 0.25 s (or discharge rate < 4 Hz) and isi < 0.02 s
        # (or discharge rate > 50 Hz) based on "Negro F (2016). Multi-
        # channel intramuscular and surface EMG decomposition by convolutive
        # blind source separation."
        isi = isi[isi < 0.25]

        # Calculate coefficient of variation
        cov[unit] = np.std(isi) / np.mean(isi)

    return cov * 100


def get_pulse_to_noise_ratio(spike_train, ips, ext_fact=8):

    # Get number of motor units and initialise PNR
    spike_train = _check_mu_format(spike_train.astype(bool))
    ips = _check_mu_format(ips)
    units = spike_train.shape[-1]
    pnr = np.zeros(units)
    pnr[:] = np.nan

    # Square IPTs
    ipts2 = ips ** 2

    for unit in range(units):
        # Get the spikes and baseline indexes, discarding the extension factor
        spikes_idx = np.nonzero(spike_train[:, unit])[0]
        spikes_idx = spikes_idx[np.greater_equal(spikes_idx, ext_fact + 1)]

        if not np.any(spikes_idx):
            continue

        spikes = ipts2[spikes_idx, unit]
        min_spikes_amp = np.amin(spikes)

        # Get baseline peaks with amplitude lower than the lowest spike
        baseline_peaks_idx, _ = signal.find_peaks(
            ipts2[:, unit], height=(0, min_spikes_amp)
            )
        if not np.any(baseline_peaks_idx):
            baseline_peaks_idx = np.nonzero(
                np.logical_not(spike_train[:, unit].astype(bool))
                )[0]
        baseline_peaks_idx = baseline_peaks_idx[
            np.greater_equal(baseline_peaks_idx, ext_fact + 1)
            ]
        baseline = ipts2[baseline_peaks_idx, unit]

        if len(spikes) == 0:
            continue

        # Compute spikes and baseline mean values
        spikes_mean = np.mean(spikes)
        baseline_mean = np.mean(baseline)

        # Compute PNR
        pnr[unit] = 20 * np.log10(spikes_mean / baseline_mean)

    return pnr


def get_silhouette_measure(spike_train, ipts, ext_fact=8):

    # Get number of motor units and initialise SIL
    spike_train = _check_mu_format(spike_train.astype(bool))
    ipts = _check_mu_format(ipts)
    units = spike_train.shape[-1]
    sil = np.zeros(units)
    sil[:] = np.nan

    # Square IPTs
    ipts2 = ipts ** 2

    for unit in range(units):
        # Get the spikes and baseline indexes, discarding the extension factor
        spikes_idx = np.nonzero(spike_train[:, unit])[0]
        spikes_idx = spikes_idx[np.greater_equal(spikes_idx, ext_fact + 1)]

        if not np.any(spikes_idx):
            continue

        spikes_amp = ipts2[spikes_idx, unit]
        min_spikes_amp = np.amin(spikes_amp)

        # Get baseline peaks with amplitude lower than the lowest spike
        baseline_peaks_idx, _ = signal.find_peaks(
            ipts2[:, unit], height=(0, min_spikes_amp)
            )
        if not np.any(baseline_peaks_idx):
            baseline_peaks_idx = np.nonzero(
                np.logical_not(spike_train[:, unit].astype(bool))
                )[0]
        baseline_peaks_idx = baseline_peaks_idx[
            np.greater_equal(baseline_peaks_idx, ext_fact + 1)
            ]
        baseline_amp = ipts2[baseline_peaks_idx, unit]

        # Compute spikes and baseline mean values
        spikes_mean = np.mean(spikes_amp)
        baseline_mean = np.mean(baseline_amp)

        # Compute distances
        dist_sum_spikes = np.sum(np.power((spikes_amp - spikes_mean), 2))
        dist_sum_baseline = np.sum(np.power((spikes_amp - baseline_mean), 2))

        # Compute SIL
        max_dist = np.amax([dist_sum_spikes, dist_sum_baseline])
        if max_dist == 0:
            sil[unit] = 0
        else:
            sil[unit] = (dist_sum_baseline - dist_sum_spikes) / max_dist

    return sil


def get_spike_baseline_amp(spike_train, ipts, ext_fact=8):

    # Get number of motor units and initialise output variables
    spike_train = _check_mu_format(spike_train.astype(bool))
    ipts = _check_mu_format(ipts)
    units = spike_train.shape[-1]
    spikes_amp = np.zeros(units)
    spikes_amp[:] = np.nan
    base_amp = np.zeros(units)
    base_amp[:] = np.nan

    for unit in range(units):
        # Get the spikes indexes discarding the extension factor
        spikes_idx = np.nonzero(spike_train[:, unit])[0]
        spikes_idx = spikes_idx[np.greater_equal(spikes_idx, ext_fact + 1)]

        if np.any(spikes_idx):
            spikes_amp[unit] = np.median(ipts[spikes_idx, unit])

        # Get the baseline indexes discarding the extension factor
        baseline_peaks_idx = np.nonzero(
            np.logical_not(spike_train[:, unit].astype(bool))
            )[0]
        baseline_peaks_idx = baseline_peaks_idx[
            np.greater_equal(baseline_peaks_idx, ext_fact + 1)
            ]

        if np.any(baseline_peaks_idx):
            base_amp[unit] = np.median(ipts[baseline_peaks_idx, unit])

    return spikes_amp, base_amp


def find_reliable_units(
        dr,
        cov,
        sil,
        pnr,
        dr_low_thr=3,
        dr_upp_thr=40,
        cov_thr=40,
        sil_thr=0.9,
        pnr_thr=30
        ):
    aux = np.vstack((
        dr >= dr_low_thr,
        dr <= dr_upp_thr,
        cov <= cov_thr,
        sil >= sil_thr,
        pnr >= pnr_thr
        ))
    reliable_units = np.all(aux, axis=0)
    return reliable_units


def get_muaps(spike_trains, SIG, fs, win_ms=25):

    # Initialise dimensions
    spike_trains = _check_mu_format(spike_trains.astype(bool))
    rows, cols, samples = SIG.shape
    half_win = round(win_ms/2/1000*fs)

    # Check spike train dimensions
    if len(spike_trains) > 0:
        if len(spike_trains.shape) == 1:
            spike_trains = np.expand_dims(spike_trains, axis=-1)

        # Initialise units and muaps
        units = spike_trains.shape[1]
        muaps = np.empty((units, rows, cols, half_win * 2))

        for unit in range(units):
            # Get the firings that fit a window around them
            unit_firings = np.nonzero(spike_trains[:, unit])[0]
            unit_firings = unit_firings[
                (unit_firings - half_win >= 0) &
                (unit_firings + half_win <= samples-1)
                ]
            n_unit_firings = len(unit_firings)

            # Initialise muap samples
            muaps_aux = np.empty((n_unit_firings, rows, cols, half_win*2))

            # Get all the muap samples for each unit firing
            for i, firing in enumerate(unit_firings):

                mask = np.arange(half_win * 2) - half_win + firing
                muaps_aux[i] = SIG[:, :, mask]

            # Compute mean
            muaps[unit] = np.mean(muaps_aux, axis=0)
    else:
        muaps = np.empty((0, rows, cols, half_win*2))

    return muaps


def center_muaps(muaps):

    # Check muaps dimensions
    if len(muaps) > 0:
        # There are units
        if len(muaps.shape) < 4:
            muaps = np.expand_dims(muaps, axis=0)

        # Initialise variables
        units, rows, cols, samples = muaps.shape
        center_sample = samples//2
        centered_muaps = copy(muaps)

        for unit in range(units):
            # Get current muap and peak amplitude channel
            muap = muaps[unit]
            ch_row, ch_col = np.unravel_index(
                np.nanargmax(np.abs(muap), axis=-1), (rows, cols)
                )

            #  Get the sample at which the amplitude is max
            max_sample = np.nanargmax(np.abs(muap[ch_row, ch_col]))

            # Center muap
            centered_muaps[unit] = np.roll(
                centered_muaps[unit], center_sample-max_sample, axis=-1
                )
    else:
        #  No units
        centered_muaps = copy(muaps)

    return centered_muaps


def get_muap_waveform_length(muaps, sel_chs_by="iqr"):
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)
    units, rows, cols, samples = muaps.shape

    if sel_chs_by is None:
        # Compute muap length for each muap channel
        wl = np.sum(np.abs(np.diff(muaps, axis=-1)), axis=-1)
    else:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": mu_comp.get_highest_iqr_ch,
            "iqr_ptp": mu_comp.get_highest_iqr_ptp_ch,
            "max_amp": mu_comp.get_highest_amp_ch,
            "ptp": mu_comp.get_highest_ptp_ch
        }

        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Initialise muap length
        wl = np.empty(muaps.shape[0:3])
        wl[:] = np.nan

        # Compute muap length
        for unit in range(units):
            muap_sel_chs = muaps[unit][sel_chs_mask]
            wl[unit][sel_chs_mask] = np.sum(np.abs(
                np.diff(muap_sel_chs, axis=-1)
                ), axis=-1)

    return wl


def get_muap_energy(muaps, sel_chs_by="iqr"):
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)
    units, rows, cols, samples = muaps.shape

    if sel_chs_by is None:
        # Compute muap energy for each muap channel
        energy = np.sum(np.power(muaps, 2), axis=-1)
    else:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": mu_comp.get_highest_iqr_ch,
            "iqr_ptp": mu_comp.get_highest_iqr_ptp_ch,
            "max_amp": mu_comp.get_highest_amp_ch,
            "ptp": mu_comp.get_highest_ptp_ch
        }
        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Initialise muap energy
        energy = np.empty(muaps.shape[0:3])
        energy[:] = np.nan

        # Compute muap energy
        for unit in range(units):
            muap_sel_chs = muaps[unit][sel_chs_mask]
            energy[unit][sel_chs_mask] = np.sum(
                np.power(muap_sel_chs, 2), axis=-1
                )

    return energy


def get_muap_ptp(muaps, sel_chs_by="iqr"):
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)
    units, rows, cols, samples = muaps.shape

    if sel_chs_by is None:
        # Compute muap peak to peak amplitude for each muap channel
        ptp = np.ptp(muaps, axis=-1)
    else:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": mu_comp.get_highest_iqr_ch,
            "iqr_ptp": mu_comp.get_highest_iqr_ptp_ch,
            "max_amp": mu_comp.get_highest_amp_ch,
            "ptp": mu_comp.get_highest_ptp_ch
        }
        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Initialise muap peak to peak amplitude
        ptp = np.empty(muaps.shape[0:3])
        ptp[:] = np.nan

        # Compute muap peak to peak amplitude
        for unit in range(units):
            muap_sel_chs = muaps[unit][sel_chs_mask]
            ptp[unit][sel_chs_mask] = np.ptp(muap_sel_chs, axis=-1)

    return ptp


def get_muap_ptp_time(muaps, sel_chs_by="iqr", fs=2048):
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)
    units, rows, cols, samples = muaps.shape

    if sel_chs_by is None:
        # Compute muap peak to peak time (in samples) for each muap channel
        ptp_time = np.abs(
            np.argmax(muaps, axis=-1) - np.argmin(muaps, axis=-1)
            )
    else:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": mu_comp.get_highest_iqr_ch,
            "iqr_ptp": mu_comp.get_highest_iqr_ptp_ch,
            "max_amp": mu_comp.get_highest_amp_ch,
            "ptp": mu_comp.get_highest_ptp_ch
        }
        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Initialise muap peak to peak time (in samples)
        ptp_time = np.empty(muaps.shape[0:3])
        ptp_time[:] = np.nan

        # Compute muap peak to peak time (in samples)
        for unit in range(units):
            muap_sel_chs = muaps[unit][sel_chs_mask]
            ptp_time[unit][sel_chs_mask] = np.abs(
                np.argmax(muap_sel_chs, axis=-1) -
                np.argmin(muap_sel_chs, axis=-1)
                )

    return ptp_time


def get_muap_peak_frequency(muaps, sel_chs_by="iqr", fs=2048):
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.abs(np.expand_dims(muaps, axis=0))

    # Compute power spectrum for each muap channel
    units, rows, cols, samples = muaps.shape
    ps = np.power(
        np.abs(np.fft.fft(muaps, n=samples, axis=-1).real), 2
        )[:, :, :, :samples//2]
    freq = np.fft.fftfreq(n=samples, d=1/fs)[:samples//2]
    max_ps_idx = np.argmax(ps, axis=-1)

    if sel_chs_by is not None:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": mu_comp.get_highest_iqr_ch,
            "iqr_ptp": mu_comp.get_highest_iqr_ptp_ch,
            "max_amp": mu_comp.get_highest_amp_ch,
            "ptp": mu_comp.get_highest_ptp_ch
        }

        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)
    else:
        sel_chs_mask = np.ones((muaps.shape[1:3])).astype(bool)

    # Initialise peak frequency
    peak_freq = np.empty(muaps.shape[0:3])
    peak_freq[:] = np.nan
    for unit, row, col in itertools.product(
            range(units), range(rows), range(cols)
            ):
        if sel_chs_mask[row, col] is False:
            continue
        peak_freq[unit, row, col] = freq[max_ps_idx[unit, row, col]]

    return peak_freq


def get_muap_median_frequency(muaps, sel_chs_by="iqr", fs=2048):
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.abs(np.expand_dims(muaps, axis=0))

    # Compute power spectrum for each muap channel
    units, rows, cols, samples = muaps.shape
    ps = np.power(
        np.abs(np.fft.fft(muaps, n=samples, axis=-1).real), 2
        )[:, :, :, :samples//2]
    freq = np.fft.fftfreq(n=samples, d=1/fs)[:samples//2]

    # Compute peak frequency
    cum_ps = np.cumsum(ps, axis=-1)
    med_ps = np.sum(ps, axis=-1)/2

    if sel_chs_by is not None:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": mu_comp.get_highest_iqr_ch,
            "iqr_ptp": mu_comp.get_highest_iqr_ptp_ch,
            "max_amp": mu_comp.get_highest_amp_ch,
            "ptp": mu_comp.get_highest_ptp_ch
        }

        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)
    else:
        sel_chs_mask = np.ones((muaps.shape[1:3])).astype(bool)

    # Initialise peak frequency
    med_freq = np.empty(muaps.shape[0:3])
    med_freq[:] = np.nan
    for unit, row, col in itertools.product(
            range(units), range(rows), range(cols)
            ):
        if sel_chs_mask[row, col] is False:
            continue
        med_freq[unit, row, col] = freq[np.argmin(np.abs(
            cum_ps[unit, row, col] -
            med_ps[unit, row, col]
            ))]
    return med_freq


def get_muap_mean_frequency(muaps, sel_chs_by="iqr", fs=2048):
    # Check muaps size
    if len(muaps.shape) < 4:
        muaps = np.abs(np.expand_dims(muaps, axis=0))

    # Compute power spectrum for each muap channel
    units, rows, cols, samples = muaps.shape
    ps = np.power(
            np.abs(np.fft.fft(muaps, n=samples, axis=-1).real), 2
            )[:, :, :, :samples//2]
    freq = np.fft.fftfreq(n=samples, d=1/fs)[:samples//2]

    # Compute peak frequency
    mean_freq = np.sum(ps * freq, axis=-1)/np.sum(ps, axis=-1)

    if sel_chs_by is not None:
        # Build channel selection
        sel_chs_mask_units = np.zeros(muaps.shape[0:3]).astype(bool)
        sel_chs_fun = {
            "iqr": mu_comp.get_highest_iqr_ch,
            "iqr_ptp": mu_comp.get_highest_iqr_ptp_ch,
            "max_amp": mu_comp.get_highest_amp_ch,
            "ptp": mu_comp.get_highest_ptp_ch
        }

        # Apply channel selection
        for unit in range(units):
            sel_chs_mask_units[unit] = sel_chs_fun[sel_chs_by](muaps[unit])
        sel_chs_mask = np.all(sel_chs_mask_units, axis=0)
        if not np.any(sel_chs_mask):
            sel_chs_mask = np.ones_like(sel_chs_mask).astype(bool)

        # Discard non selected channels
        for unit in range(units):
            mean_freq[unit][np.logical_not(sel_chs_mask)] = np.nan

    return mean_freq
