import numpy as np
from numba import njit


@njit(fastmath=True)
def delay_read_linear_1d(buffer, size, write_idx, delay_samples):
    if delay_samples < 0.0:
        delay_samples = 0.0

    max_delay = size - 2.0

    if delay_samples > max_delay:
        delay_samples = max_delay

    read_idx = write_idx - delay_samples

    while read_idx < 0.0:
        read_idx += size

    while read_idx >= size:
        read_idx -= size

    i0 = int(np.floor(read_idx))
    i1 = i0 + 1

    if i1 >= size:
        i1 = 0

    frac = read_idx - i0

    return (1.0 - frac) * buffer[i0] + frac * buffer[i1]


@njit(fastmath=True)
def delay_read_linear_2d(buffers, row, size, write_idx, delay_samples):
    if delay_samples < 0.0:
        delay_samples = 0.0

    max_delay = size - 2.0

    if delay_samples > max_delay:
        delay_samples = max_delay

    read_idx = write_idx - delay_samples

    while read_idx < 0.0:
        read_idx += size

    while read_idx >= size:
        read_idx -= size

    i0 = int(np.floor(read_idx))
    i1 = i0 + 1

    if i1 >= size:
        i1 = 0

    frac = read_idx - i0

    return (1.0 - frac) * buffers[row, i0] + frac * buffers[row, i1]

@njit(fastmath=True)
def chorus_kernel(
    x,
    buffer,
    write_idx,
    phase,
    fs,
    rate_norm,
    depth_norm,
    mix_norm,
    delay_norm,
):
    n_samples = len(x)
    y = np.empty(n_samples, dtype=np.float32)

    size = len(buffer)

    rate = 0.05 + 2.5 * rate_norm
    base_delay = 0.008 + 0.020 * delay_norm
    depth = 0.002 + 0.010 * depth_norm

    wet_mix = 0.4 * mix_norm
    phase_inc = 2.0 * np.pi * rate / fs

    for n in range(n_samples):
        xn = x[n]

        lfo1 = 0.5 + 0.5 * np.sin(phase)
        lfo2 = 0.5 + 0.5 * np.sin(phase + 0.5 * np.pi)

        tau1 = (base_delay + depth * lfo1) * fs
        tau2 = (base_delay + depth * lfo2) * fs

        if tau1 < 1.0:
            tau1 = 1.0

        if tau2 < 1.0:
            tau2 = 1.0

        d1 = delay_read_linear_1d(buffer, size, write_idx, tau1)
        d2 = delay_read_linear_1d(buffer, size, write_idx, tau2)

        wet = 0.30 * (d1 + d2)

        y[n] = (1.0 - wet_mix) * xn + wet_mix * wet

        buffer[write_idx] = xn

        write_idx += 1
        if write_idx >= size:
            write_idx = 0

        phase += phase_inc
        if phase >= 2.0 * np.pi:
            phase -= 2.0 * np.pi

    return y, write_idx, phase

@njit(fastmath=True)
def phaser_coef_from_freq(freq, fs):
    if freq < 20.0:
        freq = 20.0

    max_freq = 0.45 * fs

    if freq > max_freq:
        freq = max_freq

    t = np.tan(np.pi * freq / fs)
    coef = (1.0 - t) / (1.0 + t)

    if coef > 0.999:
        coef = 0.999

    if coef < -0.999:
        coef = -0.999

    return coef


@njit(fastmath=True)
def phaser_kernel(
    x,
    apf_states,
    phase,
    fb_state,
    current_coef,
    control_counter,
    fs,
    rate_norm,
    depth_norm,
    feedback_norm,
    mix_norm,
    min_freq,
    max_freq,
    control_div,
):
    n_samples = len(x)
    y = np.empty(n_samples, dtype=np.float32)

    n_stages = len(apf_states)

    rate = 0.05 + 5.0 * rate_norm
    phase_inc = 2.0 * np.pi * rate / fs

    feedback = 0.45 * feedback_norm
    wet_mix = 0.5 * mix_norm

    log_min = np.log(min_freq)
    log_max = np.log(max_freq)

    for n in range(n_samples):
        xn = x[n]

        if control_counter <= 0:
            lfo = 0.5 + 0.5 * np.sin(phase)
            lfo = 0.5 + depth_norm * (lfo - 0.5)

            sweep_freq = np.exp(log_min + lfo * (log_max - log_min))
            current_coef = phaser_coef_from_freq(sweep_freq, fs)

            control_counter = control_div

        control_counter -= 1

        apf_input = xn + feedback * fb_state

        if apf_input > 2.0:
            apf_input = 2.0
        elif apf_input < -2.0:
            apf_input = -2.0

        wet = apf_input

        for s in range(n_stages):
            z = apf_states[s]

            # First-order all-pass:
            # y = -a*x + z
            # z = x + a*y
            apf_y = -current_coef * wet + z
            apf_states[s] = wet + current_coef * apf_y

            wet = apf_y

        if wet > 1.0:
            fb_state = 1.0
        elif wet < -1.0:
            fb_state = -1.0
        else:
            fb_state = wet

        y[n] = (1.0 - wet_mix) * xn + wet_mix * wet

        phase += phase_inc
        if phase >= 2.0 * np.pi:
            phase -= 2.0 * np.pi

    return y, phase, fb_state, current_coef, control_counter

@njit(fastmath=True)
def lite_reverb_kernel(
    x,
    comb_buffers,
    comb_sizes,
    comb_write_idxs,
    comb_delay_samples,
    comb_filter_states,
    apf_buffers,
    apf_sizes,
    apf_write_idxs,
    apf_delay_samples,
    apf_gains,
    room_norm,
    damping_norm,
):
    n_samples = len(x)
    y = np.empty(n_samples, dtype=np.float32)

    n_combs = len(comb_sizes)
    n_apfs = len(apf_sizes)

    feedback = 0.35 + 0.25 * room_norm
    damping = 0.25 + 0.60 * damping_norm

    for n in range(n_samples):
        xn = x[n]

        wet = 0.0

        for c in range(n_combs):
            size = comb_sizes[c]
            wi = comb_write_idxs[c]
            delay = comb_delay_samples[c]

            delayed = delay_read_linear_2d(
                comb_buffers,
                c,
                size,
                wi,
                delay,
            )

            filt_state = (
                (1.0 - damping) * delayed
                + damping * comb_filter_states[c]
            )

            comb_filter_states[c] = filt_state

            comb_buffers[c, wi] = xn + feedback * filt_state

            wi += 1
            if wi >= size:
                wi = 0

            comb_write_idxs[c] = wi

            wet += delayed

        wet = wet / n_combs

        for a in range(n_apfs):
            size = apf_sizes[a]
            wi = apf_write_idxs[a]
            delay = apf_delay_samples[a]
            g = apf_gains[a]

            delayed = delay_read_linear_2d(
                apf_buffers,
                a,
                size,
                wi,
                delay,
            )

            apf_y = -g * wet + delayed

            apf_buffers[a, wi] = wet + g * apf_y

            wi += 1
            if wi >= size:
                wi = 0

            apf_write_idxs[a] = wi

            wet = apf_y

        y[n] = 0.15 * wet

    return y

@njit(fastmath=True)
def fast_delay_kernel(
    x,
    buffer,
    write_idx,
    delay_samples,
    feedback,
    mix,
):
    n_samples = len(x)
    y = np.empty(n_samples, dtype=np.float32)

    size = len(buffer)

    for n in range(n_samples):
        xn = x[n]

        delayed = delay_read_linear_1d(
            buffer,
            size,
            write_idx,
            delay_samples,
        )

        yn = (1.0 - mix) * xn + mix * delayed

        write_sample = xn + feedback * delayed

        # Safety guard against runaway feedback.
        if write_sample > 2.0:
            write_sample = 2.0
        elif write_sample < -2.0:
            write_sample = -2.0

        buffer[write_idx] = write_sample

        write_idx += 1
        if write_idx >= size:
            write_idx = 0

        y[n] = yn

    return y, write_idx

@njit(fastmath=True)
def fast_flanger_kernel(
    x,
    buffer,
    write_idx,
    phase,
    fs,
    rate_norm,
    depth_norm,
    feedback_norm,
    mix_norm,
    delay_norm,
):
    n_samples = len(x)
    y = np.empty(n_samples, dtype=np.float32)

    size = len(buffer)

    rate = 0.05 + 5.0 * rate_norm

    base_delay = 0.001 + 0.005 * delay_norm
    depth = 0.0002 + 0.003 * depth_norm

    feedback = 0.30 * feedback_norm

    # Max 50% wet.
    mix = 0.5 * mix_norm

    phase_inc = 2.0 * np.pi * rate / fs

    for n in range(n_samples):
        xn = x[n]

        lfo = 0.5 + 0.5 * np.sin(phase)
        delay_samples = (base_delay + depth * lfo) * fs

        if delay_samples < 1.0:
            delay_samples = 1.0

        delayed = delay_read_linear_1d(
            buffer,
            size,
            write_idx,
            delay_samples,
        )

        yn = (1.0 - mix) * xn + mix * delayed

        write_sample = xn + feedback * delayed

        if write_sample > 2.0:
            write_sample = 2.0
        elif write_sample < -2.0:
            write_sample = -2.0

        buffer[write_idx] = write_sample

        write_idx += 1
        if write_idx >= size:
            write_idx = 0

        phase += phase_inc
        if phase >= 2.0 * np.pi:
            phase -= 2.0 * np.pi

        y[n] = yn

    return y, write_idx, phase