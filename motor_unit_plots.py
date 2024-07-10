import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from motor_unit_comparison import (
    get_percentile_ch,
    get_highest_amp_ch,
    get_highest_ptp_ch,
    get_highest_iqr_ch,
    get_highest_iqr_ptp_ch
)
import seaborn as sns
import numpy as np
from copy import copy
import math


def plot_spike_trains(
        firings,
        timestamps,
        fs=2048,
        sorted=False,
        ax=None,
        palette_name="viridis",
        offset=1.5,
        ylabel_offset=1,
        ):

    if ax is None:
        fig, ax = plt.subplots()

    color_palette = sns.color_palette(palette_name, len(firings))

    n_units = len(firings)

    # Plot based on recruitment order based on sorted flag
    sorted_idx = np.arange(0, n_units).astype(int)
    if sorted is True:
        first_firings = np.array([firings[unit][0] for unit in range(n_units)])
        sorted_idx = np.argsort(first_firings)

    ax.eventplot(
        firings[sorted_idx]/fs + timestamps[0],
        orientation='horizontal',
        colors=color_palette,
        ineoffsets=offset
    )
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_yticks(
        np.arange(0, n_units * offset, offset*ylabel_offset),
        sorted_idx.astype(int)[::ylabel_offset]
    )
    ax.set_ylabel('Motor unit indexes')
    ax.set_xlabel('Time (s)')
    ax.set_ylim([-offset, n_units * offset + offset])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


def plot_muaps(
        muaps,
        fs,
        ax=None,
        color=None,
        palette_name="viridis",
        normalize=False,
        ch_framed="max_amp",
        ):

    # Check muap dimensions
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)

    # Normalise muaps for plots
    muaps_plot = copy(muaps)
    if normalize:
        max_amp = np.nanmax(np.abs(muaps), axis=(1, 2, 3))
        muaps_plot /= max_amp[:, None, None, None]
    else:
        max_amp = np.nanmax(np.abs(muaps))
        muaps_plot /= max_amp

    #  Initialise variables
    n_units, rows, cols, samples = muaps_plot.shape
    x_offset = round(muaps_plot.shape[-1]/10)
    y_offset = np.nanmax(np.abs(muaps_plot)) * 2.01
    x = np.arange(samples)

    # Define axes if not initialised
    if ax is None:
        fig, ax = plt.subplots(n_units, 1, figsize=(20, 5*n_units))
        ax = np.ravel(ax)

    # Create color palette
    color_palette = sns.color_palette(palette_name, n_units)

    # Plot muaps_plot
    for unit in range(n_units):

        # Get highest amplitude channels
        if ch_framed == 'max_amp':
            curr_amp_ch = get_highest_amp_ch(muaps_plot[unit])
        elif ch_framed == 'ptp':
            curr_amp_ch = get_highest_ptp_ch(muaps_plot[unit])
        elif ch_framed == 'iqr':
            curr_amp_ch = get_highest_iqr_ch(muaps_plot[unit])
        elif ch_framed == 'iqr_ptp':
            curr_amp_ch = get_highest_iqr_ptp_ch(muaps_plot[unit])
        elif ch_framed == 'per':
            curr_amp_ch = get_percentile_ch(muaps_plot[unit], thr=95)

        for col in range(cols):
            # Apply offset to signals
            curr_muap_col = muaps_plot[unit, :, col].T
            curr_muap_col -= range(0, rows) * y_offset

            # Current x
            curr_x = x + (samples + x_offset) * col

            # Plot column
            ax[unit].plot(
                curr_x,
                curr_muap_col,
                color=color_palette[unit],
                linewidth=1
                )

            # Plot frames
            if ch_framed is None:
                continue
            frames = []
            for row in range(rows):
                if curr_amp_ch[row, col] is False:
                    continue
                frames.append(Rectangle(
                    (curr_x[0] - x_offset/2, - y_offset * row - y_offset/2),
                    width=samples + x_offset,
                    height=y_offset
                    ))
            frame_collection = PatchCollection(
                frames,
                ls='-',
                ec="lightgrey",
                fc="none",
                lw=1
                )
            ax[unit].add_collection(frame_collection)

        # Add time reference
        time_y_ref = - y_offset * (rows) - y_offset/10
        ax[unit].plot(
            [0, samples],
            [time_y_ref, time_y_ref],
            '-',
            color='black'
            )
        ax[unit].annotate(
            f"{samples/fs*1000:.0f} ms",
            xy=(samples/2, time_y_ref),
            xytext=(0, time_y_ref + y_offset/10)
            )

        # Add amplitude reference
        amp_x_ref = (samples + x_offset) * 2
        max_ptp = np.nanmax(np.ptp(muaps_plot[unit], axis=-1))
        amp_y_ref = [-max_ptp/2, max_ptp/2] - y_offset * rows - y_offset/10
        ax[unit].plot(
            [amp_x_ref, amp_x_ref],
            amp_y_ref,
            '-', color='black'
            )
        ax[unit].annotate(
            f"{np.max(np.ptp(muaps[unit], axis=-1)):.2f} mV",
            xy=(amp_x_ref, np.mean(amp_y_ref)),
            xytext=(amp_x_ref*1.1, np.mean(amp_y_ref))
            )

        # Remove axes
        ax[unit].set_xlim([-x_offset, (x_offset + samples) * cols])
        ax[unit].set_ylim([-y_offset*(rows+1), y_offset])
        ax[unit].set_title(f'Motor unit: {unit}')
        ax[unit].set_axis_off()

    for i in range(unit+1, len(ax)):
        ax[i].set_axis_off()

    return ax


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
              if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1.2, 1))


def plot_clustered_muaps(
        muaps,
        cluster_labels,
        lags,
        color_labels,
        fs,
        ax=None,
        palette_name="viridis",
        color_order=None,
        normalize=False,
        ch_framed='max_amp'
        ):

    # Check muaps dimensions
    if len(muaps.shape) < 4:
        muaps = np.expand_dims(muaps, axis=0)

    # Normalise muaps for plots
    muaps_plot = copy(muaps)
    if normalize:
        max_amp = np.nanmax(muaps, axis=(1, 2, 3))
        muaps_plot /= max_amp[:, None, None, None]
    else:
        max_amp = np.nanmax(np.abs(muaps))
        muaps_plot /= max_amp

    #  Initialise variables
    n_units, rows, cols, samples = muaps_plot.shape
    x_offset = round(muaps_plot.shape[-1]/10)
    y_offset = np.nanmax(np.abs(muaps_plot)) * 2.01
    x = np.arange(samples)
    clusters = np.unique(cluster_labels)
    n_clusters = len(clusters)

    # Create color palette
    if color_order is None:
        color_order = np.unique(color_labels)
    color_palette = sns.color_palette(palette_name, len(color_order))
    color_dict = {
        label: color_palette[i] for i, label in enumerate(color_order)
        }

    # Define axes if not initialised
    if ax is None:
        fig, ax = plt.subplots(n_clusters, 1, figsize=(10, 4*n_clusters))
        ax = np.ravel(ax)
    else:
        ax = np.expand_dims(ax, axis=0)

    for c, cluster in enumerate(clusters):

        # Get all the units belonging to the cluster
        cluster_idx = np.nonzero(cluster_labels == cluster)[0]
        muaps_cluster = muaps_plot[cluster_idx]
        cluster_color_labels = color_labels[cluster_idx]
        cluster_lags = lags[cluster_idx[0], cluster_idx]

        if normalize is False:
            # Normalise cluster amplitude
            cluster_max_amp = np.nanmax(np.abs(muaps_cluster))
            muaps_cluster /= cluster_max_amp

        # Check size
        if len(muaps_cluster.shape) < 4:
            muaps_cluster = np.expand_dims(muaps_cluster, axis=0)
        n_units_cluster = muaps_cluster.shape[0]

        # Plot muaps_plot
        for unit in range(n_units_cluster):

            # Apply temporal alignment
            muaps_cluster[unit] = np.roll(
                muaps_cluster[unit], cluster_lags[unit], axis=-1
                )

            # Get highest amplitude channels
            if ch_framed == 'max_amp':
                curr_amp_ch = get_highest_amp_ch(muaps_cluster[unit])
            elif ch_framed == 'ptp':
                curr_amp_ch = get_highest_ptp_ch(muaps_cluster[unit])
            elif ch_framed == 'iqr':
                curr_amp_ch = get_highest_iqr_ch(muaps_cluster[unit])
            elif ch_framed == 'iqr_ptp':
                curr_amp_ch = get_highest_iqr_ptp_ch(muaps_cluster[unit])
            elif ch_framed == 'per':
                curr_amp_ch = get_percentile_ch(muaps_plot[unit], thr=95)

            for col in range(cols):
                # Apply offset to signals
                curr_muap_col = muaps_cluster[unit, :, col].T
                curr_muap_col -= range(0, rows) * y_offset

                # Current x
                curr_x = x + (samples + x_offset) * col

                # Plot column
                if col == 0:
                    curr_label = cluster_color_labels[unit]
                else:
                    curr_label = None
                ax[c-1].plot(
                    curr_x,
                    curr_muap_col,
                    color=color_dict[cluster_color_labels[unit]],
                    linewidth=1,
                    label=curr_label
                    )

                # Plot frames
                if ch_framed is None:
                    continue
                frames = []
                for row in range(rows):
                    if curr_amp_ch[row, col] is False:
                        continue
                    frames.append(Rectangle(
                        (curr_x[0] - x_offset/2, -y_offset * row - y_offset/2),
                        width=samples + x_offset,
                        height=y_offset
                        ))
                frame_collection = PatchCollection(
                    frames,
                    ls='-',
                    ec="lightgrey",
                    fc="none",
                    lw=1
                    )
                ax[c-1].add_collection(frame_collection)

        # Add time reference
        time_y_ref = - y_offset * (rows) - y_offset/10
        ax[c-1].plot(
            [0, samples],
            [time_y_ref, time_y_ref],
            '-',
            color='black'
            )
        ax[c-1].annotate(
            f"{samples/fs*1000:.0f} ms",
            xy=(samples/2, time_y_ref),
            xytext=(0, time_y_ref + y_offset/10)
            )

        # Add amplitude reference
        amp_x_ref = (samples + x_offset) * 2
        max_ptp = np.nanmax(
            np.ptp(muaps_cluster, axis=-1)
            )
        amp_y_ref = [-max_ptp/2, max_ptp/2] - y_offset * rows - y_offset/10
        ax[c-1].plot(
            [amp_x_ref, amp_x_ref],
            amp_y_ref,
            '-', color='black')
        ax[c-1].annotate(
            f"{np.nanmax(np.ptp(muaps[cluster_idx], axis=-1)):.2f} mV",
            xy=(amp_x_ref, np.mean(amp_y_ref)),
            xytext=(amp_x_ref*1.1, np.mean(amp_y_ref))
            )

        # Remove axes
        ax[c-1].set_xlim([-x_offset, (x_offset + samples) * cols])
        ax[c-1].set_ylim([-y_offset*(rows+1), y_offset])
        ax[c-1].set_title(f'Cluster: {cluster}')
        legend_without_duplicate_labels(ax[c-1])
        ax[c-1].set_axis_off()

    return ax


def plot_clusters(
        data_cluster_info,
        muaps,
        fs=2048,
        labels_field=None,
        labels_order=None,
        color_palette_name=None
        ):

    # Initialise clusters
    clusters = np.unique(data_cluster_info["cluster_label"])

    # Initialise timestamp
    t = np.linspace(
        - muaps.shape[-1]/fs/2,
        muaps.shape[-1]/fs/2,
        muaps.shape[-1]
        ) * 1000
    # Initialise figures
    n_cols = 5
    n_rows = math.ceil(len(clusters)/n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows), dpi=200)
    axs = np.ravel(axs)

    # Define color palette
    if color_palette_name is not None:
        color_palette = sns.color_palette(
            color_palette_name,
            len(labels_order)
            )
    else:
        color_palette = ['b'] * np.amax(data_cluster_info["cluster_size"])

    # Get amplitude range
    y_min = np.amin(muaps)
    y_max = np.amax(muaps)

    # Get indexes of the dataframe
    df_indexes = data_cluster_info.index.values

    for c in clusters:
        # Get the cluster indexes adn the maximum amplitude channel to plot
        curr_cluster_mask = data_cluster_info["cluster_label"].values == c
        curr_cluster_idxs = np.nonzero(curr_cluster_mask)[0]
        curr_max_ch = np.unravel_index(
            np.argmax(muaps[curr_cluster_mask], axis=None),
            muaps.shape
            )
        curr_max_ch = curr_max_ch[1:3]

        # Get mean and std withing clusters
        curr_cluster_R_within_mean = np.nanmean(
            data_cluster_info.loc[
                df_indexes[curr_cluster_idxs],
                "R_within_mean"
                ]
            )
        curr_cluster_R_within_std = np.nanstd(
            data_cluster_info.loc[
                df_indexes[curr_cluster_idxs],
                "R_within_mean"
                ]
            )

        # Get mean and std between clusters
        curr_cluster_R_between_mean = np.nanmean(
            data_cluster_info.loc[
                df_indexes[curr_cluster_idxs],
                "R_between_mean"
                ]
            )
        curr_cluster_R_between_std = np.nanstd(
            data_cluster_info.loc[
                df_indexes[curr_cluster_idxs],
                "R_between_mean"
                ]
            )

        for i, unit in enumerate(curr_cluster_idxs):
            # Get current angle and finger
            if labels_field is not None:
                curr_label = data_cluster_info.loc[
                    df_indexes[unit], labels_field
                    ]
                curr_color_idx = np.nonzero(
                    labels_order == curr_label
                    )[0][0]
            else:
                curr_label = i
                curr_color_idx = i

            # Get current unit and align it
            curr_muap = muaps[unit]
            curr_muap_aligned = np.roll(
                curr_muap,
                data_cluster_info.loc[df_indexes[unit], "cluster_lag"],
                axis=-1
                )

            # Plot muap
            axs[c-1].plot(
                t, curr_muap_aligned[curr_max_ch],
                label=curr_label,
                color=color_palette[curr_color_idx]
                )

        axs[c-1].spines['top'].set_visible(False)
        axs[c-1].spines['right'].set_visible(False)
        axs[c-1].spines['bottom'].set_visible(False)
        axs[c-1].spines['left'].set_visible(False)
        axs[c-1].set_title(
            f"Cluster{c}\n" +
            f"R_w = {curr_cluster_R_within_mean:.2f} ± " +
            f"{curr_cluster_R_within_std:.2f}\n" +
            f"R_b = {curr_cluster_R_between_mean:.2f} ± " +
            f"{curr_cluster_R_between_std:.2f}"
            )
        axs[c-1].legend(bbox_to_anchor=(1.2, 1))
        axs[c-1].grid(True)
        axs[c-1].set_xlim([t[0], t[-1]])
        axs[c-1].set_xlabel('Time (ms)')
        axs[c-1].set_ylim([y_min, y_max])
        axs[c-1].set_ylabel("mV")

    for j in range(c, len(axs)):
        axs[j].set_axis_off()

    plt.tight_layout()
    plt.show()
