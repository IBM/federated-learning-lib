"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2023 All Rights Reserved.
"""
#!/usr/bin/env python3

import json
import math
import pprint
from collections import OrderedDict

import dateutil as du
import yaml

pp = pprint.PrettyPrinter(indent=4)
import re
import statistics
import sys
from string import Template

import matplotlib.pyplot as plt

plt.style.use("seaborn")
plt.rcParams.update({"lines.markeredgewidth": 1})
import numpy as np

np.set_printoptions(precision=2, linewidth=100)
import pandas as pd


def json_to_table(json_data):
    """
    Convert metrics json data into a DataFrame for easier processing

    :param json_data: the raw read-in metrics file
    :type l: `dict`
    :return: reorganized subset of the data as DataFrame, based on some conventions
    :rtype: `pandas.DataFrame`
    """
    # TODO: sort the keys in this order
    table_col_order = [
        "round_no",
        "pre_train_ts",
        "pre_train_eval_reward",
        "post_train_train_reward_mean",
        "post_train_train_reward_max",
        "post_train_ts",
        "post_train_eval_reward",
        "pre_update_ts",
        "post_update_ts",
    ]
    i = 0
    table_data = []
    for row in json_data:
        flat_dict = {}
        for k1, v1 in row.items():
            if not isinstance(v1, dict):
                flat_dict[k1] = v1
            else:
                for k2, v2 in v1.items():
                    flat_dict["{}:{}".format(k1, k2)] = v2
        table_data += [flat_dict]
    return pd.DataFrame(table_data)


def parse_party_data(file_path, n_trials, n_parties):
    """
    Read in all data for an experiment into a single dictionary

    :param file_path: path to the metrics file
    :type file_path: `str`
    :param n_trials: the number of trials in the experiment
    :type n_trials: `int`
    :param n_parties: the number of parties for the experiment
    :type n_parties: `int`
    :return: Dictionary with one key per metric, whose values are lists with one element per party, \
    which is a list of trials, each trial a numpy.array with a value for each round of the trial
    :rtype: `dict[str,list[list[numpy.array]]]`
    """
    # each dict key is a list of list of arrays, outer list -> party, inner list -> trial;
    # metadata is just the round number (used to be IPs per party)
    # TODO: un-hard-code this; still treating metadata weirdly
    dat = {}
    dat["metadata"] = []
    metadata_keys = ["round_no"]
    for party in range(n_parties):
        for k, v in dat.items():
            if "metadata" in k:
                dat[k] += [{}]
            else:
                dat[k] += [[]]
        for trial in range(1, n_trials + 1):
            with open(Template(file_path).substitute({"trial": trial, "id": party, "ts": "latest"})) as json_file:
                table = json_to_table(json.load(json_file))
            for k, v in table.items():
                if k in metadata_keys:
                    dat["metadata"][party][k] = table[k].to_numpy()
                elif k not in dat:
                    dat[k] = [[]]
                    dat[k][party] += [table[k].to_numpy()]
                else:
                    dat[k][party] += [table[k].to_numpy()]
    return dat


def offset_method_first(l):
    """
    Example of an offset method that can be passed to the various offset_vals function below; \
    it offsets all values based on the first value in the list

    :param l: the values to offset
    :type l: list
    :return: a new list (NOT a reference to the input list) with the new elements
    :rtype: list
    """
    ret = np.copy(l)
    for i, v in enumerate(l):
        ret[i] -= l[0]
    return ret


def offset_method_delta(l):
    """
    Example of an offset method that can be passed to the various offset_* functions below; \
    it offsets all values based on the previous value in the list

    :param l: the values to offset
    :type l: list
    :return: a new list (NOT a reference to the input list) with the new elements
    :rtype: list
    """
    ret = np.copy(l)
    for i, v in enumerate(l):
        ret[i] -= l[i - 1 if i > 0 else 0]
    return ret


def offset_vals(metrics_dict, offset_keys, offset_methods_dict):
    """
    Compute the offset of the values for a set of metrics using the strategies specified

    :param metrics_dict: data for an experiment (output of parse_party_data)
    :type metrics_dict: `dict[list[list[np.array]]]`
    :param offset_keys: the keys in the metrics_dict to apply the offset to
    :type offset_keys: `list[str]`
    :param offset_methods_dict: a label-indexed dictionary of the strategies to use
    :type offset_methods_dict: `dict[str,callable]`
    :return: A reference to the input metrics dictionary
    :rtype: `dict[list[list[np.array]]]`
    """
    p_id = 0
    for offset_method_suffix, offset_method in offset_methods_dict.items():
        for m_key in offset_keys:
            offset_method_key = "{}_{}".format(m_key, offset_method_suffix)
            m_val = metrics_dict[m_key]
            metrics_dict[offset_method_key] = []
            for p_ind, p_vals in enumerate(m_val):
                metrics_dict[offset_method_key] += [[]]
                for i_ind, i_vals in enumerate(p_vals):
                    metrics_dict[offset_method_key][p_ind] += [[]]
                    metrics_dict[offset_method_key][p_ind][i_ind] = offset_method(i_vals)

    return metrics_dict


def offset_vals_cycle(metrics_dict, offset_keys):
    """
    Compute the offset of the values for a set of metrics using the "delta" strategy; \
    unlike the above function, the keys listed are used in sequence; instead of the offset \
    being computed per-key, the offset is assuming that all values are taken per round \
    in the order they appear in offset_keys

    :param metrics_dict: data for an experiment (output of parse_party_data)
    :type metrics_dict: `dict[list[list[np.array]]]`
    :param offset_keys: the keys in the metrics_dict to apply the offset to in sequence
    :type offset_keys: `list[str]`
    :return: A reference to the input metrics dictionary
    :rtype: `dict[list[list[np.array]]]`
    """
    p_id = 0
    for m_ind, m_key in enumerate(offset_keys):
        m_val1 = metrics_dict[offset_keys[m_ind]]
        m_val2 = metrics_dict[offset_keys[(m_ind + 1) % len(offset_keys)]]
        delta_key = "{}_delta".format(m_key)
        metrics_dict[delta_key] = []
        for p_ind, p_vals1 in enumerate(m_val1):
            metrics_dict[delta_key] += [[]]
            for i_ind, i_vals1 in enumerate(p_vals1):
                metrics_dict[delta_key][p_ind] += [[]]
                if m_ind == len(offset_keys) - 1:
                    metrics_dict[delta_key][p_ind][i_ind] = np.roll(m_val2[p_ind][i_ind], -1) - i_vals1
                    metrics_dict[delta_key][p_ind][i_ind][-1] = np.nan
                else:
                    metrics_dict[delta_key][p_ind][i_ind] = m_val2[p_ind][i_ind] - i_vals1

    return metrics_dict


def group_by_iter(metrics_dict):
    """
    Restructure our metrics dictionary to have the last list store all the trials' values \
    for a given iteration, instead of all the iterations' values for a given trial.

    :param metrics_dict: data for an experiment (output of parse_party_data)
    :type metrics_dict: `dict[list[list[np.array]]]`
    :return: A new, reorganized dict
    :rtype: `dict[list[list[np.array]]]`
    """
    # TODO: more pythonic, pandas-thonic, or numpy-thonic way of doing this?
    metrics_gbi = {}
    # look into the metrics...
    for metric_key, metric_llist in metrics_dict.items():
        metrics_gbi[metric_key] = []
        # ... for each party...
        for party_idx, metric_for_party in enumerate(metric_llist):
            metrics_gbi[metric_key] += [[]]
            # ... for each trial...
            for metric_for_trial in metric_for_party:
                # ... and finally for each iter.
                for iter_idx, iter_val in enumerate(metric_for_trial):
                    if len(metrics_gbi[metric_key][party_idx]) <= iter_idx:
                        metrics_gbi[metric_key][party_idx] += [[]]
                    metrics_gbi[metric_key][party_idx][iter_idx] += [iter_val]
    return metrics_gbi


def aggregate_over_trials(metrics_gbi, agg_methods):
    """
    Aggregate the values for all the trials at a given iteration
    Some examples of agg_methods that could be useful: \
      - statistics.mean \
      - statistics.median \
      - max \
      - lambda x: 2.086 * statistics.stdev(x)/math.sqrt(len(x)) if len(x) > 1 else 0 \
      - lambda x: x[0] \
      - random.choice

    :param metrics_gbi: data for an experiment (output of group_by_iter)
    :type metrics_gbi: `dict[list[list[np.array]]]`
    :return: A new, reorganized dict
    :rtype: `dict[str,list[dict[str,list]]]`
    """
    metrics_abt = {}
    for m_key, m_val in metrics_gbi.items():
        metrics_abt[m_key] = []
        for party_idx, metric_for_party in enumerate(m_val):
            if m_key == "metadata":
                metrics_abt[m_key] += [[]]
                continue
            metrics_abt[m_key] += [{key: [] for key in agg_methods.keys()}]
            for iter_val_list in metric_for_party:
                for key, val in agg_methods.items():
                    if None in iter_val_list:
                        metrics_abt[m_key][-1][key] += [None]
                    else:
                        metrics_abt[m_key][-1][key] += [val(iter_val_list)]
    return metrics_abt


################################################################################################


def plot_metric_vs_x(metrics_dict, x_vals, metric_key, plot_title, output_filepath=None):
    """
    A helper function for plotting metrics vs various x variables

    :param metrics_dict: data for an experiment \
    (output of aggregate_over_trials with 'mean' and 'stderr' functions applied during aggregation)
    :type metrics_dict: `dict[str,list[dict[str,list]]]`
    :param x_vals: one metric's data for use as x variable in plot
    :type x_vals: `list[list[numeric]]`
    :param metric_key: key in metrics_dict use as y variable in plot
    :type metric_key: `str`
    :param plot_title: text to use as title for plot 
    :type plot_title: `str`
    :param output_filepath: path to save plot to, or None (plot will open in a window)
    :type output_filepath: `str`
    :return: None
    """
    for party, metric_dict in enumerate(metrics_dict[metric_key]):
        plt.errorbar(
            x=x_vals[party],
            y=metric_dict["mean"],
            yerr=metric_dict["stderr"],
            label="party {}".format(party),
            capsize=2,
        )
    plt.title(plot_title)
    plt.xlabel("step")
    plt.ylabel(metric_key)
    plt.legend(loc="lower right")
    if output_filepath:
        plt.savefig(output_filepath)
    else:
        plt.show()
        plt.close()


def plot_metric_vs_time(metrics_dict, metadata_dict, metric_key, plot_title, output_filepath=None):
    """
    Plot the given metric vs time, based on a common timestamp collected as part of metrics

    :param metrics_dict: data for an experiment \
    (output of aggregate_over_trials with 'mean' and 'stderr' functions applied during aggregation)
    :type metrics_dict: `dict[str,list[dict[str,list]]]`
    :param metadata_dict: 'metadata' key from the output of parse_party_data
    :type metadata_dict: `dict`
    :param metric_key: key in metrics_dict use as y variable in plot
    :type metric_key: `str`
    :param plot_title: text to use as title for plot 
    :type plot_title: `str`
    :param output_filepath: path to save plot to, or None (plot will open in a window)
    :type output_filepath: `str`
    :return: None
    """
    party_list = range(len(metrics_dict["post_train:ts_off"]))
    x_vals = {party: metrics_dict["post_train:ts_off"][party]["mean"] for party in party_list}
    plot_metric_vs_x(metrics_dict, x_vals, metric_key, plot_title, output_filepath=None)


def plot_metric_vs_round(metrics_dict, metadata_dict, metric_key, plot_title, output_filepath=None):
    """
    Plot the given metric vs round no, which is collected as part of the metrics' metadata

    :param metrics_dict: data for an experiment \
    (output of aggregate_over_trials with 'mean' and 'stderr' functions applied during aggregation)
    :type metrics_dict: `dict[str,list[dict[str,list]]]`
    :param metadata_dict: 'metadata' key from the output of parse_party_data
    :type metadata_dict: `dict`
    :param metric_key: key in metrics_dict use as y variable in plot
    :type metric_key: `str`
    :param plot_title: text to use as title for plot 
    :type plot_title: `str`
    :param output_filepath: path to save plot to, or None (plot will open in a window)
    :type output_filepath: `str`
    :return: None
    """
    party_list = range(len(metadata_dict))
    x_vals = {party: metadata_dict[party]["round_no"] for party in party_list}
    plot_metric_vs_x(metrics_dict, x_vals, metric_key, plot_title, output_filepath=None)


def gen_reward_vs_time_plots2(dat, reward_keys, x_axis_val="round", x_axis_key=None):
    """
    Plot the given metric vs round no, which is collected as part of the metrics' metadata

    :param dat: metrics data as produced by runner.get_experiment_output \
    (the dict has one element, 'metadata', which is a dict as well, contrary to the below type)
    :type dat: `dict[list[list[np.array]]]`
    :param reward_keys: keys to plot on y axis
    :type reward_keys: `list[str]`
    :param x_axis_val: we can specify a label here to select our x-axis value; \
    'round' is treated uniquely to use the round number, any other string requires x_axis_key
    :type x_axis_val: `str`
    :param x_axis_key: the key to use, if we didn't select 'round' for the x_axis_val
    :type x_axis_key: `str`
    :return: None
    """
    # group metrics data by iter, so each inner list contains each trials' value for that iter
    metadata_dict = dat.pop("metadata")
    metrics_dict = group_by_iter(dat)
    trial_agg_methods = {
        "mean": statistics.mean,
        "stderr": lambda x: 2.086 * statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0,
        "len": len,
    }
    metrics_dict = aggregate_over_trials(metrics_dict, trial_agg_methods)

    # make plots for the desired values
    if x_axis_val == "round":
        plot_metric = plot_metric_vs_round
    elif x_axis_val == "time":
        plot_metric = plot_metric_vs_time
    else:
        print("Bad x-axis value.")
        sys.exit(1)

    for reward_key in reward_keys:
        plot_title = "{} vs {}".format(reward_key, x_axis_val)
        plot_metric(metrics_dict, metadata_dict, reward_key, plot_title)


def gen_reward_vs_time_plots(metrics_file_tmpl, n_trials, n_parties, reward_keys, x_axis_val="round", x_axis_key=None):
    """
    Plot the given metric vs round no, which is collected as part of the metrics' metadata

    :param metrics_file_tmpl: a path to the metrics files for an experiment, \
    containing a ${trial} and ${id} template parameter
    :type metrics_file_tmpl: `str`
    :param n_trials: number of trials in the experiment to use for this plot
    :type n_trials: `int`
    :param n_parties: number of parties in the experiment to use for this plot 
    :type n_parties: `int`
    :param reward_keys: keys in 
    :type reward_keys: `list[str]`
    :param x_axis_val: we can specify a label here to select our x-axis value; \
    'round' is treated uniquely to use the round number, any other string requires x_axis_key
    :type x_axis_val: `str`
    :param x_axis_key: the key to use, if we didn't select 'round' for the x_axis_val
    :type x_axis_key: `str`
    :return: None
    """
    # obtain the party data
    dat = parse_party_data(metrics_file_tmpl, n_trials, n_parties)
    offset_methods = {"off": offset_method_first, "del": offset_method_delta}
    dat = offset_vals(dat, [k for k, v in dat.items() if ":ts" in k], offset_methods)
    # pp.pprint(dat)

    gen_reward_vs_time_plots2(dat, reward_keys, x_axis_val, x_axis_key)


def gen_timing_plots(metrics_file_tmpl, n_trials, n_parties, offset_cycle_keys):
    """
    Plot the timing deltas for the code regions delinteated by a list of keys as a stacked bar

    :param metrics_file_tmpl: a path to the metrics files for an experiment, \
    containing a ${trial} and ${id} template parameter
    :type metrics_file_tmpl: `str`
    :param n_trials: number of trials in the experiment to use for this plot
    :type n_trials: `int`
    :param n_parties: number of parties in the experiment to use for this plot 
    :type n_parties: `int`
    :param offset_cycle_keys: keys of timestamp metrics which delineate regions to time
    :type offset_cycle_keys: `list[str]`
    :return: None
    """
    # obtain the party data
    dat = parse_party_data(metrics_file_tmpl, n_trials, n_parties)
    dat = offset_vals_cycle(dat, offset_cycle_keys)

    # get metadata from dictionary and remove it as it won't be treated the same way as the rest
    metadata_dict = dat.pop("metadata")

    # group metrics data by iter, so each inner list contains each trials' value for that iter
    metrics_dict = group_by_iter(dat)
    trial_agg_methods = {
        "mean": statistics.mean,
        "var": lambda x: statistics.variance(x) if len(x) > 1 else 0,
        "len": len,
    }
    metrics_dict = aggregate_over_trials(metrics_dict, trial_agg_methods)

    # filter keys to the only ones we want
    offset_cycle_keys = ["{}_delta".format(k) for k in offset_cycle_keys]
    metrics_dict = {k: metrics_dict[k] for k in metrics_dict.keys() & offset_cycle_keys}

    # compute mean over rounds, and compute standard error using error propagation
    for k, v in metrics_dict.items():
        for i, p in enumerate(v):
            n_iters = len(p["var"])
            p["mean"] = np.nanmean(p["mean"])
            p["var"] = np.nansum(p["var"]) / math.pow(n_iters, 2)
            p["sde"] = 2.086 * math.sqrt(p["var"]) / math.sqrt(n_iters)
            p["len"] = statistics.mode(p["len"])

    # reorganize values for plotting
    plot_dict = {k: {} for k in offset_cycle_keys}
    for k, v in metrics_dict.items():
        plot_dict[k]["mean"] = []
        plot_dict[k]["sde"] = []
        for party in v:
            print(party)
            plot_dict[k]["mean"] += [party["mean"]]
            plot_dict[k]["sde"] += [party["sde"]]

    (fig1, axes) = plt.subplots(
        1, 4, gridspec_kw={"width_ratios": [0.8, 1.9, 3.95, 8]}, sharey=True, constrained_layout=True
    )
    first = True
    width = 0.85
    kv_prev = [0.0 for x in range(n_parties)]
    print(kv_prev)
    for ki, kv in plot_dict.items():
        axes[1].bar(range(n_parties), kv["mean"], width, yerr=kv["sde"], bottom=kv_prev)
        kv_prev = [kv["mean"][i] + kv_prev[i] for i in range(1)]
    axes[1].set_xlabel("{} parties".format(n_parties))
    axes[1].set_xticks(range(n_parties))
    axes[1].set_xticklabels(range(1, n_parties + 1))
    axes[0].set_ylabel("time (s)")
    fig1.suptitle("Time per Iteration (8 runs, all randomized machines, sorted)")
    plt.show()
    plt.close()

    ################################################################################################

    """
    This simply shows an example call to each of the two provided plotting functions.

    We can easily perform the following postprocessing tests:
    - varying n_parties / sync_interval / n_workers_per_party:
        - reward vs round 
        - reward vs time
    - variation of timings within party (stacked bar).
    """


if __name__ == "__main__":
    gen_reward_vs_time_plots(
        #'/home/sean/repos/IBMFL_new/IBMFL/examples/data/trial${trial}/party${id}_metrics.json',
        "/tmp/ibmfl_results/trial${trial}/party${id}_metrics.json",
        1,
        2,
        ["post_train:eval:episode_reward_mean"],
    )

    # gen_timing_plots(
    #    '/home/sean/repos/IBMFL_new/IBMFL/examples/data/trial{}/party{}_metrics.json',
    #    1, 2,
    #    ['pre_update_ts', 'post_update_ts', 'pre_train_ts', 'post_train_ts'])
