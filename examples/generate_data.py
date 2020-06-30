#!/usr/bin/env python3
import os
import sys
import csv
import time
import argparse
import numpy as np

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from ibmfl.util.datasets import load_nursery, load_mnist, load_adult, load_higgs, load_airline, \
    load_diabetes, load_binovf, load_multovf, load_linovf, load_simulated_federated_clustering
from examples.constants import GENERATE_DATA_DESC, NUM_PARTIES_DESC, DATASET_DESC, PATH_DESC, PER_PARTY, \
    STRATIFY_DESC, FL_DATASETS, NEW_DESC, PER_PARTY_ERR, NAME_DESC


def setup_parser():
    """
    Sets up the parser for Python script

    :return: a command line parser
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description=GENERATE_DATA_DESC)
    p.add_argument("--num_parties", "-n", help=NUM_PARTIES_DESC,
                   type=int, required=True)
    p.add_argument("--dataset", "-d", choices=FL_DATASETS,
                   help=DATASET_DESC, required=True)
    p.add_argument("--data_path", "-p", help=PATH_DESC,
                   default=os.path.join("examples", "data"))
    p.add_argument("--points_per_party", "-pp", help=PER_PARTY,
                   nargs="+", type=int, required=True)
    p.add_argument("--stratify", "-s", help=STRATIFY_DESC, action="store_true")
    p.add_argument("--create_new", "-new", action="store_true", help=NEW_DESC)
    p.add_argument("--name", help=NAME_DESC)
    return p


def print_statistics(i, x_test_pi, x_train_pi, nb_labels, y_train_pi):
    print('Party_', i)
    print('nb_x_train: ', np.shape(x_train_pi),
          'nb_x_test: ', np.shape(x_test_pi))
    for l in range(nb_labels):
        print('* Label ', l, ' samples: ', (y_train_pi == l).sum())


def save_nursery_party_data(nb_dp_per_party, should_stratify, party_folder):
    """
    Saves Nursery party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    x_train = load_nursery(download_dir=dataset_path)
    num_train = len(x_train.index)
    y_train = x_train['class'].values.tolist()
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()
        # Use indices for data/classification subset
        x_train_pi = x_train.iloc[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        with open(name_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(x_train_pi)

        x_train_pi.to_csv(path_or_buf=name_file, index=None)

    print('Finished! :) Data saved in', party_folder)


def save_adult_party_data(nb_dp_per_party, should_stratify, party_folder):
    """
    Saves Adult party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    x_train = load_adult(download_dir=dataset_path)
    num_train = len(x_train.index)
    y_train = x_train['class'].values.tolist()
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        strat_col = y_train
        groups, counts = np.unique(strat_col, return_counts=True)
        # to use custom proportions, replace probs with a dictionary where key:value pairs are label:proportion
        probs = {group: counts[np.where(groups == group)[
            0][0]] / float(num_train) for group in groups}
        p_list = np.array([probs[strat_col[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    else:
        probs = {label: 1.0 / num_train for label in labels}
        p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()
        # Use indices for data/classification subset
        x_train_pi = x_train.iloc[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        with open(name_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(x_train_pi)

        x_train_pi.to_csv(path_or_buf=name_file, index=None)

    print('Finished! :) Data saved in', party_folder)


def save_mnist_party_data(nb_dp_per_party, should_stratify, party_folder):
    """
    Saves MNIST party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    (x_train, y_train), (x_test, y_test) = load_mnist(download_dir=dataset_path)
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)

    if should_stratify:
        # Sample according to source label distribution
        train_probs = {
            label: train_counts[label] / float(num_train) for label in labels}
        test_probs = {label: test_counts[label] /
                      float(num_test) for label in te_labels}
    else:
        # Sample uniformly
        train_probs = {label: 1.0 / len(labels) for label in labels}
        test_probs = {label: 1.0 / len(te_labels) for label in te_labels}

    for idx, dp in enumerate(nb_dp_per_party):
        train_p = np.array([train_probs[y_train[idx]]
                            for idx in range(num_train)])
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
        test_p /= np.sum(test_p)

        # Split test evenly
        test_indices = np.random.choice(
            num_test, int(num_test / nb_parties), p=test_p)

        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]

        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

        print('Finished! :) Data saved in ', party_folder)


def save_higgs_party_data(nb_dp_per_party, should_stratify, party_folder):
    """
    Saves Higgs Boson party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    x, y = load_higgs(dataset_path)
    num_train = len(x)
    labels, counts = np.unique(y, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in X[indices, :].astype(str)]
        y_part = y[indices]

        # Write to File
        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_airline_party_data(nb_dp_per_party, should_stratify, party_folder):
    """
    Saves Airline Delay party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    X, y = load_airline(dataset_path)
    num_train = len(X)
    labels, counts = np.unique(y, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in X[indices, :].astype(str)]
        y_part = y[indices]

        # Write to File
        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_diabetes_party_data(nb_dp_per_party, should_stratify, party_folder):
    """
    Saves Diabetes party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    x_train = load_diabetes(dataset_path)
    num_train = len(x_train)
    y_train = x_train['readmitted'].values.tolist()
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        strat_col = y_train
        groups, counts = np.unique(strat_col, return_counts=True)
        # to use custom proportions, replace probs with a dictionary where key:value pairs are label:proportion
        probs = {
            group: counts[np.where(groups == group)[0][0]] / float(num_train) for group in groups
        }
        p_list = np.array([probs[strat_col[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    else:
        probs = {label: 1.0 / num_train for label in labels}
        p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()
        # Use indices for data/classification subset
        x_train_pi = x_train.iloc[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        with open(name_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(x_train_pi)

        x_train_pi.to_csv(path_or_buf=name_file, index=None)

    print('Finished! :) Data saved in', party_folder)


def save_binovf_party_data(nb_dp_per_party, should_stratify, party_folder):
    """
    Saves Binary Overfit party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    X, y = load_binovf()
    num_train = len(X)
    labels, counts = np.unique(y, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in X[indices, :].astype(str)]
        y_part = y[indices]

        # Write to File
        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_multovf_party_data(nb_dp_per_party, should_stratify, party_folder):
    """
    Saves Multiclass Overfit party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    x_train, y_train = load_multovf()
    num_train = len(x_train)
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        strat_col = y_train
        groups, counts = np.unique(strat_col, return_counts=True)
        # to use custom proportions, replace probs with a dictionary where key:value pairs are label:proportion
        probs = {group: counts[np.where(groups == group)[
            0][0]] / float(num_train) for group in groups}
        p_list = np.array([probs[strat_col[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    else:
        probs = {label: 1.0 / num_train for label in labels}
        p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in x_train[indices, :].astype(str)]
        y_part = y_train[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_linovf_party_data(nb_dp_per_party, party_folder):
    """
    Saves Linear Overfit party data (For Regression)
    Data stratification is not supported in this function.

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """
    dataset_path = os.path.join("examples", "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    x_train, y_train = load_linovf()
    num_train = len(x_train)

    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [item for item in x_train[indices].astype(str)]
        y_part = y_train[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(y_part[i])+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)

def save_federated_clustering_data(nb_dp_per_party, party_folder):
    """
    Saves simulated federated clustering dataset for unsupervised federated
    learning setting

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """

    num_clients = len(nb_dp_per_party)
    # Same number of data points are generated for each party
    nb_datapoints = nb_dp_per_party[0]
    # use true clusters depending on the number of clients
    # this prevents creating too many global centroids but fewer
    # cumulative local centroids
    true_clusters = 5 * num_clients

    kwargs = {
        "J": num_clients,
        "M": nb_datapoints,
        "L": true_clusters
    }

    # data returned is (J, M, D=100) dimensions
    data = load_simulated_federated_clustering(**kwargs)

    for idx in range(num_clients):
        x_train_np = np.array(data[idx])
        x_test_np = x_train_np      # Duplicating x_train to x_test

        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_np, x_test=x_test_np)

        print('Finished! :) Data saved in ', party_folder)


if __name__ == '__main__':
    # Parse command line options
    parser = setup_parser()
    args = parser.parse_args()

    # Collect arguments
    num_parties = args.num_parties
    dataset = args.dataset
    data_path = args.data_path
    points_per_party = args.points_per_party
    stratify = args.stratify
    create_new = args.create_new
    exp_name = args.name

    # Check for errors
    if len(points_per_party) == 1:
        points_per_party = [points_per_party[0] for _ in range(num_parties)]
    elif len(points_per_party) != num_parties:
        parser.error(PER_PARTY_ERR)

    # Create folder to save party data
    folder = os.path.join("examples", "data")
    strat = 'balanced' if stratify else 'random'

    if create_new:
        folder = os.path.join(folder, exp_name if exp_name else str(
            int(time.time())) + '_' + strat)
    else:
        folder = os.path.join(folder, dataset, strat)

    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # clear folder of old data
        for f_name in os.listdir(folder):
            f_path = os.path.join(folder, f_name)
            if os.path.isfile(f_path):
                os.unlink(f_path)

    # Save new files
    if dataset == 'nursery':
        save_nursery_party_data(points_per_party, stratify, folder)
    elif dataset == 'adult':
        save_adult_party_data(points_per_party, stratify, folder)
    elif args.dataset == 'mnist':
        save_mnist_party_data(points_per_party, stratify, folder)
    elif dataset == 'higgs':
        save_higgs_party_data(points_per_party, stratify, folder)
    elif dataset == 'airline':
        save_airline_party_data(points_per_party, stratify, folder)
    elif dataset == 'diabetes':
        save_diabetes_party_data(points_per_party, stratify, folder)
    elif dataset == 'binovf':
        save_binovf_party_data(points_per_party, stratify, folder)
    elif dataset == 'multovf':
        save_multovf_party_data(points_per_party, stratify, folder)
    elif dataset == 'linovf':
        save_linovf_party_data(points_per_party, folder)
    elif dataset == 'federated-clustering':
        save_federated_clustering_data(points_per_party, folder)
