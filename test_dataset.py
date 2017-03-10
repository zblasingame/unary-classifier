"""Tests all subsets of dataset and each combination of features

Author:         Zander Blasingame
Instiution:     Clarkson University
Lab:            CAMEL
"""

import argparse
import csv
import classifier
import itertools
import json
import os

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('dir',
                    type=str,
                    help=('Location of subset directories containing '
                          'training and testing csv files '
                          '(must be labeled train.csv and test.csv)'))
parser.add_argument('features',
                    type=str,
                    help=('Location of the file containing the features '
                          'to use(csv formatted)'))
parser.add_argument('out',
                    type=str,
                    help=('Location of output file, no extension '
                          '(csv and json formatted)'))

args = parser.parse_args()

features = []
with open(args.features, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        features = row

hpc_combinations = list(itertools.combinations(features, 4))

dataset_data = []

num_subsets = 0

# Iterate through each subloop
for path, dirs, files in os.walk(args.dir):
    if 'subset' in path:
        train = '{}/train_set.csv'.format(path)
        test = '{}/test_set.csv'.format(path)

        # Iterate through each combinations of HPCs
        for i, entry in enumerate(hpc_combinations):
            X, Y = classifier.grab_data(train, whitelist=entry)

            num_input = len(X[0])

            ae = classifier.Classifier(num_input, batch_size=100,
                                       num_epochs=35, whitelist=entry)

            # train and test neural net
            ae.train(train)

            data = ae.test(test)
            data['subset'] = num_subsets+1
            data['combination'] = ','.join(entry)

            print(data)

            dataset_data.append(data)

        num_subsets += 1

ignore_keys = ['subset', 'combination']
keys = [key for key in dataset_data[0] if key not in ignore_keys]

# calculate averages for each combination
for entry in hpc_combinations:
    combination = ','.join(entry)

    ae_data = {key: sum(entry[key] for entry in dataset_data
                        if entry['combination'] == combination)/num_subsets
               for key in keys}

    data['subset'] = 6
    data['combination'] = combination

    dataset_data.append(data)

dataset_data = sorted(dataset_data, key=lambda x: (x['combination'],
                                                   x['subset']))

with open('{}.json'.format(args.out), 'w') as f:
    json.dump(dataset_data, f, indent=2)

with open('{}.csv'.format(args.out), 'w') as f:
    writer = csv.writer(f)
    writer.writerow('hpc1,hpc2,hpc3,hpc4,subset,accuracy,'
                    'tp_rate,tn_rate,fp_rate,fn_rate')

    for entry in dataset_data:
        out_list = [entry['combination'], entry['subset'], entry['accuracy'],
                    entry['tp_rate'], entry['tn_rate'],
                    entry['fp_rate'], entry['fn_rate']]

        out_list = [str(el) for el in out_list]

        writer.writerow(out_list)
