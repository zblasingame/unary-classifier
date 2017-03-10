"""Script to graph the dataset accuracy in relation to HPC events

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import argparse
import json
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('file',
                    type=str,
                    help='Location of input file (json formatted)')
parser.add_argument('name',
                    type=str,
                    help='Name of dataset')
parser.add_argument('--online',
                    action='store_true',
                    help='Flag to publish graphs online')

args = parser.parse_args()

ONLINE = args.online


# Functions
def make_graph(data, filename, layout=go.Layout()):
    fig = go.Figure(data=data, layout=layout)

    if ONLINE:
        return py.iplot(fig, filename=filename)
    else:
        filename = './graphs/{}.png'.format(filename)
        py.image.save_as(fig, filename=filename)
        return None

dataset_stats = []

with open(args.file, 'r') as f:
    dataset_stats = json.load(f)

combinations = [entry['combination'] for entry in dataset_stats]
combinations = list(set(combinations))

accuracies = [[entry['accuracy'] for entry in dataset_stats
               if entry['combination'] == combination and entry['subset'] != 6]
              for combination in combinations]

data = [go.Box(y=accuracies[i],
               whiskerwidth=0.2,
               name='{}'.format(combination))
        for i, combination in enumerate(combinations)]

layout = go.Layout(title='Accuracy for Combination in {}'.format(args.name),
                   yaxis=dict(title='Accuracy (%)'))

plot_url = make_graph(data=data, layout=layout,
                      filename='{}-unary-classification-box'.format(args.name))
