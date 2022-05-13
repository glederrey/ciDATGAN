import os
import argparse
import pandas as pd
from datgan import DATGAN
import numpy as np
import networkx as nx

def train(args):

    bias = True if args.bias == 1 else False

    # Load data
    if args.bias == 0:
        file_name = 'trips.csv'
    elif args.bias == 1:
        file_name = 'trips_bias.csv'
    elif args.bias == 2:
        file_name = 'trips_small_bias.csv'

    df = pd.read_csv('../data/LPMC/' + file_name)

    # First, define the specificities of continuous variables
    data_info = {
        'start_time_linear': {
            'type': 'continuous',
            'bounds': [0.0, 23.999],
            'discrete': False,
        },
        'age': {
            'type': 'continuous',
            'bounds': [0, 100],
            'discrete': True
        },
        'distance': {
            'type': 'continuous',
            'bounds': [0, np.infty],
            'discrete': True,
            'apply_func': (lambda x: np.log(x+1))
        },
        'dur_walking': {
            'type': 'continuous',
            'bounds': [0, np.infty],
            'enforce_bounds': True,
            'discrete': False,
            'apply_func': (lambda x: np.log(x+1))
        },
        'dur_cycling': {
            'type': 'continuous',
            'bounds': [0, np.infty],
            'enforce_bounds': True,
            'discrete': False,
            'apply_func': (lambda x: np.log(x+1))
        },
        'dur_pt': {
            'type': 'continuous',
            'bounds': [0, np.infty],
            'enforce_bounds': True,
            'discrete': False,
            'apply_func': (lambda x: np.log(x+1))
        },
        'dur_driving': {
            'type': 'continuous',
            'bounds': [0, np.infty],
            'enforce_bounds': True,
            'discrete': False,
            'apply_func': (lambda x: np.log(x+1))
        },
        'driving_traffic_percent': {
            'type': 'continuous',
            'bounds': [0, np.infty],
            'discrete': False,
        },
    }

    # Add the other variables as categorical
    for c in df.columns:
        if c not in data_info.keys():
            data_info[c] = {'type': 'categorical'}

    # personalised graph
    graph = nx.DiGraph()

    graph.add_edges_from([
        ('hh_region', 'hh_people'),
        ('hh_region', 'distance'),
        ('hh_region', 'hh_income'),
        ('hh_region', 'travel_mode'),
        ('hh_income', 'hh_vehicles'),
        ('hh_people', 'hh_vehicles'),
        ('age', 'hh_people'),
        ('age', 'faretype'),
        ('age', 'driving_license'),
        ('age', 'purpose'),
        ('age', 'travel_mode'),
        ('female', 'driving_license'),
        ('female', 'hh_people'),
        ('driving_license', 'travel_mode'),
        ('hh_vehicles', 'driving_license'),
        ('hh_vehicles', 'travel_mode'),
        ('faretype', 'travel_mode'),
        ('day_of_week', 'purpose'),
        ('day_of_week', 'start_time_linear'),
        ('day_of_week', 'driving_traffic_percent'),
        ('purpose', 'start_time_linear'),
        ('purpose', 'travel_mode'),
        ('purpose', 'distance'),
        ('start_time_linear', 'driving_traffic_percent'),
        ('driving_traffic_percent', 'dur_driving'),
        ('distance', 'driving_traffic_percent'),
        ('distance', 'dur_walking'),
        ('distance', 'dur_cycling'),
        ('distance', 'dur_pt'),
        ('distance', 'dur_driving'),
        ('distance', 'travel_mode')
    ])

    ci = True if args.cond_inputs == 1 else False

    name = 'ciDATGAN' if ci else 'DATGAN'
    name += '_' + str(args.number)

    folder = None
    bs = 0
    if args.bias == 0:
        folder = 'normal'
        bs = 1300
    elif args.bias == 1:
        folder = 'bias'
        bs = 1205
    elif args.bias == 2:
        folder = 'small_bias'
        bs = 1315

    output_folder = './output/{}/{}/'.format(folder, name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    encoded_data_folder = './output/{}/encoded_data/'.format(folder)

    if not os.path.exists(encoded_data_folder):
        os.makedirs(encoded_data_folder)

    cond_inputs = ['age', 'female', 'hh_region'] if ci else None

    datgan = DATGAN(output=output_folder,
                    loss_function='WGGP',
                    conditional_inputs=cond_inputs,
                    num_epochs=1000,
                    batch_size=bs,
                    verbose=1)

    datgan.fit(df, data_info, graph, encoded_data_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--number", type=int)
    parser.add_argument("-b", "--bias", type=int)
    parser.add_argument("-ci", "--cond_inputs", type=int)

    args = parser.parse_args()

    train(args)