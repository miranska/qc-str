import json

import numpy as np
import pandas as pd
from qiskit.test.mock import FakeQasmSimulator
from qiskit.transpiler.exceptions import TranspilerError

from string_comparison import StringComparator


class NumpyEnc(json.JSONEncoder):
    """
    Convert numpy int64 to the format comprehensible by the JSON encoder
    """

    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def standardize_column_elements(column):
    """
    Update column values to make sure that column element values are consistent
    Note that the changes will happen in place
    :param column: pandas series
    :return: updated column, number of unique attributes
    """
    dic = {}
    numeric_id = 0
    for ind in range(0, len(column)):
        element_value = column.iloc[ind]
        if element_value not in dic:
            dic[element_value] = numeric_id
            numeric_id += 1
        column.iloc[ind] = dic[element_value]
    return column, len(dic)


def get_data(file_name, label_location, encoding, columns_to_remove=None,
             fraction_of_rows=0.9, random_seed=42):
    """
    Take the dataset, reshuffle, retain 90% of it and return a list of datasets (one per label/class)

    :param file_name: name of the file to read the data from
    :param label_location: location of the label column (first or last), applied after undesired columns are removed
    :param encoding: Type of encoding: either one-hot or label
    :param columns_to_remove: List of unwanted columns to remove
    :param fraction_of_rows: Fraction of rows to retain for analysis
    :param random_seed: The value of random seed needed for reproducibility
    :return: a list of datasets, max number of attributes, features count
    """

    df = pd.read_csv(file_name, header=None)

    # remove unwanted columns
    if columns_to_remove is not None:
        df = df.drop(df.columns[columns_to_remove], axis=1)
        # update column names
        df.columns = list(range(0, len(df.columns)))

    # get indexes of data columns
    col_count = len(df.columns)
    if label_location == "first":
        data_columns = range(1, col_count)
        label_column = 0
    elif label_location == "last":
        data_columns = range(0, col_count - 1)
        label_column = col_count - 1
    else:
        raise Exception(f"Unknown label_location {label_location}")

    features_cnt = len(data_columns)
    # standardize column elements and get max number of attributes in a column/feature
    max_attr_cnt = -1
    for data_column in data_columns:
        updated_column, attr_cnt = standardize_column_elements(df[data_column].copy())
        df[data_column] = updated_column
        if attr_cnt > max_attr_cnt:
            max_attr_cnt = attr_cnt

    # get 90% of strings (drawn at random)
    df = df.sample(n=round(len(df.index) * fraction_of_rows), random_state=random_seed)

    # get labels
    labels = df[label_column].unique()

    # generate strings
    strings = {}
    for label in labels:
        single_class = df[df[label_column] == label]
        class_strings = []
        for ind in range(0, len(single_class.index)):
            observation = single_class.iloc[ind]
            if encoding == "label":
                my_string = []
                for feature_ind in data_columns:
                    my_string.append(str(observation.iloc[feature_ind]))
                class_strings.append(my_string)
            elif encoding == "one-hot":
                my_string = ""
                if max_attr_cnt > 2:
                    for feature_ind in data_columns:
                        value = observation.iloc[feature_ind]
                        one_hot = [0] * max_attr_cnt
                        one_hot[value] = 1
                        my_string += ''.join(map(str, one_hot))
                else:  # use binary string for the 2-attribute case
                    for feature_ind in data_columns:
                        value = observation.iloc[feature_ind]
                        one_hot = [value]
                        my_string += ''.join(map(str, one_hot))

                class_strings.append(my_string)
            else:
                raise Exception(f"Unknown encoding {encoding}")
        strings[label] = class_strings

    return strings, max_attr_cnt, features_cnt


if __name__ == "__main__":
    stats = []

    files = [
        {"file_name": "./datasets/balance_scale.csv", "label_location": "first", 'labels': ['R'],
         'is_laborious': False},
        {"file_name": "./datasets/tictactoe.csv", "label_location": "last", 'labels': ['positive'],
         'is_laborious': True},
        {"file_name": "./datasets/breast_cancer.csv", "label_location": "last", "remove_columns": [0], 'labels': [2],
         'is_laborious': True},
        {"file_name": "./datasets/zoo.csv", "label_location": "last", "remove_columns": [0], 'labels': [1],
         'is_laborious': True},
        {"file_name": "./datasets/SPECTrain.csv", "label_location": "first", 'labels': [1], 'is_laborious': False}
    ]
    encodings = ["one-hot", "label"]

    is_fake_circuit_off = input("Creation of the circuit for fake simulator is laborious. "
                                "Do you want to skip it? (Y/n): ") or "Y"
    if is_fake_circuit_off.upper() == 'Y':
        print("Skip fake simulator")
        backend_types = ["abstract"]
    elif is_fake_circuit_off.upper() == 'N':
        print("Keep fake simulator")
        backend_types = ["abstract", "fake_simulator"]
    else:
        raise ValueError("Please enter y or n.")

    for file in files:
        if "remove_columns" in file:
            remove_columns = file["remove_columns"]
        else:
            remove_columns = None
        for encoding in encodings:

            classes, max_attr_count, features_count = get_data(file["file_name"],
                                                               label_location=file["label_location"],
                                                               encoding=encoding,
                                                               columns_to_remove=remove_columns)
            # parameters for String Comparisons
            if encoding == "one-hot":
                is_binary = True
                symbol_length = max_attr_count
                p_pqm = True
                symbol_count = None
            elif encoding == "label":
                is_binary = False
                symbol_length = None
                p_pqm = False
                symbol_count = max_attr_count
            else:
                raise Exception(f"Unknown encoding {encoding}")

            for label in classes:
                if 'labels' in file:  # process only a subset of labels present in file['labels']
                    if label not in file['labels']:
                        continue
                database = classes[label]
                target = database[0]  # dummy target string
                for backend_type in backend_types:
                    print(f"Analyzing {file['file_name']} for label {label} on {backend_type}")
                    if backend_type == "abstract":
                        x = StringComparator(target, database, symbol_length=symbol_length, is_binary=is_binary,
                                             symbol_count=symbol_count,
                                             p_pqm=p_pqm)
                    elif backend_type == "fake_simulator":
                        if file['is_laborious']:
                            print(f"  Skipping {file['file_name']} as it requires too much computing power")
                            continue
                        print("  Keep only two rows to speed up processing")
                        database = database[1:3]  # keep only two rows to make it simpler to compute the circuit
                        try:
                            x = StringComparator(target, database, symbol_length=symbol_length, is_binary=is_binary,
                                                 symbol_count=symbol_count,
                                                 p_pqm=p_pqm, optimize_for=FakeQasmSimulator(),
                                                 optimization_levels=[0], attempts_per_optimization_level=1)
                        except TranspilerError as e:
                            print(print(f"Unexpected {e=}, {type(e)=}"))
                            break
                    else:
                        raise Exception(f"Unknown backend type {backend_type}")

                    circ_decomposed = x.circuit.decompose().decompose(['c3sx', 'rcccx', 'rcccx_dg']).decompose('cu1')
                    run_stats = {'file_name': file['file_name'], 'encoding': encoding, 'label': label,
                                 'observations_count': len(database), 'features_count': features_count,
                                 'max_attr_count': max_attr_count, 'backend_type': backend_type,
                                 'qubits_count': x.circuit.num_qubits, 'circuit_depth': x.circuit.depth(),
                                 'circuit_count_ops': x.circuit.count_ops(),
                                 'qubits_count_decomposed': circ_decomposed.num_qubits,
                                 'circuit_depth_decomposed': circ_decomposed.depth(),
                                 'circuit_count_ops_decomposed': circ_decomposed.count_ops()
                                 }
                    stats.append(run_stats)

    print(f"Final stats in basic dictionary")
    print(stats)

    # save stats in JSON format
    out_json = json.dumps(stats, cls=NumpyEnc)
    with open('stats.json', 'w') as f:
        json.dump(out_json, f)

    # let's also save it in a table
    pd.json_normalize(stats).to_csv('stats.csv')
