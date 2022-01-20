import logging
import math
import textwrap

import pandas as pd
from qiskit import Aer
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import execute
from qiskit.circuit.add_control import add_control
from qiskit.compiler import transpile
from qiskit.extensions import *
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import plot_histogram
from tabulate import tabulate

logger = logging.getLogger(__name__)


class StringComparator:

    def __init__(self, target, db, symbol_length=1, symbol_count=None, is_binary=True,
                 shots=8192, quantum_instance=Aer.get_backend('qasm_simulator'),
                 optimize_for=None, optimization_levels=None, attempts_per_optimization_level=None,
                 default_dataset=False, t=1, p_pqm=False
                 ):
        """
        Compare a string against a set of strings.

        :param target: target string
        :param db: a set of strings (passed in a list) against which we compare the target string
        :param symbol_length: the number of characters that codify a symbol; used only when is_binary == True
        :param symbol_count: the number of characters in the alphabet; used only when is_binary == False.
               the default value is None -- in this case the number of symbols is determined automatically based on the
               number of distinct characters in the `db`. However, we may need to override this number for machine
               learning tasks as a particular dataset may not have all the characters present in all the classes
        :param is_binary: are we dealing with binary strings?
        :param shots: the number of measurements to take
        :param quantum_instance: the pointer to the backend on which we want to execute the code
        :param optimize_for: architecture for which the code should be optimized, e.g., `FakeMontreal()`.
                             If none -- no optimization will be performed.
        :param optimization_levels: a list of optimization levels (QisKit transpiler takes values between 0 and 3)
        :param attempts_per_optimization_level: a number of times transpiler will be executed to find optimal circuit
        :param default_dataset: When True, this enables creation of a database containing all strings in equal superpositon.
                                When False, the database is initialized with strings passed in parameter `db`.
        :param p_pqm: When True, this will run the storage and retrieval algorithms of parametric probabilistic quantum memory
                      When False, this will run the extended p-pqm storage and retrieval algorithms
        :param t: parameter `t, a value within `(0, 1]` range is used by P-PQM algorithm to compute weighted Hamming distance, 
                  which may improve performance of machine learning classification. 
                  When `t=1` (the default value), P-PQM reduces to PQM.
        """
        self.t = t
        self.quantum_instance = quantum_instance
        self.shots = shots
        self.is_binary = is_binary
        self.default_dataset = default_dataset
        self.p_pqm = p_pqm

        if is_binary:  # check that strings contain only 0s and 1s
            self.symbol_length = symbol_length
            self.target_string, self.string_db = self._massage_binary_strings(target, db)
        else:
            self.target_string, self.string_db, self.symbol_length, self.symb_map = \
                self._massage_symbol_strings(target, db, symbol_count)

        self.input_size = len(self.target_string)

        logger.debug(f"Target string is '{self.target_string}'")
        logger.debug(f"Database is {self.string_db}")

        # Create Circuit
        if p_pqm:
            self.u_register_len = 2
            self.u_register = QuantumRegister(self.u_register_len)
            self.memory_register = QuantumRegister(self.input_size)
            self.pattern_register = QuantumRegister(self.input_size)
            self.qubits_range = self.u_register_len + self.input_size
            self.classic_register = ClassicalRegister(self.input_size + 1)

            self.circuit = QuantumCircuit(self.u_register, self.memory_register, self.pattern_register,
                                          self.classic_register)
            self._store_information_p_pqm(self.string_db)
            self._retrieve_information_P_PQM(self.target_string)

            self.circuit.measure(range(1, self.qubits_range), range(0, self.input_size + 1))

        else:
            self.u_register_len = 2
            self.u_register = QuantumRegister(self.u_register_len)
            self.memory_register = QuantumRegister(self.input_size)
            self.size_of_single_ham_register = math.floor(self.input_size / self.symbol_length)
            self.single_ham_dist_register = QuantumRegister(self.size_of_single_ham_register)
            self.qubits_range = self.size_of_single_ham_register + self.u_register_len + self.input_size
            self.classic_register = ClassicalRegister(self.qubits_range - self.size_of_single_ham_register - 1)
            self.circuit = QuantumCircuit(self.u_register, self.memory_register, self.single_ham_dist_register,
                                          self.classic_register)

            # TODO: we can use the attribute directly and not pass it as a parameter in the next two function calls

            if default_dataset:
                self._store_default_database(len(self.string_db[0]))
            if not default_dataset:
                self._store_information(self.string_db)

            self._retrieve_information(self.target_string)

            self.circuit.measure(range(1, self.qubits_range - self.size_of_single_ham_register),
                                 range(0, self.qubits_range - self.size_of_single_ham_register - 1))

        if optimize_for is not None:
            self._optimize_circuit(optimize_for, optimization_levels=optimization_levels,
                                   attempts_per_optimization_level=attempts_per_optimization_level)

        self.results = None

    def _optimize_circuit(self, backend_architecture, optimization_levels=range(0, 1),
                          attempts_per_optimization_level=1):
        """
        Try to optimize circuit by minimizing it's depth. Currently, it does a naive grid search.

        :param backend_architecture: the architecture for which the code should be optimized
        :param optimization_levels: a range of optimization levels
                                    (the transpiler currently supports the values between 0 and 3)
        :param attempts_per_optimization_level: the number of attempts per optimization level
        :return: None
        """
        cfg = backend_architecture.configuration()

        best_depth = math.inf
        best_circuit = None

        depth_stats = []

        for opt_level in optimization_levels:
            for attempt in range(0, attempts_per_optimization_level):
                optimized_circuit = transpile(self.circuit, coupling_map=cfg.coupling_map, basis_gates=cfg.basis_gates,
                                              optimization_level=opt_level) #, layout_method='sabre',
                                              # routing_method='sabre')
                current_depth = optimized_circuit.depth()
                depth_stats.append([opt_level, current_depth])
                if current_depth < best_depth:
                    best_circuit = optimized_circuit
                    best_depth = current_depth
        self.circuit = best_circuit

        self.optimizer_stats = pd.DataFrame(depth_stats, columns=['optimization_level', 'circuit_depth']).\
            groupby('optimization_level').describe().unstack(1).reset_index().\
            pivot(index='optimization_level', values=0, columns='level_1')
        logger.debug(f"Optimized depth is {self.circuit.depth()}")
        logger.debug(f"Summary stats of transpiler attempts{tabulate(self.optimizer_stats, headers='keys')}")

    def get_optimizer_stats(self):
        """
        Get optimizer stats

        :return: pandas data frame with the optimizer stats
        """
        try:
            return self.optimizer_stats
        except AttributeError:
            raise AttributeError("Optimizer was not invoked, no stats present")

    def _massage_binary_strings(self, target, db):
        """
        Massage binary strings and perform sanity checks

        :param target: target string
        :param db: database of strings
        :return: massaged target and database strings
        """
        # sanity checks
        if not isinstance(target, str):
            raise TypeError("Target string should be of type str")
        for my_str in db:
            if not isinstance(my_str, str):
                raise TypeError(f"Database string {my_str} should be of type str")

        bits_in_str_cnt = len(target)
        symbols_in_str_cnt = bits_in_str_cnt / self.symbol_length
        if bits_in_str_cnt % symbols_in_str_cnt != 0:
            raise TypeError(f"Possible data corruption: bit_count MOD symbol_length should be 0, but got "
                            f"{bits_in_str_cnt % symbols_in_str_cnt}")

        for my_str in db:
            if len(my_str) != bits_in_str_cnt:
                raise TypeError(
                    f"Target string size is {bits_in_str_cnt}, but db string {my_str} size is {len(my_str)}")

        if not self.is_str_binary(target):
            raise TypeError(
                f"Target string should be binary, but the string {target} has these characters {set(target)}")

        for my_str in db:
            if not self.is_str_binary(my_str):
                raise TypeError(f"Strings in the database should be binary, but the string {my_str} "
                                f"has these characters {set(my_str)}")

        return target, db

    @staticmethod
    def _massage_symbol_strings(target, db, override_symbol_count=None):
        """
        Massage binary strings and perform sanity checks

        :param target: target string
        :param db: database of strings
        :param override_symbol_count: number of symbols in the alphabet, if None -- determined automatically
        :return: target string converted to binary format,
                 database strings converted to binary format,
                 length of symbol in binary format,
                 map of textual symbols to their binary representation (used only for debugging)
        """

        # sanity checks
        if not isinstance(target, list):
            raise TypeError("Target string should be of type list")
        for my_str in db:
            if not isinstance(my_str, list):
                raise TypeError(f"Database string {my_str} should be of type list")

        # compute  strings' length
        symbols_in_str_cnt = len(target)
        for my_str in db:
            if len(my_str) != symbols_in_str_cnt:
                raise TypeError(
                    f"Target string has {symbols_in_str_cnt} symbols, but db string {my_str} has {len(my_str)}")

        # get distinct symbols
        symbols = {}
        id_cnt = 0
        for symbol in target:
            if symbol not in symbols:
                symbols[symbol] = id_cnt
                id_cnt += 1
        for my_str in db:
            for symbol in my_str:
                if symbol not in symbols:
                    symbols[symbol] = id_cnt
                    id_cnt += 1

        # override symbol length if symbol count was specified by the user
        dic_symbol_count = len(symbols)
        if override_symbol_count is not None:
            if dic_symbol_count > override_symbol_count:
                raise ValueError(f"Alphabet has at least {dic_symbol_count}, "
                                 f"but the user asked only for {override_symbol_count} symbols")
            dic_symbol_count = override_symbol_count

        # figure out how many bits a symbol needs
        symbol_length = math.ceil(math.log2(dic_symbol_count))
        logger.debug(f"We got {dic_symbol_count} distinct symbols requiring {symbol_length} bits per symbol")

        # convert ids for the symbols to binary strings
        bin_format = f"0{symbol_length}b"
        for symbol in symbols:
            symbols[symbol] = format(symbols[symbol], bin_format)

        # now let's produce binary strings
        # TODO: += is not the most efficient way to concatenate strings, think of a better way
        target_bin = ""
        for symbol in target:
            target_bin += symbols[symbol]

        db_bin = []
        for my_str in db:
            db_str_bin = ""
            for symbol in my_str:
                db_str_bin += symbols[symbol]
            db_bin.append(db_str_bin)

        return target_bin, db_bin, symbol_length, symbols

    def run(self, quantum_instance=None):
        """
        Execute the circuit and return a data structure with details of the results

        :param quantum_instance: the pointer to the backend on which we want to execute the code
               (overwrites the backend specified in the constructor)
        :return: a dictionary containing hamming distance and p-values for each string in the database, 
                 along with extra debug info (raw frequency count and the probability of measuring 
                 register c as 0)
        """
        if quantum_instance is not None:
            self.quantum_instance = quantum_instance

        job = execute(self.circuit, self.quantum_instance, shots=self.shots)
        results_raw = job.result().get_counts(self.circuit)

        # tweak raw results and add those strings that have 0 shots/pulses associated with them
        # these are the strings that will have hamming distance equal to the total number of symbols
        for string in self.string_db:
            full_binary_string = string[::-1] + "1"
            if full_binary_string not in results_raw:
                results_raw[full_binary_string] = 0

        # Massage results
        count_dic, useful_shots_count = self._get_count_of_useful_values(results_raw)
        p_values = []

        for my_str in self.string_db:
            p_values.append(count_dic[my_str] / useful_shots_count)

        probability_of_measuring_register_c_as_0 = float(sum(p_values))
        # re-normalize p-values, so that they sum up to 1.0
        if probability_of_measuring_register_c_as_0 != 0:  # else all values are zero anyway
            for ind in range(len(p_values)):
                p_values[ind] = p_values[ind] / probability_of_measuring_register_c_as_0

        ham_distances = self._convert_p_value_to_hamming_distance(p_values, probability_of_measuring_register_c_as_0)
        self.results = {'p_values': p_values,
                        'hamming_distances': ham_distances,
                        'prob_of_measuring_register_c_as_0': probability_of_measuring_register_c_as_0,
                        'raw_results': results_raw,
                        'useful_shots_count': useful_shots_count
                        }
        return self.results

    def get_circuit_depth(self):
        """
        Get circuit depth
        :return: circuit's depth
        """
        return self.circuit.depth()

    def get_transpiled_circuit_depth(self):
        """
        Get transpiled circuit depth
        :return: circuit's depth
        """
        return self.circuit.decompose().depth()

    def visualise_circuit(self, file_name):
        """
        Visualise circuit

        :param file_name: The name of the file to save the circuit to
        :return: None
        """
        self.circuit.draw(output='mpl', filename=file_name)

    def visualise_transpiled_circuit(self, file_name):
        """
        Visualise transpiled circuit

        :param file_name: The name of the file to save the circuit to
        :return: None
        """
        self.circuit.decompose().draw(output='mpl', filename=file_name)

    def debug_print_raw_shots(self):
        """
        Print raw pulse counts
        :return: None
        """
        print("Raw results")
        print(self.results['raw_results'])

    def debug_produce_histogram(self):
        """
        Generate histogram of raw pulse counts
        :return: None
        """
        print("Histogram")
        plot_histogram(self.results['raw_results'])

    def debug_produce_summary_stats(self):
        """
        Produce summary stats and print it
        :return: summary stats Pandas DataFrame
        """
        print("Summary stats")
        print(f"The number of useful shots is {self.results['useful_shots_count']} out of {self.shots}")
        # compute expected hamming distance
        string_db_expected_hd = []
        for my_str in self.string_db:
            string_db_expected_hd.append(
                self.hamming_distance(self.target_string, my_str, symbol_length=self.symbol_length))
        actual_vs_expected = self._test_output(self.string_db, string_db_expected_hd)
        print(tabulate(actual_vs_expected, headers='keys'))
        return actual_vs_expected

    @staticmethod
    def is_str_binary(my_str):
        """
        Check if a string contains only 0s and 1s

        :param my_str: string to check
        :return: True if binary, False -- otherwise
        """
        my_chars = set(my_str)
        if my_chars.issubset({'0', '1'}):
            return True
        else:
            return False

    def _get_count_of_useful_values(self, raw_results):
        """
        Get count of the strings present in the database and the useful number of shots

        :param raw_results: dictionary of registries and count of measurements
        :return: a dictionary of counts, number of useful shots
        """
        p_val_dic = {}
        suffix_length = 1
        useful_shots_count = 0
        for registry_value in raw_results:
            # assume that if the last two bits are set to `00` -- then we measure the degree of closeness
            # and are interested in this observation
            suffix = registry_value[-suffix_length:]

            # extract the middle of the string, which represents the original input
            input_string = registry_value[:-suffix_length]
            # it seems that the values of the strings are stored backward -- inverting
            input_string = input_string[::-1]

            # retain only the strings that were in the database
            # the rest are returned by the actual QC due to noise
            if input_string in self.string_db:
                input_string_cnt = raw_results[registry_value]
                useful_shots_count += input_string_cnt
                if suffix == '1':
                    p_val_dic[input_string] = input_string_cnt

        logging.debug(f"The useful number of shots is {useful_shots_count} out of {self.shots}")
        return p_val_dic, useful_shots_count

    def _store_information(self, logs):
        # Set up initial state
        self.circuit.x(self.u_register[1])
        for my_reg in range(self.size_of_single_ham_register):
            self.circuit.x(self.single_ham_dist_register[my_reg])

        # Load logs into memory register
        for ind in range(len(logs)):
            log = logs[ind]

            self._copy_pattern_to_memory_register(log)
            self.circuit.mct(self.memory_register, self.u_register[0])
            _x = len(logs) + 1 - (ind + 1)
            cs = Operator([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, math.sqrt((_x - 1) / _x), 1 / (math.sqrt(_x))],
                [0, 0, -1 / (math.sqrt(_x)), math.sqrt((_x - 1) / _x)]
            ])
            self.circuit.unitary(cs, [1, 0], label='cs')

            # Reverse previous operations
            self.circuit.mct(self.memory_register, self.u_register[0])
            self._copy_pattern_to_memory_register(log)

    def _store_information_p_pqm(self, logs):
        self.circuit.x(self.u_register[1])
        for i in range(len(logs)):
            string = logs[i]
            logging.debug(f"Processing {string}")
            j = len(string) - 1
            while (j >= 0):
                if (string[j] == '1'):
                    self.circuit.x(self.pattern_register[j])
                j -= 1

            for j in range(self.input_size):
                self.circuit.ccx(self.pattern_register[j], self.u_register[1], self.memory_register[j])

            for j in range(self.input_size):
                self.circuit.cx(self.pattern_register[j], self.memory_register[j])
                self.circuit.x(self.memory_register[j])

            self.circuit.mct(self.memory_register, self.u_register[0])

            x = len(logs) + 1 - (i + 1)
            cs = Operator([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, math.sqrt((x - 1) / x), 1 / (math.sqrt(x))],
                [0, 0, -1 / (math.sqrt(x)), math.sqrt((x - 1) / x)]
            ])

            self.circuit.unitary(cs, [1, 0], label='cs')

            self.circuit.mct(self.memory_register, self.u_register[0])

            for j in range(self.input_size):
                self.circuit.cx(self.pattern_register[j], self.memory_register[j])
                self.circuit.x(self.memory_register[j])

            for j in range(self.input_size):
                self.circuit.ccx(self.pattern_register[j], self.u_register[1], self.memory_register[j])

            j = len(string) - 1
            while (j >= 0):
                if (string[j] == '1'):
                    self.circuit.x(self.pattern_register[j])
                j -= 1

    def _store_default_database(self, length):
        for j in range(length):
            self.circuit.h(self.memory_register[j])
        for j in range(self.size_of_single_ham_register):
            self.circuit.x(self.single_ham_dist_register[j])

    def _fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(self, my_string):
        if not self.p_pqm:
            for j in range(self.input_size):
                if my_string[j] == "0":
                    self.circuit.x(self.memory_register[j])
        else:
            for j in range(self.input_size):
                if my_string[j] == "1":
                    self.circuit.x(self.memory_register[j])

    def _copy_pattern_to_memory_register(self, my_string):
        for j in range(len(my_string)):
            if my_string[j] == "1":
                self.circuit.cx(self.u_register[1], self.memory_register[j])
            else:
                self.circuit.x(self.memory_register[j])

    def _compare_input_and_pattern_for_single_ham_register(self):
        for j in range(self.size_of_single_ham_register):
            idx = self.symbol_length * j
            temp = []
            for ind in range(idx, idx + self.symbol_length):
                temp.append(ind + 2)
            self.circuit.mct(temp, self.single_ham_dist_register[j])

    def _retrieve_information(self, input_string):
        self.circuit.h(1)
        self._fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(input_string)
        self._compare_input_and_pattern_for_single_ham_register()
        u_gate = Operator([
            [math.e ** (complex(0, 1) * math.pi / (2 * ((self.input_size/self.symbol_length) * self.t))), 0],
            [0, 1]
        ])
        for ind in range(self.size_of_single_ham_register):
            self.circuit.unitary(u_gate, self.single_ham_dist_register[ind], label='U')
        u_minus_2_gate = Operator([
            [1 / math.e ** (complex(0, 1) * math.pi / ((self.input_size/self.symbol_length) * self.t)), 0],
            [0, 1]
        ])
        gate2x2 = UnitaryGate(u_minus_2_gate)
        gate2x2_ctrl = add_control(gate2x2, 1, 'CU2x2', '1')
        for j in range(self.size_of_single_ham_register):
            self.circuit.append(gate2x2_ctrl, [1, self.single_ham_dist_register[j]])

        # Reverse previous operations
        self._compare_input_and_pattern_for_single_ham_register()
        self._fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(input_string)

        self.circuit.h(1)

    def _retrieve_information_P_PQM(self, input_string):
        self.circuit.h(1)
        self._fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(input_string)

        u_gate = Operator([
            [math.e ** (complex(0, 1) * math.pi / (2 * ((self.input_size / self.symbol_length) * self.t))), 0],
            [0, 1]
        ])
        for ind in range(self.input_size):
            self.circuit.unitary(u_gate, self.memory_register[ind], label='U')
        u_minus_2_gate = Operator([
            [1 / math.e ** (complex(0, 1) * math.pi / ((self.input_size / self.symbol_length) * self.t)), 0],
            [0, 1]
        ])
        gate2x2 = UnitaryGate(u_minus_2_gate)
        gate2x2_ctrl = add_control(gate2x2, 1, 'CU2x2', '1')
        for j in range(self.input_size):
            self.circuit.append(gate2x2_ctrl, [1, self.memory_register[j]])

        # Reverse previous operations
        self._fill_ones_in_memory_register_which_are_equal_to_bits_in_pattern(input_string)

        self.circuit.h(1)


    @staticmethod
    def hamming_distance(str_one, str_two, symbol_length=1):
        """
        Compute hamming distance assuming that symbol may have more than one character

        :param str_one: first string
        :param str_two: second string
        :param symbol_length: the number of characters in a symbol (default is one)
        :return: Hamming distance
        """
        if len(str_one) != len(str_two):
            raise ValueError("Strings' lengths are not equal")

        sym_x = textwrap.wrap(str_one, symbol_length)
        sym_y = textwrap.wrap(str_two, symbol_length)

        return sum(s_x != s_y for s_x, s_y in zip(sym_x, sym_y))

    def _test_output(self, expected, expected_hd):
        """
        Produce stats to compare actual and expected values

        :param expected: the list of expected strings
        :param expected_hd: the list of expected hamming distances
        :return: summary stats as Pandas data frame
        """
        string_col_name = 'string'
        shots_count_col_name = 'shots_count'
        # massage expected ranking
        expected_ranking = pd.DataFrame(data={string_col_name: expected,
                                              'expected_hd': expected_hd
                                              })
        # cleanup actual output
        actual_ranking = pd.DataFrame(columns=[string_col_name, shots_count_col_name])
        actual = self.results['raw_results']
        count_dic, useful_shots_cnt = self._get_count_of_useful_values(actual)
        for input_string in count_dic:
            # the append is slow, but it will do for now
            actual_ranking = actual_ranking.append({string_col_name: input_string,
                                                    shots_count_col_name: count_dic[input_string]},
                                                   ignore_index=True)
        # sort observations from most common to list common
        actual_ranking.sort_values(by=[shots_count_col_name], ascending=False, inplace=True)

        # add shots fraction
        actual_ranking['shots_frac'] = actual_ranking[shots_count_col_name] / useful_shots_cnt
        # add actual ranks
        actual_ranking['actual_rank'] = range(len(actual_ranking))

        actual_computed = pd.DataFrame({
            string_col_name: self.string_db,
            'actual_p_value': self.results['p_values'],
            'actual_hd': self.results['hamming_distances']
        })

        # merge the tables
        actual_ranking = pd.merge(actual_ranking, actual_computed, on=string_col_name, how='outer')
        summary = pd.merge(actual_ranking, expected_ranking, on=string_col_name, how='outer')

        # sort
        summary.sort_values(by='expected_hd', inplace=True)

        # convert the strings back from binary to text representation
        if not self.is_binary:
            # "reverse" symbol lookup
            bin_code_map = dict((v, k) for k, v in self.symb_map.items())

            # reconstruct original text from binary strings
            # TODO: this can probably be vectorized
            for ind in summary.index:
                bin_str = summary.at[ind, string_col_name]
                txt_str = ""
                for symbol in textwrap.wrap(bin_str, self.symbol_length):
                    try:
                        txt_str += f"'{bin_code_map[symbol]}' "
                    except KeyError:
                        raise KeyError(f"Symbol {symbol} not found. "
                                       "Probably something is broken in conversion from text to bin")
                # get rid of last space
                txt_str = txt_str[:-1]

                # store original text
                summary.at[ind, string_col_name] = txt_str

        return summary

    def _convert_p_value_to_hamming_distance(self, p_values, prob_of_c):
        """
        Convert p-values into hamming distances
        :param p_values: p-values of strings
        :param prob_of_c: probability of measuring register c as 0
        :return: a list of Hamming distances
        """

        ham_distances = []
        for p_value in p_values:
            temp = 2 * prob_of_c * len(p_values) * p_value - 1
            if temp > 1:
                temp = 1.0
            ham_distances.append(int(round(((self.input_size/(self.symbol_length * math.pi)) * self.t) * (math.acos(temp)))))
        return ham_distances


if __name__ == "__main__":
    # To see debugging messages, uncomment the line below
    # logging.basicConfig(level=logging.DEBUG)

    # Example 1
    print("Example 1")
    # Normal execution
    dataset = ['1001', '1000', '1011', '0001', '1101', '1111', '0110']
    x = StringComparator('1001', dataset, symbol_length=2)
    results = x.run()
    print(f"probability of measuring register c as 0 is {results['prob_of_measuring_register_c_as_0']}")
    print(f"p-values are {results['p_values']}")
    print(f"hamming distances are {results['hamming_distances']}")
    # Extra debug info
    x.debug_print_raw_shots()
    x.debug_produce_summary_stats()
    print(f"Circuit's depth is {x.get_circuit_depth()}")
    print(f"Transpiled circuit's depth is {x.get_transpiled_circuit_depth()}")
    x.visualise_circuit("example1_circuit.pdf")
    x.visualise_transpiled_circuit("example1_transpiled_circuit.pdf")

    #     Example 2
    print("\nExample 2")
    # Normal execution
    dataset = ['01001', '11010', '01110', '10110']
    x = StringComparator('10110', dataset, symbol_length=1)
    results = x.run()
    print(f"probability of measuring register c as 0 is {results['prob_of_measuring_register_c_as_0']}")
    print(f"p-values are {results['p_values']}")
    print(f"hamming distances are {results['hamming_distances']}")
    # Extra debug info
    x.debug_print_raw_shots()
    x.debug_produce_summary_stats()
    print(f"Circuit's depth is {x.get_circuit_depth()}")
    print(f"Transpiled circuit's depth is {x.get_transpiled_circuit_depth()}")
    x.visualise_circuit("example2_circuit.pdf")
    x.visualise_transpiled_circuit("example2_transpiled_circuit.pdf")

    # Example 3
    print("\nExample 3")
    # Normal execution
    dataset = [['foo', 'bar', 'foo'],
               ['foo', 'bar', 'quux'],
               ['foo', 'quux', 'foo'],
               ['quux', 'bar', 'foo'],
               ['quux', 'baz', 'foo'],
               ['quux', 'baz', 'qux']
               ]
    target = ['foo', 'bar', 'foo']
    x = StringComparator(target, dataset, is_binary=False, shots=100000)
    # Note that we need to increase shots from 10K to 100K-- otherwise some times ranking gets broken
    results = x.run()
    print(f"probability of measuring register c as 0 is {results['prob_of_measuring_register_c_as_0']}")
    print(f"p-values are {results['p_values']}")
    print(f"hamming distances are {results['hamming_distances']}")
    # Extra debug info
    x.debug_print_raw_shots()
    x.debug_produce_summary_stats()
    print(f"Circuit's depth is {x.get_circuit_depth()}")
    print(f"Transpiled circuit's depth is {x.get_transpiled_circuit_depth()}")
    x.visualise_circuit("example3_circuit.pdf")
    x.visualise_transpiled_circuit("example3_transpiled_circuit.pdf")

    # Example 4
    # Now let's swap
    print("\nExample 4 -- this one is identical to Example 3, just in the binary form")
    # Normal execution
    dataset = ['000001000', '000001010', '000010000', '010001000', '010011000', '010011100']
    x = StringComparator('000001000', dataset,
                         symbol_length=3, shots=100000)
    results = x.run()
    print(f"probability of measuring register c as 0 is {results['prob_of_measuring_register_c_as_0']}")
    print(f"p-values are {results['p_values']}")
    print(f"hamming distances are {results['hamming_distances']}")
    # Extra debug info
    x.debug_print_raw_shots()
    x.debug_produce_summary_stats()
    print(f"Circuit's depth is {x.get_circuit_depth()}")
    print(f"Transpiled circuit's depth is {x.get_transpiled_circuit_depth()}")
    x.visualise_circuit("example4_circuit.pdf")
    x.visualise_transpiled_circuit("example4_transpiled_circuit.pdf")

    # Example 5
    print("\nExample 5 -- for the paper")
    # Normal execution
    dataset = [['foo', 'bar', 'foo'],  ['foo', 'bar', 'bar'], ['foo', 'quux', 'bar'], ['bar', 'foo', 'foo']]
    target = ['foo', 'quux', 'foo']
    x = StringComparator(target, dataset, is_binary=False, shots=10000)
    results = x.run()
    print(f"probability of measuring register c as 0 is {results['prob_of_measuring_register_c_as_0']}")
    print(f"p-values are {results['p_values']}")
    print(f"hamming distances are {results['hamming_distances']}")
    # Extra debug info
    x.debug_print_raw_shots()
    x.debug_produce_summary_stats()
    print(f"Circuit's depth is {x.get_circuit_depth()}")
    print(f"Transpiled circuit's depth is {x.get_transpiled_circuit_depth()}")
    x.visualise_circuit("example5_circuit.pdf")
    x.visualise_transpiled_circuit("example5_transpiled_circuit.pdf")

    # Example 6
    print("Example 6: test optimizer")
    # Normal execution
    dataset = ['1001', '1000']
    from qiskit.test.mock import FakeMontreal
    x = StringComparator('1001', dataset, symbol_length=2, shots=8192, optimize_for=FakeMontreal())
    # Extra debug info
    print(f"Optimized circuit's depth is {x.get_circuit_depth()}")
    print(f"Summary stats of transpiler's attempts\n{tabulate(x.get_optimizer_stats(), headers='keys')}")
    results = x.run(quantum_instance=FakeMontreal())
    x.debug_produce_summary_stats()
    print(f"Raw results: {results}")
