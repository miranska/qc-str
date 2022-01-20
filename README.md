# String Comparison on a Quantum Computer

This repository contains the files needed to compare and classify the strings using Hamming distance between a string and a set of strings using a quantum computer.
For computing the distance between a target string and the closest string in the group of strings, see [preprint](https://arxiv.org/abs/2106.16173). The code for this preprint is packaged in [v0.1.0](https://github.com/miranska/qc-str/releases/tag/v0.1.0). The core code resides in `string_comparison.py`.

Furthermore, this repository extends the above codebase by creating an efficient version of the Parametric Probabilistic Quantum Memory ([P-PQM](https://doi.org/10.1016/j.neucom.2020.01.116)) approach for computing the probability of a string belonging to a particular group of strings (i.e., a machine learning classification problem). We call our algorithm EP-PQM, see [preprint](https://arxiv.org/abs/2201.07265) for details. The code for this preprint is packaged in [v0.2.0](https://github.com/miranska/qc-str/releases/tag/v0.2.0).


## Setup
To set up the environment, run
```bash
pip install -r requirements.txt 
```

## Usage examples
### Computing the distance between a target string and the closest string in a group
`test_string_comparison.py` contains unit test cases for `string_comparison.py`. This file can also be interpreted as a 
set of examples of how `StringComparator` in `string_comparison.py` should be invoked. To run, do
```bash
python -m unittest test_string_comparison.py
```
For additional examples (containing invocations of the debug-related methods), run 
```bash
python string_comparison.py
```
To execute code listings shown in the [preprint](https://arxiv.org/abs/2106.16173), do
```bash
python hd_paper_listings.py
```

### EP-PQM

The file `compute_empirical_complexity.py` simulates the generation of quantum circuits for string classification as described in the [preprint](https://arxiv.org/abs/2201.07265). Datasets found in `./datasets` (namely, Balance Scale, Breast Cancer, SPECT Heart, Tic-Tac-Toe Endgame, and Zoo) are taken from the UCI Machine Learning [Repository](https://archive.ics.uci.edu/ml/index.php).
To execute, run
```bash
python compute_empirical_complexity.py
```
The output is saved in `stats.csv` and `stats.json`.


## Citation
If you use the algorithm or code, please cite them as follows. For computing Hamming distance:
```bibtex
@article{khan2021string,
  author        = {Mushahid Khan and Andriy Miranskyy},
  title         = {{String Comparison on a Quantum Computer Using Hamming Distance}},
  journal       = {CoRR},
  volume        = {abs/2106.16173},
  year          = {2021},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2106.16173},
  eprint        = {2106.16173}
}
```

For EP-PQM:
```bibtex
@article{khan2022string,
  author        = {Mushahid Khan and Jean Paul Latyr Faye and Udson C. Mendes and Andriy Miranskyy},
  title         = {{EP-PQM: Efficient Parametric Probabilistic Quantum Memory with Fewer Qubits and Gates}},
  journal       = {CoRR},
  volume        = {abs/2201.07265},
  year          = {2022},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2201.07265},
  eprint        = {2201.07265}
}
```

## Contact us
If you found a bug or came up with a new feature -- 
please open an [issue](https://github.com/miranska/qc-str/issues) 
or [pull request](https://github.com/miranska/qc-str/pulls).
