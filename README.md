# String Comparison on a Quantum Computer Using Hamming Distance

This repository contains the files needed to compute the Hamming distance between a string and a set of strings
using a quantum computer; see [preprint](https://arxiv.org/abs/2106.16173) for details. 
The code resides in `string_comparison.py`.

## Setup
To set up the environment, run
```bash
pip install -r requirements.txt 
```

## Usage examples
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

## Citation
If you use the algorithm or code, please cite them as follows:
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

## Contact us
If you found a bug or came up with a new feature -- 
please open an [issue](https://github.com/miranska/qc-str/issues) 
or [pull request](https://github.com/miranska/qc-str/pulls).
