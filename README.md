# String Comparison on a Quantum Computer Using Hamming Distance

This repository contains the files needed to compute the Hamming distance between a string and a set of strings. 
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
