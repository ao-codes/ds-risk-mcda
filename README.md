# ds-risk-mcda

Description
-----------

ds-risk-mcda is a Python library that is intended to enable researchers to analyze risks of data
science projects by using procedures from so-called multi-criteria decision analysis. In the current
version, ds-risk-mcda can apply two popular mcda approaches, the Analytic Hierarchy Process (AHP)
and a combined approach of AHP and the Technique for Order Preference by Similarity to Ideal
Solution (AHP-TOPSIS). It creates Excel-based assessment files, which can be filled out by experts
or risk analysts, and can then read the filled Excel files back in to carry out the corresponding
analyses. In addition, ds-risk-mcda has rudimentary options for simulating risk analyzes using the
methods mentioned.


Installation
------------
ds-risk-mcda is not yet available on the Python Package Index (PyPI). To install follow these steps:

1. download or clone this repo
2. build a python-wheel
```
python setup.py bdist_wheel
```
3. install wheel for your virtual environment
```
pip install /path/to/ds_risk_mcda-1.0.0-py3-none-any.whl
```


Example usage
-------------
```py3
print("hello world")
```


running unit-tests
------------------
1. If not already done, create a virtual environment in the folder of the unpacked distribution.
2. Install the required packages.
```
pip install pyDecision openpyxl pyyaml pytest
```
3. Execute command pytest -v
