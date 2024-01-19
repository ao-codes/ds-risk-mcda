# ds-risk-mcda

Description
-----------

**ds-risk-mcda** is a Python library that is intended to enable researchers to analyze risks of data
science projects by using procedures from so-called multi-criteria decision analysis. In the current
version, ds-risk-mcda can apply two popular mcda approaches, the Analytic Hierarchy Process (AHP)
and a combined approach of AHP and the Technique for Order Preference by Similarity to Ideal
Solution (AHP-TOPSIS). It creates Excel-based assessment files, which can be filled out by experts
or risk analysts, and can then read the filled Excel files back in to carry out the corresponding
analysis. In addition, ds-risk-mcda has rudimentary options for simulating risk analysis using the
methods mentioned.


Installation
------------
ds-risk-mcda is not yet available on the Python Package Index (PyPI). To install follow these steps:

#### 1. download or clone this repo
#### 2. build a python-wheel
```
python setup.py bdist_wheel
```
#### 3. install wheel for your virtual environment
```
pip install /path/to/ds_risk_mcda-1.0.0-py3-none-any.whl
```


Example usage
-------------
#### 1. Generate an AHP-assessment template (xlsx-file)

```py3
from ds_risk_mcda.assessments import AHPAssessment

# providing initial data from a risk breakdown structure (RBS)
# input may also be a pandas dataframe
RBS_data = {
    "Risks": ["R-01", "R-02", "R-03", "R-04", "R-05"],
    "Categories": ["C-01", "C-01", "C-02", "C-02", "C-02"],
}

# instantiating an AHPAssessment and preparing an assessment file
assessment = AHPAssessment(RBS_data)
assessment.prepare_excel_assessment("test.xlsx")
```
#### 2. Fill out the pairwise comparison matrices in the template.
#### 3. Execute risk analyses
```py3
from ds_risk_mcda.assessments import AHPAssessment
from ds_risk_mcda.analyzers import AHPAnalyzer

# read template data
pcm_level_1, pcms_level_2, _ = AHPAssessment.read_excel_assessment("test.xlsx")

# instantiate analyzer and perform risk analysis
analyzer = AHPAnalyzer(pcm_level_1, pcms_level_2)
risk_analysis, consistencies, _ = analyzer.perform_analysis(verbose=True)

# show results
print(risk_analysis)
print(consistencies)

```


Running unit-tests
------------------
1. If not already done, create a virtual environment in the folder of the unpacked distribution.
2. Install the required packages.
```
pip install pyDecision openpyxl pyyaml pytest
```
3. Execute command pytest -v
