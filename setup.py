from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

setup(
    name="ds-risk-mcda",
    version="1.0.0",
    description="MCDA-based risk analysis for data science projects",
    long_description=long_desc,
    package_dir={"": "src"},
    packages=find_packages(include=["ds_risk_mcda"], where="src"),
    author="Alexander Oberst",
    author_email="alex@o-space.de",
    install_requires=["pyDecision", "openpyxl", "pyyaml"],
    package_data={"ds_risk_mcda": ["**/*"]},
    include_package_data=True,
)
