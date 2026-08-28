from setuptools import setup, find_packages

setup(
    name="AMuSA",
    version="0.1.0",

    description="Mutational signature assignment framework",

    author="Yang Ying",

    packages=find_packages(),

    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ],

    python_requires=">=3.10",
)
