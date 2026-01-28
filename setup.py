from setuptools import setup, find_packages

setup(
    name="news-recommender",
    version="1.0.0",
    description="Контент-ориентированная рекомендательная система новостных статей",
    author="Elena",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "networkx>=3.1",
        "streamlit>=1.28.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)