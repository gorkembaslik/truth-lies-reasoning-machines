"""Setup script for the truth-lies-reasoning-machines package."""

from setuptools import setup, find_packages

setup(
    name="truth-lies-reasoning-machines",
    version="0.1.0",
    description="Investigating LLM reasoning under truth distortion scenarios",
    author="Gorkem Baslik",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "google-cloud-aiplatform>=1.38.0",
        "google-generativeai>=0.3.0",
        "requests>=2.31.0",
        "nltk>=3.8.0",
        "rouge-score>=0.1.2",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
)