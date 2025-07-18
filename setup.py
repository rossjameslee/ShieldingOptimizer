"""
Setup script for ShieldingOptimizer package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="shielding-optimizer",
    version="1.0.0",
    author="Larsen, A., Lee, R., Wilson, C., Hedengren, J.D., Benson, J., Memmott, M.",
    author_email="memmott@byu.edu",
    description="ML-based optimization of nuclear reactor shielding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ShieldingOptimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Nuclear",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "nuclear",
        "reactor",
        "shielding",
        "optimization",
        "machine-learning",
        "gekko",
        "radiation",
        "physics",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ShieldingOptimizer/issues",
        "Source": "https://github.com/yourusername/ShieldingOptimizer",
        "Documentation": "https://github.com/yourusername/ShieldingOptimizer#readme",
    },
) 