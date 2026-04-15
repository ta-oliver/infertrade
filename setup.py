from pathlib import Path

from setuptools import find_packages, setup

from infertrade._version import __version__

README_PATH = Path(__file__).parent / "README.md"
LONG_DESCRIPTION = README_PATH.read_text(encoding="utf-8")

setup(
    name="infertrade",
    version=__version__,
    author="InferStat Ltd",
    description="InferTrade package for trading strategy research and performance analytics.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/ta-oliver/infertrade",
    license="Apache-2.0",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    install_requires=[
        "pandas>=1.2.4",
        "numpy>=1.22.2",
        "ta>=0.7.0",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.3.4",
        "typing_extensions>=3.7.4.3",
        "Markdown>=3.3.4",
        "requests>=2.32.4",
        "fonttools>=4.61.0",
        "pillow>=12.2.0",
        "urllib3>=2.6.3",
    ],
    extras_require={
        "dev": [
            "pytest==6.2.2",
            "sphinx==3.5.4",
            "pdftex==1.0.0",
            "myst-parser==0.14.0",
            "pytest-cov==2.12.1",
            "black==26.3.1",
            "GitPython>=3.1.18",
            "setuptools>=78.1.1",
            "zipp>=3.19.1",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 7 - Inactive",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
