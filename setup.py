# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup


version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "dydim/version/version"), encoding="utf-8") as f:
    __version__ = f.read().strip()

install_requires = [
    "datasets",
    "mteb",
    "numpy",
    "pandas",
    "pyyaml",
    "scipy",
    "torch",
    "tqdm",
    "transformers",
]

TEST_REQUIRES = ["pytest"]

extras_require = {
    "test": TEST_REQUIRES,
}

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="dydim",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="",
    author="",
    author_email="",
    description="DyDim: dynamic dimension evaluation with attention-sum semantic vectors.",
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "dydim": ["version/version"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
