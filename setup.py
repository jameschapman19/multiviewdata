from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("./requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read()

setup(
    name="multiviewdata",
    version="0.0.0",
    packages=find_packages(),
    url="",
    license="",
    author="James Chapman",
    author_email="james.chapman.19@ucl.ac.uk",
    install_requires=REQUIRED_PACKAGES,
    description="Packaged data modules for multiview learning benchmarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
