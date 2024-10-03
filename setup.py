"""Python setup.py for nd_vq_vae package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("nd_vq_vae", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="nd_vq_vae",
    version=read("nd_vq_vae", "VERSION"),
    description="PyTorch Implementation of an N-Dimensional VQ-VAE by AdityaNG",
    url="https://github.com/AdityaNG/nD_VQ_VAE/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="AdityaNG",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["nd_vq_vae = nd_vq_vae.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
