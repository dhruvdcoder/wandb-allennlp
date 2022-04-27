from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "allennlp>=2.5.0,<2.10.0",
    "wandb>=0.10.11,<=0.12.15",
    "pyyaml",
    "tensorboard",
    "overrides",
    "shortuuid",
    # allennlp 2.9+ needs a newer version - this may break older versions
    # "nltk<3.6.6" # remove this once the support for older versions of ALLENNLP is dropped.
]

setup(
    name="wandb_allennlp",
    version="0.3.2",
    author="Dhruvesh Patel",
    author_email="1793dnp@gmail.com",
    description="Utilities to use allennlp with wandb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhruvdcoder/wandb-allennlp",
    packages=find_packages(
        where="src",
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "examples",
            "wandb",
        ],
    ),
    package_dir={"": "src"},
    package_data={"wandb_allennlp": ["py.typed"]},
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
