from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "allennlp>=2.5.0",
    "wandb>=0.8.35",
    "pyyaml",
    "tensorboard",
    "shortuuid",
]

setup(
    name="wandb_allennlp",
    version="0.3.0",
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
