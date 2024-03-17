import pathlib

import setuptools

setuptools.setup(
    name="fedem",
    version="0.0.7",
    description="A decentralized framework to train foundational models",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/mlsquare/fedem",
    author="MLSquare",
    author_email="mail@mlsquare.com",
    license="MIT",
    project_urls={
        "Documentation": "https://mlsquare.github.io/fedem/",
        "Source": "https://github.com/mlsquare/fedem/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=[
        "transformers",
        "torch",
        "datasets",
        "einops",
        "accelerate",
        "peft",
        "huggingface-hub",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
