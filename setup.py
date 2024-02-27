import pathlib

import setuptools

# The directory containing this file
setuptools.setup(
    name="fedem",
    version="0.0.2",
    description="A simple example package ",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/mlsquare/",
    author="Seshu",
    author_email="seshu@mlsquare.com",
    license="MIT",
    project_urls={
        "Documentation": "https://github.com/mlsquare/",
        "Source": "https://github.com/mlsquare/seshu/",
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        # "Programming Language :: Python :: 3.11",
        # "Topic :: Large Language Models",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=["transformers", "torch"],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
