#!/usr/bin/env python3

from setuptools import setup, find_namespace_packages


setup(
    name="revng",
    version="0.1.0",
    description="rev.ng python libraries",
    author="Filippo Cremonese (rev.ng SRLs)",
    author_email="filippocremonese@rev.ng",
    url="https://github.com/revng/revng",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=open("requirements.txt").readlines(),
    entry_points={
        "console_scripts": [
            "revng-merge-dynamic=revng.merge_dynamic:main",
            "revng-convert-idb=revng.idb_converter:main",
        ]
    },
)
