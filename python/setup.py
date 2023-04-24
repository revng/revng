#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from setuptools import find_namespace_packages, setup

setup(
    name="revng",
    version="0.1.0",
    description="rev.ng Python libraries",
    author="Filippo Cremonese (rev.ng Labs)",
    author_email="filippocremonese@rev.ng",
    url="https://github.com/revng/revng",
    packages=find_namespace_packages(),
    include_package_data=True,
    install_requires=open("requirements.txt", encoding="utf-8").readlines(),  # noqa: SIM115
    scripts=["scripts/revng"],
)
