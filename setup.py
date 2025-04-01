from setuptools import setup, find_packages

setup(
    name="multicopy_refinement",
    version="0.1",
    packages=find_packages(exclude=["tests*", "test_data*"]),
    include_package_data=True,
    install_requires=[],  # Add any dependencies here
)