from setuptools import setup, find_packages

setup(
    name="blobtracker",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "opencv-python",
        "numpy",
    ],
    python_requires=">=3.7",
)
