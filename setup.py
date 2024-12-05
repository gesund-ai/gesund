from setuptools import setup

setup(
    packages=["gesund-src"],
    install_requires=[
        "bson",
        "jsonschema",
        "scikit-learn",
        "pandas",
        "seaborn",
        "opencv-python",
        "dictances==1.5.3",
        "miseval==1.2.2",
        "numpy==1.21.6",
        "numba==0.55.2",
        "tqdm",
        "pycocotools",
    ],
    dependency_links=[
        "git+https://github.com/HammadK44/cocoapi.git@Dev#egg=pycocotools&subdirectory=PythonAPI"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
