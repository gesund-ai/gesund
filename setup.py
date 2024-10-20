from setuptools import setup, find_packages

setup(
    name='gesund',
    version='0.3.17',
    author='Gesund AI',
    author_email='veysel@gesund.ai',
    license="MIT",
    description='Gesund.ai package for running validation metrics for classification, semantic segmentation, instance segmentation, and object detection models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gesund-ai/gesund',
    packages=find_packages(),
    install_requires=[
        'bson',
        'jsonschema',
        'scikit-learn',
        'pandas',
        'seaborn',
        'opencv-python',
        'dictances==1.5.3',
        'miseval==1.2.2',
        'numpy==1.21.6',
        'numba==0.55.2',
        'tqdm',
        'pycocotools',
    ],
    dependency_links=[
        'git+https://github.com/HammadK44/cocoapi.git@Dev#egg=pycocotools&subdirectory=PythonAPI'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
