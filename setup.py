from setuptools import setup, find_packages

setup(
    name='gesund',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Open source SDK for Gesund.ai platform',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/gesund',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[],
    include_package_data=True,
)
