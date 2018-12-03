from setuptools import setup, find_packages

setup(
    name='camera-trap-classifier',
    url='https://github.com/marco-willi/camera-trap-classifier',
    author='Marco Willi',
    version='2.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow==1.12',
        'pyyaml',
        'pillow'
    ],
    python_requires='>=3.5'
)
