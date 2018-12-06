from setuptools import setup, find_packages

setup(
    name='camera_trap_classifier',
    url='https://github.com/marco-willi/camera-trap-classifier',
    author='Marco Willi',
    version='2.0.1',
    packages=find_packages(),
    package_data={'': ['*.yaml']},
    include_package_data=True,
    install_requires=[
        'pyyaml'
    ],
    extras_require={
        'tf': ['tensorflow==1.12'],
        'tf-gpu': ['tensorflow-gpu==1.12']
    },
    entry_points={
        'console_scripts': [
            'ctc.create_dataset_inventory = camera_trap_classifier.create_dataset_inventory:main',
            'ctc.create_dataset = camera_trap_classifier.create_dataset:main',
            'ctc.train = camera_trap_classifier.train:main',
            'ctc.predict = camera_trap_classifier.predict:main',
            'ctc.export = camera_trap_classifier.export:main'
            ]
    },
    python_requires='>=3.5'
)
