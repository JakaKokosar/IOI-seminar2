#!/usr/bin/env python
import setuptools

if __name__ == '__main__':
    setuptools.setup(
        use_scm_version=True,
        setup_requires=[
            'setuptools-scm',
            'setuptools>=40.0'
        ],
        install_requires=[
            'Orange3~=3.23.1',
            'opencv-python~=4.1.2.30'
        ],
        extras_require={
            'doc': ['sphinx', 'recommonmark'],
            'test': [
                'flake8~=3.7.8',
                'flake8-comprehensions~=2.2.0',
                'flake8-black~=0.1.0',
                'pep8-naming~=0.8.2',
                'isort~=4.3.21',
                'pre-commit~=1.18.3',
                'pytest~=5.1.0',
                'pytest-cov~=2.7.1',
                'coverage~=4.5.4',
                'codecov~=2.0.15'
            ],
        },
    )