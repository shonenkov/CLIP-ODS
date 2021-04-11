# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup


def read(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as f:
        file_content = f.read()
    return file_content


def get_requirements():
    requirements = []
    for requirement in read('requirements.txt').splitlines():
        if requirement.startswith('git+') or requirement.startswith('svn+') or requirement.startswith('hg+'):
            parsed_requires = re.findall(r'#egg=([\w\d\.]+)-([\d\.]+)$', requirement)
            if parsed_requires:
                package, version = parsed_requires[0]
                requirements.append(f'{package}=={version}')
            else:
                print('WARNING! For correct matching dependency links need to specify package name and version'
                      'such as <dependency url>#egg=<package_name>-<version>')
        else:
            requirements.append(requirement)
    return requirements


def get_links():
    return [
        requirement for requirement in read('requirements.txt').splitlines()
        if requirement.startswith('git+') or requirement.startswith('svn+') or requirement.startswith('hg+')
    ]


def get_version():
    """ Get version from the package without actually importing it. """
    init = read('clip_ods/__init__.py')
    for line in init.split('\n'):
        if line.startswith('__version__'):
            return eval(line.split('=')[1])


setup(
    name='clip_ods',
    version=get_version(),
    author='Alex Shonenkov',
    author_email='shonenkov@phystech.edu',
    description='This lib is about a simple add-on over CLIP by OpenAI for Unsupervised Object Detection (Zeroshot).'
                ' You can search bounding boxes of objects using NATURAL LANGUAGE UNDERSTANDING - '
                'no classes, only text.',
    packages=['clip_ods'],
    package_data={'clip_ods': ['resources/*']},
    install_requires=get_requirements(),
    dependency_links=get_links(),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
)
