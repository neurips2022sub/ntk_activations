"""Setup the package with pip.

Adapted from https://github.com/google/neural-tangents/blob/main/setup.py.
"""


import os
import setuptools


# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


INSTALL_REQUIRES = [
    'neural-tangents>=0.5.0'
]


def _get_version() -> str:
  """Returns the package version.

  Adapted from:
  https://github.com/deepmind/dm-haiku/blob/d4807e77b0b03c41467e24a247bed9d1897d336c/setup.py#L22

  Returns:
    Version number.
  """
  path = 'ntk_activations/__init__.py'
  version = '__version__'
  with open(path) as fp:
    for line in fp:
      if line.startswith(version):
        g = {}
        exec(line, g)
        return g[version]
    raise ValueError(f'`{version}` not defined in `{path}`.')


setuptools.setup(
    name='ntk-activations',
    version=_get_version(),
    license='Apache 2.0',
    author='Anonymous',
    author_email='neurips2022anon@gmail.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/neurips2022sub/ntk_activations',
    download_url='https://github.com/neurips2022sub/ntk_activations',
    project_urls={
        'Source Code': 'https://github.com/neurips2022sub/ntk_activations',
        'Paper': 'https://openreview.net/forum?id=yLilJ1vZgMe',
        'Bug Tracker': 'https://github.com/neurips2022sub/ntk_activations/issues',
        'Release Notes': 'https://github.com/neurips2022sub/ntk_activations/releases',
    },
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Fast Neural Kernel Embeddings for General Activations',
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Development Status :: 4 - Beta',
    ])
