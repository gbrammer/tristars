#from distutils.core import setup
from setuptools import setup

version = "0.1" # First tagged version

version_str = """# 
__version__ = "{0}"\n""".format(version)

fp = open('tristars/version.py','w')
fp.write(version_str)
fp.close()


setup(name='tristars',
      version=version,
      description='Match coordinate lists with similar triangles',
      author='Gabriel Brammer',
      author_email='gbrammer@gmail.com',
      url='https://github.com/gbrammer/tristars',
      packages=['tristars'],
      scripts=[],
      install_requires=['numpy', 'scipy', 'scikit-image']
     )
