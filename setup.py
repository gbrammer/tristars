from distutils.core import setup

setup(name='tristars',
      version='0.1',
      description='Match coordinate lists with similar triangles',
      author='Gabriel Brammer',
      author_email='gbrammer@gmail.com',
      url='https://github.com/gbrammer/tristars',
      packages=['tristars'],
      scripts=[],
      requires=['numpy', 'scipy', 'skimage']
     )
