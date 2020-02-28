from setuptools import setup

setup(name='table1',
      version='0.1',
      description='Autogenerate pretty baseline characteristics table from wide data',
      url='http://github.com/chaudenschild/Table1',
      author='Christian Haudenschild',
      author_email='christian.c.haudenschild.jr.gr@dartmouth.edu',
      license='MIT',
      py_modules=['table1'],
      install_requires=['numpy', 'scipy', 'pandas', 'docx'],
      zip_safe=False)
