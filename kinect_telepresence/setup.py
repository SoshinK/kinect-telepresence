from setuptools import setup, find_packages
import glob
def get_data_files():
    from pathlib import Path 
    awb_path = Path(__file__).parent / 'kinect_telepresence'
    
    extensions = ['txt', 'xls', 'csv', 'json', 'pkl']
    data_files = []
    for ext in extensions:
        data_files.extend(
            [(str(f.parent), [str(f)]) for f in awb_path.rglob('*.' + ext)])
    return data_files
setup(
    name='kinect_telepresence',
    version='1.0',
    description='Different tools for Kinect Telepresence project (Skoltech)',
    #   url='',
    author='Konstantin Soshin',
    #   author_email='',
    # packages=find_packages(),
    packages=[
        'kinect_telepresence',
        'kinect_telepresence.camera',
        'kinect_telepresence.geometry',
        'kinect_telepresence.utils',
        ],
    package_data={
          'awb': ['*.txt', '*.xls', '*.csv', '*.json', '*.pkl'],
    },
    data_files=get_data_files(),
    include_package_data=True,
    zip_safe=False)