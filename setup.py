from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ras598_assignment_3'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('*.rviz')),
        (os.path.join('share', package_name, 'world'), glob('world/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Priyanka Dutta',
    maintainer_email='pdutta9@asu.edu',
    description='RAS598 Assignment 3: 3D Discrete Bayes Filter',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bayes_filter = ras598_assignment_3.bayes_boilerplate:main',
        ],
    },
)
