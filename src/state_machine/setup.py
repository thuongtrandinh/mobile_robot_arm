from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'state_machine'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hdt',
    maintainer_email='thien1932004@gmail.com',
    description='State machine for mobile robot arm with ZED2 optimization',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'state_machine = state_machine.state_machine_node:main',
        ],
    },
)
