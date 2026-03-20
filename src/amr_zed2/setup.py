from setuptools import setup
from glob import glob
import os

package_name = 'amr_zed2'

setup(
    name=package_name,
    version='2.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Thuong Tran Dinh',
    maintainer_email='thuong.trandinh@hcmutcmut.edu.vn',
    description='ZED 2 hand sign detection and person tracking',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'handsign_detector = amr_zed2.handsign_detector_node:main',
        ],
    },
)
