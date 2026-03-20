from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'amr_zed2'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hdt',
    maintainer_email='hdt@example.com',
    description='AMR ZED2 Hand Sign Detection and Person Tracking with ByteTrack',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'handsign_detector = amr_zed2.handsign_detector_node:main',
        ],
    },
)
