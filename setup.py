from setuptools import setup, find_packages


def read(file) -> list:
    """读取requirements.txt，并转换为列表"""
    try:
        with open(file, 'r') as f:
            requirements = f.read().splitlines()
        return requirements
    except FileNotFoundError:
        return []


setup(
    name='hagike',
    version='0.0.3',
    packages=find_packages(),
    install_requires=read('requirements.txt'),
    include_package_data=True,
    package_data={
        'hagike': read('data.txt'),
    },
    description='hagike toolkit for everything',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='hagikehappy',
    author_email='hagikehappy@163.com',
    python_requires='>=3.10',
    classifiers=[  
        "Programming Language :: Python :: 3",  
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",  
    ],  
    project_urls={  
        'Homepage': 'https://github.com/hagikehappy/hagike',
        'Issues': 'https://github.com/hagikehappy/hagike/issues',
    },  
)

