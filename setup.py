import re

from setuptools import setup


# access variables within .init
def get_property(prop):
    result = re.search(
        rf'{prop}\s*=\s*[\'"]([^\'"]*)[\'"]',
        open("dasp/__init__.py").read(),
    )
    return result.group(1)


with open("README.md", encoding="utf-8") as infile:
    long_description = infile.read()


setup(
    name="dasp",
    version=get_property("__version__"),
    description="Dimensionally aligned signal projection library",
    keywords=["signal processing", "signal", "visualization"],
    long_description_content_type="text/markdown",
    long_description=long_description,
    author="Jacob Smith",
    python_requires=">=3.9",
    url="https://github.com/ORNL/DASP",
    # project_urls={
    #     "Documentation": "https://ornl.github.io/dasp/latest/index.html"
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    packages=["dasp"],
    install_requires=[
        "numpy",
        "scipy",
        "ipywidget",
        "matplotlib",
        "ipython",
        "scikit-image",
    ],
)
