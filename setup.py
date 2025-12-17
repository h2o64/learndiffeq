import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

with open("requirements.txt", "r") as file:
    requirements = [line.strip() for line in file]

setuptools.setup(
    name="learndiffeq",
    version="0.0.1",
    author="LG, MA, NB, MG, MH, EVE ",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
