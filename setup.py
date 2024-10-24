import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chipsff",
    version="2024.10.20",
    author="Daniel Wines, Kamal Choudhary",
    author_email="daniel.wines@nist.gov",
    description="chipsff",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.6.3",
        "jarvis-tools>=2021.07.19",
        "pydantic_settings",
        # "alignn",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usnistgov/chipsff",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=[
        "chipsff/run_chipsff.py",
    ],
    python_requires=">=3.8",
)
