from setuptools import setup, find_packages

setup(
    name="spam_email_classifier",
    version="0.1.0",
    author="Jose Navarro Meneses",
    author_email="josenavarrojmx@gmail.com",
    description="A machine learning project to classify spam emails using logistic regression and Naive Bayes.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "kagglehub"
    ],
    entry_points={
        "console_scripts": [
            # Optional: You can define CLI commands here
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    test_suite="tests",
)