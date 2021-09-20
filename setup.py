from setuptools import setup

setup(
    name="aivle_grader",
    version="0.1.3",
    description="Auto grader for OpenAI Gym tasks.",
    url="https://github.com/edu-ai/aivle-grader",
    author="Yuanhong Tan",
    author_email="tan.yuanhong@u.nus.edu",
    packages=["aivle_grader", "aivle_grader.abc"],
    install_requires=["gym"],
    setup_requires=['wheel'],
    zip_safe=False,
)
