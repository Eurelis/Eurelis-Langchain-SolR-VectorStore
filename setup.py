from setuptools import setup

setup(
    name='eurelis_langchain_solr_vectorstore',
    version='0.0.1',
    author='Jérôme DIAZ',
    author_email='j.diaz@eurelis.com',
    install_requires=['langchain', 'requests', 'numpy'],
    packages=['eurelis_langchain_solr_vectorstore']
)
