# BloomFilter in Python

Setup the virtual environment for python via:

```bash
python3 -m venv venv
```

Activate the venv:

```bash
source venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

Install the test dependencies:

```bash
pip install -e ".[test]"
```

Run the full test suite from the project directory via:

```bash
python -m pytest
```

Close off the venv:

```bash
deactivate
```

You can also build and test in a Docker container.

Build the container:

```bash
docker build -t bloom-filter-python-test .
```

Run the tests in the container:

```bash
docker run --rm bloom-filter-python-test
```
