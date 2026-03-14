FROM python:3.14-slim AS build

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY tests/ tests/

RUN pip install --no-cache-dir -e ".[test]"

FROM python:3.14-slim
COPY --from=build /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=build /src /src
WORKDIR /src
ENTRYPOINT ["python", "-m", "pytest"]
