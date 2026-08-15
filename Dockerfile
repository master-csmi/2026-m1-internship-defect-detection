FROM python:3.11.9-slim


# do not write .pyc files, do not buffer the prints so we see the logs directly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app


COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -e ".[notebook,deep]"

COPY tests/ ./tests/
COPY notebooks/ ./notebooks/

RUN mkdir -p data/raw results

# a non-root user, and it must own /app to be able to write the results
# -m gives him a home folder, -s gives him a shell, then gives him /app. Without the chown he could not 
# write in /app/results and your notebook would fail every time you save a figure.
RUN useradd -m -s /bin/bash researcher && chown -R researcher:researcher /app
USER researcher

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]


