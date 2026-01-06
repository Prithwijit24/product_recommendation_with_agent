FROM ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN apt-get update && apt-get install -y \
	ca-certificates \
	libgl1 \
  libgomp1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*



RUN uv sync

COPY . /app/

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "src/project_folder/app.py"]
