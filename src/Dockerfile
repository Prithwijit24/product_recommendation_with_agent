FROM python:3.11.14-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY project_folder project_folder
WORKDIR project_folder

# RUN pip install uv
# RUN uv venv pyvenv311 && uv pip install -r requirements.txt

RUN pip install -r requirements.txt


# RUN source pyvenv311/bin/activate
# RUN uv pip install -r requirements.txt


# ENV $MOTHERDUCK_TOKEN
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
