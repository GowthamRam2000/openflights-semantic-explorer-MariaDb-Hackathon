FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY frontend /app/frontend

EXPOSE 8501

CMD [
  "streamlit",
  "run",
  "frontend/streamlit_app.py",
  "--server.address=0.0.0.0",
  "--server.port=8501"
]
