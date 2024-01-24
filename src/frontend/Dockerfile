FROM python:3.9-slim

WORKDIR /frontend

COPY requirements.txt ./
# RUN pip install -U pip
# RUN apt-get update && apt-get install -y
# RUN pip install -r requirements.txt
RUN pip install -r requirements.txt


COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT [ "streamlit", "run", "./app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]