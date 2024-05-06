FROM nvcr.io/nvidia/pytorch:23.04-py3

COPY requirements.txt .

RUN apt-get update && pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /project

CMD ["python", "main.py"]