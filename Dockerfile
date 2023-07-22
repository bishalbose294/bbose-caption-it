FROM python:3.10.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

COPY ./packages.txt /code/packages.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

ENV TRANSFORMERS_CACHE=/code/hf_model
ENV HF_HOME=/code/hf_model
ENV HF_DATASETS_CACHE=/code/hf_model
ENV XDG_CACHE_HOME=/code/hf_model

RUN chmod -R 777 .

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]