FROM python:3.7
USER root

WORKDIR /FL
COPY . /FL

ARG BACKEND=keras
ENV PYTHONPATH="$PYTHONPATH:/FL"

RUN apt update && \
    apt install -y curl && \
    curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl && \
    chmod +x ./kubectl && \
    mv ./kubectl /usr/local/bin/kubectl
RUN chmod -R 0777 /FL
RUN pip install -U pip setuptools wheel
RUN pip install "$(find federated-learning-lib -name '*.whl')[$BACKEND]"

EXPOSE 8080
CMD ["/bin/bash"]
