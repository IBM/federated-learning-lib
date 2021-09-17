FROM centos/python-36-centos7:latest
USER root
RUN ls

WORKDIR /FL
COPY . /FL

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install virtualenv
RUN virtualenv venv
RUN source venv/bin/activate
RUN pip3 install -q -r requirements.txt
RUN pip3 install federated-learning-lib/federated_learning_lib-*-py3-none-any.whl

CMD ["/bin/bash"]