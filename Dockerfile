FROM ubuntu
COPY requirements.txt /
RUN apt-get update
RUN apt install python3-pip -y
RUN pip install -r requirements.txt
CMD "ls"