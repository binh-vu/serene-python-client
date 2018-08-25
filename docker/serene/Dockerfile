FROM java:8-jdk-alpine

RUN apk update && apk add git openssl bash && \
    wget https://piccolo.link/sbt-0.13.17.tgz && \
    tar xzf sbt-0.13.17.tgz

ENV PATH $PATH:/sbt/bin/

RUN git clone https://github.com/NICTA/serene.git && \
    cd serene && \
    git checkout c3059f35cb95b08a119fb375aae51e3d09e51a65 && \
    bin/build

WORKDIR serene
CMD ["sbt", "-mem", "8000", "run"]