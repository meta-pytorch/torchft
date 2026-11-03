FROM ghcr.io/pyo3/maturin:v1.16.0@sha256:50bf2ff99eeff7b09332bad9baf4a5e925e2502304ae2f88f48ce8c06fd9957c

RUN PB_REL="https://github.com/protocolbuffers/protobuf/releases" && curl -LO $PB_REL/download/v25.1/protoc-25.1-linux-x86_64.zip
RUN unzip protoc-25.1-linux-x86_64.zip -d /usr/

#ENTRYPOINT ["/bin/bash"]