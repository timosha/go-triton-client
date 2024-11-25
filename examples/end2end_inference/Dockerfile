FROM golang:1.23.0 AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y curl

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

RUN git clone https://github.com/berkayersoyy/triton-ty-models.git /repo && \
    cd /repo && git lfs pull && \
    echo "Repository cloned with LFS files pulled"

RUN cp -r /repo/models /models && \
    cp -r /repo/tokenizers /tokenizers

RUN rm -rf /repo

COPY . .

RUN apt-get update && apt-get install -y gcc

ENV CGO_ENABLED=1

RUN go mod tidy && go mod download

RUN go build -o main .

FROM nvcr.io/nvidia/tritonserver:23.08-py3

RUN apt-get update && apt-get install -y curl

COPY --from=builder /models /models
COPY --from=builder /tokenizers /tokenizers
COPY --from=builder /app/main /main

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000 8001 8002

ENTRYPOINT ["/start.sh"]