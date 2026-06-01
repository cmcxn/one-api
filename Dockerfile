FROM node:16 as builder

WORKDIR /web
COPY ./VERSION .
COPY ./web .

WORKDIR /web/default
RUN npm install
RUN DISABLE_ESLINT_PLUGIN='true' REACT_APP_VERSION=$(cat VERSION) npm run build

WORKDIR /web/berry
RUN npm install
RUN DISABLE_ESLINT_PLUGIN='true' REACT_APP_VERSION=$(cat VERSION) npm run build

WORKDIR /web/air
RUN npm install
RUN DISABLE_ESLINT_PLUGIN='true' REACT_APP_VERSION=$(cat VERSION) npm run build

FROM golang AS builder2

ENV GO111MODULE=on \
    CGO_ENABLED=1 \
    GOOS=linux

WORKDIR /build
ADD go.mod go.sum ./
RUN go env -w GOPROXY=https://goproxy.cn,direct
RUN go env -w GOSUMDB=goproxy.cn/sumdb/sum.golang.org
RUN go mod download

# Pre-download tiktoken encoding files so the container can start without internet access.
# The tiktoken-go library caches files as SHA1(url) under TIKTOKEN_CACHE_DIR.
# cl100k_base is used by gpt-3.5-turbo and gpt-4; o200k_base is used by gpt-4o.
RUN mkdir -p /tiktoken_cache && \
    wget -O /tiktoken_cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" && \
    wget -O /tiktoken_cache/fb374d419588a4632f3f557e76b4b70aebbca790 \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"

COPY . .
COPY --from=builder /web/build ./web/build
RUN go build -ldflags "-s -w -X 'github.com/songquanpeng/one-api/common.Version=$(cat VERSION)' -extldflags '-static'" -o one-api

FROM alpine

RUN apk update \
    && apk upgrade \
    && apk add --no-cache ca-certificates tzdata \
    && update-ca-certificates 2>/dev/null || true

COPY --from=builder2 /build/one-api /
COPY --from=builder2 /tiktoken_cache /tiktoken_cache
ENV TIKTOKEN_CACHE_DIR=/tiktoken_cache
EXPOSE 3000
WORKDIR /data
ENTRYPOINT ["/one-api"]