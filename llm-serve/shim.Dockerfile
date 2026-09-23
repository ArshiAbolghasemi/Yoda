# The OpenJev decision shim: a single stdlib HTTP server that turns typed
# questions into one prefill against vLLM and reads the option letters back as
# calibrated probabilities. It ships inside the model repo as helper/shim.py,
# so this image carries only its dependencies.
FROM python:3.14-slim

# Pinned to the versions published on the model card.
RUN pip install --no-cache-dir \
      "openai==3.16.2" \
      "httpx==0.28.1" \
      "transformers>=5.0.0"

# shim.py and the tokenizer both live on the shared /models volume.
WORKDIR /models
EXPOSE 3000
