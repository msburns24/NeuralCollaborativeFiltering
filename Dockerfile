FROM python:3.11-slim

RUN pip install --no-cache-dir \
    tensorflow \
    keras \
    h5py \
    numpy \
    scipy

WORKDIR /home
CMD ["/bin/bash"]
