FROM ubuntu

WORKDIR /src

RUN apt-get update
RUN apt-get -y install python3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pandas \
    python3-matplotlib \
    python3-sklearn \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY recursive_ts_forecasting.py ./recursive_ts_forecasting.py
COPY co2.csv ./co2.csv

CMD ["python3", "recursive_ts_forecasting.py"]
