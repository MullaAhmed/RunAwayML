FROM python:3.10.6-slim-buster
COPY . .
RUN sh build.sh
EXPOSE 5000
CMD ["python", "api.py"]
