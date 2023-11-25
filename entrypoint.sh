#!/bin/sh
gunicorn -k uvicorn.workers.UvicornWorker src.main:app --bind=0.0.0.0:8000 --workers=4 --timeout=300 --log-level=debug --log-file=-