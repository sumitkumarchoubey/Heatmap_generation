#!/bin/bash
#pip install -r requirements.txt
gunicorn -w 1 --threads 1 --bind 0.0.0.0:4036  wsgi:app --timeout 1000
