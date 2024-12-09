#!/bin/bash
set -e

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

/usr/bin/python3 -u create_engines.py
/usr/bin/python3 -m uvicorn main:app --workers 1 --host 0.0.0.0 --port 9001
