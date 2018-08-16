#!/bin/bash
python3 extract_features.py
python3 xgb.py
python3 combine.py
