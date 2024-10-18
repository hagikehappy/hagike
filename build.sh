#!/bin/sh
pip freeze > requirements.txt
python -m build

