#!/bin/sh
find docs/hagike -type f -name "*.html" -delete
pdoc --output-dir docs hagike --html --force --config latex_math=True
