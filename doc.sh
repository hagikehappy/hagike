#!/bin/sh
rm -r ./docs/hagike
pdoc --output-dir docs hagike --html
