#!/bin/bash
find hagike -type f \( \
-name "*.c" -o -name "*.m" -o -name "*.py" -o -name "*.cpp" -o -name "*.cu" -o -name \
"*.txt" -o -name "*.json" -o -name "*.yaml" \
\) -print -exec wc -l {} +
