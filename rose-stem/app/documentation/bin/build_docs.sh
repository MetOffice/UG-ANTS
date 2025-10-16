#!/bin/bash -l

# This is the canonical documentaion build.

# Any changes to this file should also be made to the top level build_docs.sh.

set -eu
echo "Building UGANTS Documentation"
cd ${DOCSDIR}
make clean
# API docs
sphinx-apidoc ../lib/ "../lib/*tests" -o source/lib/ --force --module-first --no-toc
# CLI docs
sphinx-apidoc ../bin/ -o source/applications --force --module-first --templatedir source/_templates/ -d 2 --tocfile app_index
make html
echo "Documentation built. "
echo "Run 'gio open ${DOCSDIR}/build/html/index.html' to view"
