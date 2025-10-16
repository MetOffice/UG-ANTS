#!/bin/bash -l
set -eu
echo "Checking for broken links"
cd ${DOCSDIR}
make linkcheck
