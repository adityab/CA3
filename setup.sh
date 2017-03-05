#!/bin/bash

BOLD="\e[1m"
GREEN="\e[32m"
RESET="\e[0m"

# Build SONETS
echo -e "$BOLD* Building SONETS program... $RESET"
unzip sonets.zip
mv sonets-master sonets
cd sonets
make
echo -e "$BOLD$GREEN* Done! $RESET"

exit