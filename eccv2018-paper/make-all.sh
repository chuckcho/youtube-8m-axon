#!/usr/bin/env bash

TEXFILE=axonai-eccv2018
rm -f ${TEXFILE}.bbl
pdflatex ${TEXFILE} && bibtex ${TEXFILE} && pdflatex ${TEXFILE} && pdflatex ${TEXFILE}
