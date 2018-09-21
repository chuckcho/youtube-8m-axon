#!/usr/bin/env bash

TEXFILE=axonai-eccv2018-paper
rm -f ${TEXFILE}.bbl
pdflatex ${TEXFILE} && bibtex ${TEXFILE} && pdflatex ${TEXFILE} && pdflatex ${TEXFILE}
