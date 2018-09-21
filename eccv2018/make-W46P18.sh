#!/usr/bin/env bash

TEXFILE=W46P18

cd ${TEXFILE}
rm -f ${TEXFILE}.pdf

cd source
rm -f ${TEXFILE}.bbl
pdflatex ${TEXFILE} && bibtex ${TEXFILE} && pdflatex ${TEXFILE} && pdflatex ${TEXFILE}

if [ $? -eq 0 ]; then
  echo --------------------------------------------------------------
  echo Everything looks good. Moving things around...
  mv ${TEXFILE}.pdf ..
  rm -f ${TEXFILE}.aux ${TEXFILE}.blg ${TEXFILE}.log
  cd ../..
  zip -r W46P18.zip W46P18
  echo --------------------------------------------------------------
  echo Done!
else
  echo --------------------------------------------------------------
  echo Compilation failed. Check!
fi
