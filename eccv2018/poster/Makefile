BUILD := \
	axonai-eccv2018-poster \


DEPS := \
	beamerthemegemini.sty \
	beamercolorthemegemini.sty \
	beamercolorthememit.sty \
	beamercolorthemelabsix.sty \


LATEX  := lualatex
BIBTEX := bibtex

TARGETS := $(patsubst %, %.pdf, $(BUILD))

# phony targets

all: $(TARGETS)

clean:
	rm -f *.pdf *.aux *.bbl *.blg *.log *.nav *.out *.snm *.toc

.PHONY: all clean

# main targets

axonai-eccv2018-poster.pdf: axonai-eccv2018-poster.tex axonai-eccv2018-poster.bib $(DEPS)
	$(eval SRC_$@ = $(patsubst %.tex, %, $<))
	$(LATEX) $(SRC_$@)
	$(BIBTEX) $(SRC_$@)
	$(LATEX) $(SRC_$@)
	$(LATEX) $(SRC_$@)
