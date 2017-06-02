tell:=../scripts/tell_fast5.py

fast5Files:=$(wildcard *.fast5)
schemas:=$(patsubst %.fast5,%.schema,${fast5Files})

.PHONY: all
all: md5sum

.PHONY: schemas
schemas: ${schemas}

md5sum: ${schemas}
	md5sum ${schemas} > $@

${schemas}: %.schema: %.fast5 ${tell}
	${tell} --schema $< > $@

.PHONY: test
test:
	md5sum -c md5sum
	${tell} --hash ${fast5Files} --schema_extension > md5sum.reference
	md5sum -c md5sum.reference
