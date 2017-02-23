sloikaVersion:=$(shell ./scripts/show-version.sh)
ifndef sloikaVersion
$(error $${sloikaVersion} is empty (not set))
endif
whlFile:=dist/sloika-${sloikaVersion}-cp27-cp27mu-linux_x86_64.whl

pyDirs:=sloika test bin models misc
pyFiles:=$(shell find *.py ${pyDirs} -type f -regextype sed -regex ".*\.py")

include Makefile.common

.PHONY: deps
deps:
	apt-get update
	apt-get install -y \
	    python-virtualenv python-pip python-setuptools ont-ca-certs git \
	    libblas3 libblas-dev python-dev python3-dev lsb-release

include Makefile.res
