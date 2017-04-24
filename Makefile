sloikaVersion:=$(shell ./scripts/show-version.sh)
ifndef sloikaVersion
$(error $${sloikaVersion} is empty (not set))
endif
whlFile:=dist/sloika-${sloikaVersion}-cp27-cp27mu-linux_x86_64.whl
whlFilePy3:=dist/sloika-${sloikaVersion}-cp35-cp35m-linux_x86_64.whl

pyDirs:=sloika test bin models misc
pyFiles:=$(shell find *.py ${pyDirs} -type f -regextype sed -regex ".*\.py")

include Makefile.common

.PHONY: deps
deps:
	apt-get update
	apt-get install -y \
	    python-virtualenv python-pip python-setuptools ont-ca-certs git \
	    libblas3 libblas-dev python-dev python3-dev lsb-release bwa

.PHONY: workflow
workflow:
	${inDevEnv} $${SCRIPTS_DIR}/workflow.sh
	${inEnv} if [[ ! -e $${BUILD_DIR}/workflow/training/model_final.pkl ]]; then exit 1; fi

include Makefile.res
