projectVersion:=$(shell ./scripts/show-version.sh)
ifndef projectVersion
$(error $${projectVersion} is empty (not set))
endif

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
	${inDevEnvPy3} $${SCRIPTS_DIR}/workflow.sh
	${inEnv} if [[ ! -e $${BUILD_DIR}/workflow/training/model_final.pkl ]]; then exit 1; fi
