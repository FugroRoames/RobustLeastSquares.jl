#!/usr/bin/env make

.PHONY: default
default: help

CONTAINER_NAME = robust_least_square.jl

.PHONY: docker-build
## Builds a docker container with all the dependencies (REQUIRE)
docker-build:
	docker build -t $(CONTAINER_NAME):latest .

.PHONY: docker-test
## Runs tests in docker container
docker-test: docker-build
	docker run -ti --env "JULIA_LOAD_PATH=:/home/src" $(CONTAINER_NAME):latest julia "test/runtests.jl"

.PHONY: test
## Run tests locally
test:
	JULIA_LOAD_PATH=$$PWD/src julia test/runtests.jl

################################ HELPER TARGETS - DO NOT EDIT #############################
## `help` target will show description of each target
## Target description should be immediate line before target starting with `##`

# COLORS
RED    := $(shell tput -Txterm setaf 1)
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

TARGET_MAX_CHAR_NUM=20
## Show help
help:
	@echo ''
	@echo 'Usage:'
	@echo '  $(YELLOW)make$(RESET) $(GREEN)<target>$(RESET)'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			split($$1, arr, ":"); \
			helpCommand = arr[1]; \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  $(YELLOW)%-$(TARGET_MAX_CHAR_NUM)s$(RESET) $(GREEN)%s$(RESET)\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)

.PHONY: help
