# ==============================================================================
# Usage

define USAGE_OPTIONS

Options:
	pub				publish to code registry .
	pub2local		publish to local maven repo.
endef
export USAGE_OPTIONS

## help: Show this help info.
.PHONY: help
help: Makefile
	@printf "\nUsage: make <TARGETS> <OPTIONS> ...\n\nTargets:\n"
	@sed -n 's/^##//p' $< | column -t -s ':' | sed -e 's/^/ /'
	@echo "$$USAGE_OPTIONS"



define gradle
./gradlew $(1)
endef

.PHONY: pub2local clean build
pub2local:
	$(call gradle, clean publishToMavenLocal  -x test)
clean:
	$(call gradle, clean -x test)
build:
	$(call gradle, clean build-x test)