# Makefile for Stable Gym Sphinx documentation.

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
GH_PAGES_SOURCES = stable_gym

# Put it first so that "make" without argument is like "make help".
.PHONY: help Makefile
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Also remove the _autosummary folder on clean.
.PHONY: clean
clean:
	@rm -rf $(BUILDDIR)/*
	@rm -rf $(SOURCEDIR)/dev/api/_autosummary
	@echo "Build folder cleaned"

# Publish gh-pages command
.PHONY: gh-pages
gh-pages:
	cd .. && \
	git checkout gh-pages && \
	git rm -rf . && git clean -ffxd && \
	git checkout noetic $(GH_PAGES_SOURCES) && \
	git reset HEAD && \
	cd panda_gazebo/docs && \
	make clean && \
	make html && \
	cd .. && \
	git add docs/build/html && \
	git clean -ffxd && \
	mv -fv docs/build/html/* ../ && \
	cp -fv docs/build/html/.[!.]* ../ && \
	cd .. && \
	rm -rf $(GH_PAGES_SOURCES) && \
	git add -A && \
	git commit -m "docs: Generated gh-pages for `git log noetic -1 --pretty=short --abbrev-commit`" && \
	git push origin gh-pages && \
	git checkout noetic && \
	git submodule update --recursive && \
	pnpm install && \
	pnpm run prepare && \
	git restore pnpm-lock.yaml

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
