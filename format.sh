#!/bin/sh
set -ex
ufmt format -- *.py
autoflake --remove-all-unused-imports --in-place -- *.py
