#!/usr/bin/env bash
python -m build
twine upload dist/*
