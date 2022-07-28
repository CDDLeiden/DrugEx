#!/usr/bin/env bash

set -e

COMMIT_ID=`git log -1 --pretty=short --abbrev-commit`
MSG="Adding docs to gh-pages for $COMMIT_ID"

BASE_DIR="`pwd`"
HTML_DIR="$BASE_DIR/_build/html/"

TMPREPO=/tmp/docs/$USER/DrugEx/
rm -rf $TMPREPO
mkdir -p -m 0755 $TMPREPO
git clone --single-branch --branch gh-pages `git config --get remote.origin.url` $TMPREPO

cd $TMPREPO
rm -rf "docs"
mkdir -p "docs"
cp -r $HTML_DIR/* "docs"
touch .nojekyll

git add -A
git commit -m "$MSG" && git push origin gh-pages
