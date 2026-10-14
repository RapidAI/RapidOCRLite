#!/usr/bin/env bash
# Rebuild SHA256SUMS on a GitHub Release from every asset currently attached
# (wheels, sdist and SDK archives) and upload it with --clobber. Both the
# wheel and SDK workflows call this after uploading, so whichever finishes
# last leaves a file that covers all assets.
# usage: release_sha256sums.sh TAG   (needs GH_TOKEN and GITHUB_REPOSITORY)
set -euo pipefail
TAG="$1"
work="$(mktemp -d)"
gh release view "$TAG" --repo "$GITHUB_REPOSITORY" --json assets --jq '.assets[].name' \
  | grep -vx 'SHA256SUMS' | sort > "$work/names"
mkdir "$work/assets"
while read -r name; do
  gh release download "$TAG" --repo "$GITHUB_REPOSITORY" --dir "$work/assets" --pattern "$name" --clobber
done < "$work/names"
(cd "$work/assets" && while read -r name; do sha256sum "./$name"; done < "$work/names") > "$work/SHA256SUMS"
cat "$work/SHA256SUMS"
gh release upload "$TAG" "$work/SHA256SUMS" --repo "$GITHUB_REPOSITORY" --clobber
