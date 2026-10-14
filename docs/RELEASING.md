# Releasing RapidOCRLite

The PyPI distribution name is `RapidOCRLite`. The import name is still `rapidocr`.

The workflow is `.github/workflows/build-wheels.yml`.

It builds CPython 3.9–3.13 wheels inside the official manylinux2014 images
(devtoolset GCC 10, glibc 2.17 / `manylinux_2_17`) and an sdist with
`python -m build --sdist`. x86_64 uses `ubuntu-latest`. aarch64 uses the
native `ubuntu-24.04-arm` runner. If that job never starts, the same
aarch64 build runs under QEMU on `ubuntu-latest`. After `auditwheel repair`,
the workflow rejects a wheel whose filename and `auditwheel show` output are
not `manylinux2014` or `manylinux_2_17`.

macOS wheels are CPython 3.9–3.13 `macosx_14_0_arm64` builds from the
`wheels-macos-arm64` job on `macos-14`, with the Metal backend compiled in
(glslang and spirv-cross are build-time tools only). The install test runs
the shared wheel check, checks `minos 14.0` and arm64 on the library, and
reads the hello image on Metal.

Triggers:

- push to `main`
- pull request targeting `main`
- tag `v5*` (for example `v5.0.0a6`)
- Actions → Run workflow

The wheel build passes `-DPPOCR_ENABLE_VULKAN=OFF`. Shaders stay out of the
published library, so the manylinux image does not need glslang and the
wheel does not gain a `libvulkan` dependency. A Vulkan-enabled build is the
source default when `glslangValidator` and `third_party/Vulkan-Headers` are
present; `.github/workflows/vulkan-lavapipe.yml` runs that build on lavapipe.

The wheel and sdist files are uploaded as the `rapidocrlite-dist` artifact (files `rapidocrlite-5.0.0a6-*.whl` and `rapidocrlite-5.0.0a6.tar.gz`),
with `SHA256SUMS`. PyPI publishing runs only for a
push of a `v5*` tag. It does not run for a branch push, a pull request, or a
manual run from a branch.

`5.0.0a6` is a pre-release, so install it with `pip install --pre rapidocrlite`
or `pip install rapidocrlite==5.0.0a6`. `pip install rapidocr` still installs
the official 3.x package, which is a different project.

## Trusted publisher

Daniel configures this once on PyPI and GitHub. The workflow already
requests `id-token: write` and uses the GitHub Environment named `pypi`.

1. On the `RapidOCRLite` project at PyPI (a new name, so use a pending
   publisher), open Publishing → Add a new pending
   publisher (or an ordinary publisher if the project already exists).
2. Owner: `RapidAI`. Repository: `RapidOCRLite`. Workflow name:
   `build-wheels.yml`. Environment name: `pypi`.
3. On GitHub, open Settings → Environments and create an environment named
   `pypi`. A required reviewer is optional. Do not put a PyPI password or
   API token in the repository files. A pending publisher is the only way
   to publish a brand-new name: PyPI cannot issue a project-scoped token for
   a project that does not exist yet. After the first upload, a
   project-scoped token can be stored as repository secret
   `PYPI_API_TOKEN_V5_NG` (used instead of Trusted Publishing when set).
4. The publish job needs `id-token: write` (PyPI Trusted Publishing) and
   `contents: write` (GitHub Release). After PyPI upload it creates a
   GitHub Release for the tag, attaches wheels/sdist/`SHA256SUMS`, and
   marks alpha/beta/rc tags as prerelease.

Native C/C++ SDK archives (`RapidOCRLite-SDK-{win32,win_amd64,macosx_arm64,linux_x86_64}-<version>`)
are built by `.github/workflows/build-sdk.yml` (MSVC Win32/x64, macOS-14, and manylinux2014 x86_64
arm64 with Metal). On a `v5*` tag they are attached to the same GitHub
Release. To attach SDKs to an existing release without retagging, run that
workflow with `workflow_dispatch` and set `release_tag` (e.g. `v5.0.0a6`).

5. The `ubuntu-24.04-arm` label has to be offered to this repository. Public
   repositories normally have it. If GitHub never assigns a runner, the
   QEMU job builds aarch64 instead.

## Trigger a release

From `main`, with the version in `pyproject.toml` already set:

```bash
git tag v5.0.0a6
git push origin v5.0.0a6
```

That push runs the wheel build, publishes `dist/*.whl` and the sdist to
PyPI, then creates a GitHub Release for the tag with the same files and
`SHA256SUMS` attached (prerelease when the tag contains `a`, `b`, or `rc`).
Re-running a tag that is already on PyPI skips the existing file
(`skip-existing`); the Release step updates/creates the GitHub Release.

## PyPI credentials

The `publish` job runs only on a pushed `v5*` tag, in GitHub environment `pypi`.
If repository secret `PYPI_API_TOKEN_V5_NG` is set, it uploads with that token;
otherwise it uses PyPI Trusted Publishing (owner `RapidAI`, repository
`RapidOCRLite`, workflow `build-wheels.yml`, environment `pypi`). The tag must equal
`v` + the package version (e.g. `v5.0.0a6`), or the job stops before uploading.
