# Releasing RapidOCRLite

The PyPI distribution name is `RapidOCRLite`. The import name is still `rapidocr`.

Everything is released by one workflow: `.github/workflows/release.yml`. It
builds the C/C++ SDK archives, the Python wheels + sdist and the Java fat jar,
then publishes them: SDK archives and the jar go to the GitHub Release, the
wheels and sdist go to PyPI **and** to the same Release. The former
`build-sdk.yml` / `build-wheels.yml` / `build-java.yml` workflows were merged
into it so the ordering is deterministic (the Java jobs consume the SDK
archives of the very same run instead of racing a release upload).

**Release target.** A push to `main` and a push of a `v5*` tag both publish.
On a tag push the release is that tag; on a `main` push the release is
`v<version from pyproject.toml>`, and the workflow creates that tag (on the
pushed commit) and the release if they do not exist yet. An existing tag is
never moved or deleted — deleting a tag empties the release assets.

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
   `release.yml`. Environment name: `pypi`.
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

Native C/C++ SDK archives (`RapidOCRLite-SDK-{win32,win_amd64,macosx_arm64,linux_x86_64,linux_aarch64}-<version>`)
are built by `.github/workflows/release.yml` (MSVC Win32/x64, macOS-14 arm64,
macOS-15-intel x86_64, and manylinux2014 x86_64/aarch64). They are attached
to the GitHub Release of the resolved tag, on a `v5*` tag push and on every
push to `main`. The Java fat jar (`rapidocrlite-java-<version>.jar`, built
from the same run's SDK archives for win-x86_64 / linux-x86_64 /
linux-aarch64 / macos-aarch64 / macos-x86_64) is attached to the same
release. LoongArch SDKs are attached by `loongarch64-qemu.yml`, which follows
the same tag rule.

5. The `ubuntu-24.04-arm` label has to be offered to this repository. Public
   repositories normally have it. If GitHub never assigns a runner, the
   QEMU job builds aarch64 instead.

## Trigger a release

Set the version in `pyproject.toml`, then push it to `main`:

```bash
git push origin main
```

or create the tag explicitly (same release; useful when `main` already carries
the version and only the assets have to be refreshed):

```bash
git tag v5.0.0a6
git push origin v5.0.0a6
```

A `main` push builds everything and publishes to the release of `v<version>`;
the workflow creates that tag and the release when they are missing, so no
manual tagging is required any more. It publishes `dist/*.whl` and the sdist to
PyPI, then attaches wheels, sdist, SDK archives, the Java jar and `SHA256SUMS`
to the GitHub Release (prerelease when the tag contains `a`, `b`, or `rc`).
Files already on PyPI are skipped (`skip-existing`), so repeated `main` pushes
are safe.

## PyPI credentials

The `publish` job runs on a push to `main` and on a pushed `v5*` tag, in GitHub
environment `pypi`. If repository secret `PYPI_API_TOKEN_V5_NG` is set, it uploads
with that token; otherwise it uses PyPI Trusted Publishing (owner `RapidAI`,
repository `RapidOCRLite`, workflow `release.yml`, environment `pypi`). The
resolved tag must equal `v` + the package version (e.g. `v5.0.0a6`), or the job
stops before uploading.
