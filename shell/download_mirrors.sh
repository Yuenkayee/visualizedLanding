#!/usr/bin/env bash
# Sourced by the closed-loop launcher; changes only its process environment.
# A PyPI index cannot serve uv's managed CPython release archives.
case "${LANDING_DOWNLOAD_PROFILE:-default}" in
  default) ;;
  china)
    # Explicit profile selection supersedes PIP_INDEX_URL/UV_DEFAULT_INDEX from
    # an earlier terminal session. LANDING_* variables provide custom overrides.
    export PIP_INDEX_URL="${LANDING_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export UV_DEFAULT_INDEX="$PIP_INDEX_URL"
    export UV_PYTHON_INSTALL_MIRROR="${LANDING_PYTHON_BINARY_MIRROR:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone}"
    export LANDING_PYTHON_SOURCE_MIRROR="${LANDING_PYTHON_SOURCE_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/python}"
    export LANDING_APT_MIRROR="${LANDING_APT_MIRROR:-https://mirrors.tuna.tsinghua.edu.cn/ubuntu}"
    ;;
  *) printf 'Unknown LANDING_DOWNLOAD_PROFILE: %s (choose default or china)\n' "$LANDING_DOWNLOAD_PROFILE" >&2; return 2 ;;
esac
