#!/usr/bin/env bash
# Keep Isaac's Python runtime ahead of system/Conda libraries in container shells.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_dir="${EXCAVATOR_ISAAC_DIR:-$repo_root/isaacsim}"

if [[ ! -x "$isaac_dir/python.sh" || ! -d "$isaac_dir/kit/python/lib" ]]; then
    echo "Isaac Sim installation not found at $isaac_dir. Run preperations.sh or set EXCAVATOR_ISAAC_DIR." >&2
    exit 1
fi

# Isaac's setup_python_env.sh appends its paths. A caller-provided system library
# directory can otherwise select a different libpython with the same SONAME,
# causing bundled native modules (e.g. _asyncio) to segfault before Kit starts.
export LD_LIBRARY_PATH="$isaac_dir/kit/python/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$isaac_dir/python.sh" "$@"
