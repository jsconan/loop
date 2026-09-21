"""Provide the shared offline product corpus for the single sandbox image."""

from __future__ import annotations

from loop.execution.runtime import load_sandbox_image_definition


def sandbox_image_corpus() -> str:
    """Return representative VCS, shell, build, test, lint, and format operations.

    Returns:
        str: POSIX shell source exercising every promised sandbox tool family.
    """
    uv_version = load_sandbox_image_definition().tools["uv"].version
    return r"""
set -eu
export HOME=/tmp/home XDG_CONFIG_HOME=/tmp/config XDG_CACHE_HOME=/tmp/cache
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"

git status --short >/tmp/git-status
git diff --cached --stat >/tmp/git-cached
git log -1 --format=%H >/tmp/git-history
git worktree list --porcelain >/tmp/git-worktrees
rg -n '^alpha$' tracked.txt >/tmp/rg-result
printf 'alpha\nbeta\n' >/tmp/input.txt
sed -n '2p' /tmp/input.txt | grep '^beta$'
file /tmp/input.txt | grep -i text
tar -czf /tmp/archive.tar.gz -C /tmp input.txt
tar -tzf /tmp/archive.tar.gz | grep '^input.txt$'

python_work=/tmp/loop-python
rm -rf "$python_work" && mkdir "$python_work" && cd "$python_work"
printf 'def add(a, b):\n    return a + b\n' > calc.py
printf 'from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n' > test_calc.py
python3 -m compileall -q .
pytest -q
pip3 list --format=json --disable-pip-version-check >/tmp/pip.json
case "$(uv --version)" in "uv __UV_VERSION__ ("*")") ;; *) exit 1 ;; esac
case "$(uvx --version)" in "uvx __UV_VERSION__ ("*")") ;; *) exit 1 ;; esac
uv venv --offline --python /usr/bin/python3 /tmp/uv-venv
/tmp/uv-venv/bin/python -c 'import sys; assert sys.prefix == "/tmp/uv-venv"'
black calc.py test_calc.py
flake8 calc.py test_calc.py
black --check calc.py test_calc.py

node_work=/tmp/loop-node
rm -rf "$node_work" && mkdir "$node_work" && cd "$node_work"
printf '{"name":"loop-fixture","version":"1.0.0","scripts":{"test":"node test.js"}}\n' > package.json
printf '{"env":{"es6":true,"node":true},"parserOptions":{"ecmaVersion":2018},"rules":{"semi":"error"}}\n' > .eslintrc.json
printf 'function add(a, b) { return a + b; }\nmodule.exports = add;\n' > index.js
printf 'const add = require("./index"); if (add(1, 2) !== 3) process.exit(1);\n' > test.js
npm test --offline
npm pack --offline --ignore-scripts >/tmp/npm-pack.txt
eslint index.js test.js
eslint --fix index.js test.js

cpp_work=/tmp/loop-cpp
rm -rf "$cpp_work" && mkdir "$cpp_work" && cd "$cpp_work"
printf '#include <cassert>\nint add(int a,int b){return a+b;}\nint main(){assert(add(1,2)==3);}\n' > main.cpp
c++ -Wall -Wextra -Werror main.cpp -o app
./app
cmake --version >/tmp/cmake-version
pkg-config --list-all >/tmp/pkg-config
clang-tidy main.cpp -- -std=c++17 >/tmp/clang-tidy
clang-format -i main.cpp
clang-format --dry-run --Werror main.cpp

rust_work=/tmp/loop-rust
rm -rf "$rust_work" && mkdir "$rust_work" && cd "$rust_work"
printf '[package]\nname="loop_fixture"\nversion="0.1.0"\nedition="2021"\n' > Cargo.toml
mkdir src
printf 'pub fn add(a:i32,b:i32)->i32{a+b}\n#[cfg(test)] mod tests{use super::*;#[test]fn adds(){assert_eq!(add(1,2),3);}}\n' > src/lib.rs
cargo build --offline
cargo test --offline
cargo clippy --offline -- -D warnings
cargo fmt
cargo fmt --check

go_work=/tmp/loop-go
rm -rf "$go_work" && mkdir "$go_work" "$go_work/.tmp" && cd "$go_work"
export GOTMPDIR="$go_work/.tmp"
printf 'module example.invalid/loop\n\ngo 1.19\n' > go.mod
printf 'package loop\nfunc Add(a,b int)int{return a+b}\n' > add.go
printf 'package loop\nimport "testing"\nfunc TestAdd(t *testing.T){if Add(1,2)!=3{t.Fatal("bad")}}\n' > add_test.go
GOPROXY=off go build ./...
GOPROXY=off go test ./...
GOPROXY=off go list -m all >/tmp/go-modules
go-staticcheck ./...
gofmt -w .
test -z "$(gofmt -l .)"
""".replace("__UV_VERSION__", uv_version)
