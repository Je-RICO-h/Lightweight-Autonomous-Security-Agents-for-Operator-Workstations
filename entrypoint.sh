#!/usr/bin/env bash
# Runs run.py using this project's venv, elevated to root.
# Root is required under Wayland so pynput's uinput backend can open
# /dev/uinput and /dev/input/event* directly (see run.py's Wayland note).
#
# sudo's env_reset (this system's default) drops -E-preserved vars like
# XDG_SESSION_TYPE, so run.py's own Wayland auto-detect never sees them
# under sudo. Passed explicitly here instead, via `sudo env`, which survives
# env_reset because it's not inherited -- it's set directly in the command.
#
# PYNPUT_BACKEND_KEYBOARD (not the generic PYNPUT_BACKEND): pynput has no
# mouse._uinput backend, and pynput/__init__.py imports pynput.mouse
# unconditionally, so the generic var would crash that import before
# keyboard is ever used.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

exec sudo env PYNPUT_BACKEND_KEYBOARD=uinput "$PWD/.venv/bin/python" run.py "$@"
