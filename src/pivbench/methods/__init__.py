"""Phương pháp PIV dày đặc. Giao diện chung: ``estimate(img1, img2, **params) -> (u, v)`` cỡ ảnh."""

from __future__ import annotations

from importlib import import_module

METHODS = {
    "ofpiv": "pivbench.methods.ofpiv",
    "crosscorr": "pivbench.methods.crosscorr",
    "raftpiv": "pivbench.methods.raftpiv",
}


def get_method(name: str):
    """Trả về hàm ``estimate`` của phương pháp ``name``."""
    try:
        return import_module(METHODS[name]).estimate
    except KeyError:
        raise ValueError(f"Phương pháp không biết: {name}; có: {sorted(METHODS)}") from None


def estimate(name: str, img1, img2, **params):
    return get_method(name)(img1, img2, **params)


__all__ = ["METHODS", "get_method", "estimate"]
