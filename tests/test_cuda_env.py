"""The cuTile compiler must see CUDA libraries that match it.

When a system CUDA toolkit module is loaded, ``LD_LIBRARY_PATH`` points the
pip-installed tileiras at the toolkit's ``libnvvm`` (13.2 here), which is
older than the wheel's compiler (13.4) and fails on larger tiles with
"failed to compile Tile IR program".  Prepending the wheel's
``nvidia/cu13/lib`` directory fixes the subprocess without touching the
running interpreter.  With the toolkit compiler (no pip wheel) the toolkit's
own libraries are the right ones, so nothing must change.
"""

import os

from cutile.runtime import cuda_env


def _fake_site(tmp_path, monkeypatch, with_lib=True, with_tileiras=True):
    site = tmp_path / "site-packages"
    root = site / "nvidia" / "cu13"
    if with_lib:
        (root / "lib").mkdir(parents=True)
        (root / "lib" / "libnvvm.so.4").write_bytes(b"")
    if with_tileiras:
        (root / "bin").mkdir(parents=True, exist_ok=True)
        (root / "bin" / "tileiras").write_bytes(b"")
    monkeypatch.setattr(cuda_env, "_site_dirs", lambda: [str(site)])
    return root / "lib"


class TestPipCudaLibDir:
    def test_finds_bundled_lib_dir(self, tmp_path, monkeypatch):
        lib = _fake_site(tmp_path, monkeypatch)
        assert cuda_env.pip_cuda_lib_dir() == str(lib)

    def test_returns_none_without_pip_cuda(self, tmp_path, monkeypatch):
        _fake_site(tmp_path, monkeypatch, with_lib=False, with_tileiras=False)
        assert cuda_env.pip_cuda_lib_dir() is None

    def test_finds_pip_tileiras(self, tmp_path, monkeypatch):
        lib = _fake_site(tmp_path, monkeypatch)
        assert cuda_env.pip_tileiras() == str(lib.parent / "bin" / "tileiras")


class TestPreferPipCudaLibs:
    def test_prepends_to_existing_path(self, tmp_path, monkeypatch):
        lib = _fake_site(tmp_path, monkeypatch)
        env = {"LD_LIBRARY_PATH": "/opt/cuda-13.2/lib64:/lib"}
        assert cuda_env.prefer_pip_cuda_libs(env) is True
        assert env["LD_LIBRARY_PATH"] == f"{lib}:/opt/cuda-13.2/lib64:/lib"

    def test_sets_path_when_unset(self, tmp_path, monkeypatch):
        lib = _fake_site(tmp_path, monkeypatch)
        env = {}
        assert cuda_env.prefer_pip_cuda_libs(env) is True
        assert env["LD_LIBRARY_PATH"] == str(lib)

    def test_idempotent(self, tmp_path, monkeypatch):
        lib = _fake_site(tmp_path, monkeypatch)
        env = {"LD_LIBRARY_PATH": f"{lib}:/lib"}
        assert cuda_env.prefer_pip_cuda_libs(env) is False
        assert env["LD_LIBRARY_PATH"] == f"{lib}:/lib"

    def test_noop_without_pip_cuda(self, tmp_path, monkeypatch):
        _fake_site(tmp_path, monkeypatch, with_lib=False, with_tileiras=False)
        env = {"LD_LIBRARY_PATH": "/lib"}
        assert cuda_env.prefer_pip_cuda_libs(env) is False
        assert env == {"LD_LIBRARY_PATH": "/lib"}

    def test_noop_when_only_runtime_wheels_but_toolkit_compiler(self, tmp_path, monkeypatch):
        # e.g. nvidia-nvvm pulled in by another package: the toolkit tileiras
        # must keep using the toolkit's libnvvm.
        _fake_site(tmp_path, monkeypatch, with_lib=True, with_tileiras=False)
        env = {"LD_LIBRARY_PATH": "/opt/cuda/lib64"}
        assert cuda_env.prefer_pip_cuda_libs(env) is False
        assert env == {"LD_LIBRARY_PATH": "/opt/cuda/lib64"}

    def test_applied_to_process_environment_on_import(self):
        lib = cuda_env.pip_cuda_lib_dir()
        if lib is None or cuda_env.pip_tileiras() is None:
            return  # toolkit compiler in use here; nothing to check
        import cutile  # noqa: F401  (import applies the fix)
        assert os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[0] == lib
