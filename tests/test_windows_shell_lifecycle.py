import asyncio
import ctypes
import subprocess
import sys
import time

import pytest

from openagent.tools.shell import ShellTool

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows process ownership')


@pytest.mark.asyncio
async def test_descendant_output_survives_shell_exit(tmp_path):
    workspace = tmp_path / 'workspace with spaces'
    workspace.mkdir()
    script = workspace / 'parent.py'
    script.write_text(
        'import subprocess, sys\n'
        'subprocess.Popen([sys.executable, "-c", '
        '"import time; time.sleep(0.2); print(12345)"])\n', encoding='utf-8',
    )
    command = subprocess.list2cmdline([sys.executable, str(script)])
    result = await ShellTool(workspace).execute(command=command, timeout=5)
    assert not result.is_error
    assert result.output == '12345'


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True], ids=['timeout', 'cancellation'])
async def test_exited_shell_descendants_are_terminated(tmp_path, cancel):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    script = tmp_path / 'parent.py'
    script.write_text(
        'import subprocess, sys\nfrom pathlib import Path\n'
        'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
        'Path(sys.argv[1]).write_text(str(child.pid))\n', encoding='utf-8',
    )
    pidfile = tmp_path / 'child.pid'
    command = subprocess.list2cmdline([sys.executable, str(script), str(pidfile)])
    task = asyncio.create_task(ShellTool(tmp_path, shell_type='cmd').execute(command=command, timeout=2))
    handle = None
    try:
        async with asyncio.timeout(5):
            while not pidfile.exists() or not pidfile.read_text():
                await asyncio.sleep(0.02)
        handle = kernel.OpenProcess(0x1001, False, int(pidfile.read_text()))
        assert handle
        # Allow the launching shell to exit while its descendant retains pipes.
        await asyncio.sleep(0.2)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.is_error and 'timed out' in result.output
        deadline = time.monotonic() + 2
        status = ctypes.c_ulong()
        while True:
            assert kernel.GetExitCodeProcess(handle, ctypes.byref(status))
            if status.value != 259 or time.monotonic() > deadline:
                break
            await asyncio.sleep(0.02)
        assert status.value != 259, 'descendant survived tool cleanup'
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if handle:
            kernel.TerminateProcess(handle, 1)
            kernel.CloseHandle(handle)
