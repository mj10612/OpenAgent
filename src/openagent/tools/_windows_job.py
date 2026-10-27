"""Windows shell supervisor, launched as a script using only the standard library.

The supervisor owns a non-inheritable kill-on-close Job Object before launching
any command. Its lifetime includes descendants, even after the shell exits.
Terminating the supervisor closes the last job handle and kills its descendants.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time


def main() -> int:
    if sys.platform != 'win32':
        raise OSError('Windows shell supervision is only available on Windows')
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
            ('flags', wintypes.DWORD), ('min_working_set', ctypes.c_size_t),
            ('max_working_set', ctypes.c_size_t), ('process_limit', wintypes.DWORD),
            ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
            ('scheduling', wintypes.DWORD),
        ]

    class IOCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            'read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes',
        )]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ('basic', BasicLimits), ('io', IOCounters),
            ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
            ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t),
        ]

    class Accounting(ctypes.Structure):
        _fields_ = [
            ('user_time', ctypes.c_int64), ('kernel_time', ctypes.c_int64),
            ('period_user_time', ctypes.c_int64), ('period_kernel_time', ctypes.c_int64),
            ('page_faults', wintypes.DWORD), ('total_processes', wintypes.DWORD),
            ('active_processes', wintypes.DWORD), ('terminated_processes', wintypes.DWORD),
        ]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]

    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        error = ctypes.get_last_error()
        kernel.CloseHandle(job)
        raise ctypes.WinError(error)

    # Keep this handle until process exit: closing it explicitly would also
    # terminate this supervisor, which is itself a job member.
    if sys.argv[1] == '--shell-command':
        # Preserve subprocess shell=True quoting instead of re-encoding cmd's
        # command string as an ordinary argv element.
        child = subprocess.Popen(sys.argv[2], shell=True)
    else:
        child = subprocess.Popen(sys.argv[1:])
    code = child.wait()
    while True:
        accounting = Accounting()
        if not kernel.QueryInformationJobObject(job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if accounting.active_processes <= 1:
            return code
        time.sleep(0.05)


if __name__ == '__main__':
    sys.exit(main())
