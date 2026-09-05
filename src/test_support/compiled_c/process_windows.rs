//! Spawn suspended, attach to a kill-on-close Job Object, then resume.
use super::*;
use std::os::windows::{io::AsRawHandle, process::CommandExt};
use windows_sys::Win32::{
    Foundation::*,
    System::{JobObjects::*, Threading::*},
};

pub struct Tree {
    pub child: Child,
    job: HANDLE,
}
impl Tree {
    pub fn spawn(command: &mut Command) -> io::Result<Self> {
        // SAFETY: null security/name arguments create a new owned, unnamed job.
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the C structure is valid when zero initialized.
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let size = u32::try_from(std::mem::size_of_val(&limits))
            .map_err(|_| io::Error::other("job structure size"))?;
        // SAFETY: job is owned; limits points to a live structure of the supplied size.
        let set = unsafe {
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const limits).cast(),
                size,
            )
        };
        if set == 0 {
            // SAFETY: release the job just created above.
            unsafe {
                CloseHandle(job);
            }
            return Err(io::Error::last_os_error());
        }
        command.creation_flags(CREATE_SUSPENDED);
        let child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                /* SAFETY: this branch still owns the job. */
                unsafe {
                    CloseHandle(job);
                }
                return Err(error);
            }
        };
        let mut tree = Self { child, job };
        // SAFETY: the child handle and owned job are live; the child has not run.
        if unsafe { AssignProcessToJobObject(job, tree.child.as_raw_handle().cast()) } == 0 {
            let error = io::Error::last_os_error();
            let _ = tree.child.kill();
            return Err(error);
        }
        // Rust doesn't expose the primary thread handle. NtResumeProcess resumes
        // the suspended process only after job ownership is established.
        #[link(name = "ntdll")]
        unsafe extern "system" {
            fn NtResumeProcess(process: HANDLE) -> i32;
        }
        // SAFETY: process is owned and was created suspended by this function.
        if unsafe { NtResumeProcess(tree.child.as_raw_handle().cast()) } < 0 {
            return Err(io::Error::other("cannot resume job-owned child"));
        }
        Ok(tree)
    }
    pub fn terminate(&mut self) {
        // SAFETY: job remains owned until Drop; terminating it includes children.
        unsafe {
            TerminateJobObject(self.job, 1);
        }
        let _ = self.child.wait();
    }
}
impl Drop for Tree {
    fn drop(&mut self) {
        self.terminate();
        // SAFETY: release exactly once after terminating/waiting the child.
        unsafe {
            CloseHandle(self.job);
        }
    }
}
