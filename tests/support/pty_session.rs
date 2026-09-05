use std::io;
use std::os::fd::BorrowedFd;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command};

/// The command's stdin must already refer to its PTY slave. Merely redirecting
/// stdio leaves `/dev/tty` attached to the caller's terminal (for example iTerm2).
pub(super) fn spawn(mut command: Command) -> io::Result<Child> {
    // SAFETY: The hook uses only async-signal-safe session/ioctl syscalls and
    // allocation-free OS error conversion. No parent locks or Rust I/O are used
    // between fork and exec. Command has installed the slave on stdin by then.
    unsafe { command.pre_exec(establish_session) };
    command.spawn()
}

fn establish_session() -> io::Result<()> {
    nix::unistd::setsid().map_err(io::Error::from)?;
    // Darwin declares this request as u32 but ioctl's request argument is ulong.
    // Other Unix libc targets already give the constant ioctl's native type.
    #[cfg(target_vendor = "apple")]
    let request = nix::libc::c_ulong::from(nix::libc::TIOCSCTTY);
    #[cfg(not(target_vendor = "apple"))]
    let request = nix::libc::TIOCSCTTY;
    // SAFETY: stdin is an open PTY slave owned by the child. TIOCSCTTY takes an
    // integer flag, not a pointer; zero avoids stealing another session's tty.
    if unsafe { nix::libc::ioctl(nix::libc::STDIN_FILENO, request, 0) } == -1 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: Command installed the still-open slave on fd 0; this borrow does
    // not close it and cannot outlive this pre-exec hook.
    let stdin = unsafe { BorrowedFd::borrow_raw(nix::libc::STDIN_FILENO) };
    // Check inside the session: macOS rejects tcgetpgrp from the parent, whose
    // controlling terminal intentionally differs from this slave.
    if nix::unistd::tcgetpgrp(stdin).map_err(io::Error::from)? != nix::unistd::getpid() {
        return Err(io::Error::from_raw_os_error(nix::libc::ENOTTY));
    }
    Ok(())
}
