#[cfg(unix)]
pub fn terminal_view_size(default_width: usize, default_height: usize) -> (usize, usize) {
    use std::mem::MaybeUninit;
    use std::os::fd::AsRawFd;

    let stdout = std::io::stdout();
    let fd = stdout.as_raw_fd();
    let mut winsize = MaybeUninit::<libc::winsize>::zeroed();
    let rc = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, winsize.as_mut_ptr()) };

    if rc == 0 {
        let winsize = unsafe { winsize.assume_init() };
        let cols = usize::from(winsize.ws_col);
        let rows = usize::from(winsize.ws_row);
        if cols > 0 && rows > 1 {
            return (cols, rows - 1);
        }
    }

    (default_width, default_height)
}

#[cfg(not(unix))]
pub fn terminal_view_size(default_width: usize, default_height: usize) -> (usize, usize) {
    (default_width, default_height)
}
