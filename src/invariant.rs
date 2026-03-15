use std::fmt::Debug;

pub trait RequiredExt<T> {
    fn or_invariant(self, context: &str) -> T;
}

impl<T> RequiredExt<T> for Option<T> {
    #[track_caller]
    fn or_invariant(self, context: &str) -> T {
        match self {
            Some(value) => value,
            None => invariant_failure(context, "value was absent"),
        }
    }
}

impl<T, E: Debug> RequiredExt<T> for Result<T, E> {
    #[track_caller]
    fn or_invariant(self, context: &str) -> T {
        match self {
            Ok(value) => value,
            Err(error) => invariant_failure(context, &format!("{error:?}")),
        }
    }
}

pub trait RequiredErrorExt<E> {
    fn error_or_invariant(self, context: &str) -> E;
}

impl<T: Debug, E> RequiredErrorExt<E> for Result<T, E> {
    #[track_caller]
    fn error_or_invariant(self, context: &str) -> E {
        match self {
            Err(error) => error,
            Ok(value) => invariant_failure(context, &format!("unexpected success: {value:?}")),
        }
    }
}

#[cold]
#[inline(never)]
#[track_caller]
pub fn invariant_failure(context: &str, detail: &str) -> ! {
    #[cfg(test)]
    assert!(
        std::thread::panicking(),
        "invariant failed: {context}; {detail}"
    );
    eprintln!("invariant failed: {context}; {detail}");
    std::process::abort()
}

#[macro_export]
macro_rules! invariant_failure {
    () => {{
        $crate::invariant_failure("explicit invariant failure", "unreachable state")
    }};
    ($($argument:tt)*) => {{
        $crate::invariant_failure("explicit invariant failure", &format!($($argument)*))
    }};
}
