/// Conditionally print a formatted message to stdout.
///
/// Evaluates `$cond` at runtime; if `true`, calls [`println!`] with the
/// remaining arguments unchanged.  When `$cond` is a compile-time constant
/// `false`, the body is still compiled but the optimizer will eliminate it.
///
/// # Example
///
/// ```rust
/// const VERBOSE: bool = true;
/// print_if!(VERBOSE, "step {} loss = {:.4}", epoch, loss);
/// ```
#[macro_export]
macro_rules! print_if {
    ($cond:expr, $($arg:tt)*) => {
        if $cond {
            println!($($arg)*);
        }
    };
}
