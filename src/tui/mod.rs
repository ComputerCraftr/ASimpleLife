mod analysis;
mod headless;
mod protocol;
mod source;
mod termination;
mod ui;
mod worker;

pub use headless::run_headless_source;
pub use protocol::{ControlCommand, MAX_CONTINUOUS_QUANTUM, WorkerEvent};
pub use source::{prepare_bf_source, prepare_config_source, prepare_file_source};
pub use termination::TuiExit;
pub use ui::{run, should_run_tui};
