use super::*;
use std::collections::BTreeMap;

#[derive(Default, Serialize)]
pub struct Request {
    pub(super) outcome: Option<&'static str>,
    pub(super) compiler_invocations: u64,
    compiler_outcome: Option<&'static str>,
    pub(super) execution_attempts: u64,
    milliseconds: BTreeMap<&'static str, u128>,
    notes: Vec<String>,
}
impl Request {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn time(&mut self, phase: &'static str, elapsed: Duration) {
        self.milliseconds.insert(phase, elapsed.as_millis());
    }
    pub fn outcome(&mut self, outcome: &'static str) {
        self.outcome = Some(outcome);
    }
    pub fn compiler_started(&mut self) {
        self.compiler_invocations += 1;
    }
    pub fn compiler_finished(&mut self, outcome: &'static str) {
        self.compiler_outcome = Some(outcome);
    }
    pub fn execution_started(&mut self) {
        self.execution_attempts += 1;
    }
    pub fn failed(&mut self) {
        if self.outcome.is_none() {
            self.outcome = Some("request_failure");
        }
    }
    pub fn note(&mut self, message: &str) {
        self.notes.push(message.to_string());
    }
}
impl Drop for Request {
    fn drop(&mut self) {
        if std::env::var_os("ASIMPLELIFE_C_TEST_STATS").is_some()
            && let Ok(json) = serde_json::to_string(self)
        {
            eprintln!("C_TEST_WORK {json}");
        }
    }
}
