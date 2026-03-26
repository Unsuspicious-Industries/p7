//! Python bindings for the Aufbau constrained completion engine

use aufbau::logic::grammar::Grammar as RustGrammar;
use aufbau::logic::partial::{CompletionSet, Parser, Synthesizer as RustSynthesizer};
use aufbau::logic::typing::core::{Context, TreeStatus};
use aufbau::logic::typing::eval::check_tree_with_context;
use aufbau::regex::Regex as DerivativeRegex;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct Grammar {
    inner: RustGrammar,
}

#[pymethods]
impl Grammar {
    #[new]
    fn new(spec: &str) -> PyResult<Self> {
        let inner = RustGrammar::load(spec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Grammar parse error: {}", e))
        })?;
        Ok(Self { inner })
    }

    fn start_nonterminal(&self) -> Option<String> {
        self.inner.start_nonterminal().cloned()
    }
}

#[pyclass]
pub struct Synthesizer {
    inner: RustSynthesizer,
}

#[pymethods]
impl Synthesizer {
    #[new]
    fn new(grammar: &Grammar, input: &str) -> Self {
        Self {
            inner: RustSynthesizer::new(grammar.inner.clone(), input),
        }
    }

    fn current_text(&self) -> String {
        self.inner.input().to_string()
    }

    fn set_input(&mut self, input: &str) {
        self.inner.set_input(input);
    }

    fn get_completions(&mut self) -> Vec<String> {
        let completions = self.inner.completions();
        completions
            .iter()
            .map(|r| match r.example() {
                Some(e) => e,
                None => r.to_pattern(),
            })
            .collect()
    }

    fn extend(&mut self, token: &str) -> PyResult<bool> {
        let ctx = Context::new();
        match self.inner.extend(token, &ctx) {
            Ok(_) => Ok(true),
            Err(e) => Err(PyRuntimeError::new_err(e)),
        }
    }

    fn is_complete(&mut self) -> bool {
        self.inner.complete().is_some()
    }
}

#[pyfunction]
fn regex_matches(pattern: &str, text: &str) -> PyResult<bool> {
    let regex = DerivativeRegex::from_str(pattern).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid regex: {}", e))
    })?;
    Ok(regex.matches(text))
}

#[pyfunction]
fn regex_prefix_valid(pattern: &str, prefix: &str) -> PyResult<bool> {
    let regex = DerivativeRegex::from_str(pattern).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid regex: {}", e))
    })?;
    match regex.prefix_match(prefix) {
        aufbau::regex::PrefixStatus::NoMatch => Ok(false),
        _ => Ok(true),
    }
}

#[pymodule]
fn p7(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Grammar>()?;
    m.add_class::<Synthesizer>()?;
    m.add_function(wrap_pyfunction!(regex_matches, m)?)?;
    m.add_function(wrap_pyfunction!(regex_prefix_valid, m)?)?;
    Ok(())
}
