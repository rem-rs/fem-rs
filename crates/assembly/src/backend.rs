//! Backend selection primitives for operator execution.
//!
//! This module introduces a stable backend enum used by higher layers to
//! choose between classic assembled operators and reed-backed operator paths.

/// Assembly/execution backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorBackend {
    /// Classic fem-rs assembly path (assembled sparse matrices).
    Native,
    /// reed/libCEED-style operator path.
    Reed,
}

impl OperatorBackend {
    /// Parse from user-facing backend name.
    ///
    /// Accepted values: `"native"`, `"reed"` (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "native" => Some(Self::Native),
            "reed" => Some(Self::Reed),
            _ => None,
        }
    }

    /// Canonical backend name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Reed => "reed",
        }
    }
}
