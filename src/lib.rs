//! # oxideav-mp3
//!
//! **Status:** orphan-rebuild scaffold (reset 2026-05-24).
//!
//! The prior implementation was retired under the workspace clean-room
//! policy: several of its data tables and decode-loop structures were
//! documented as having been consulted from external reference
//! implementations (their source, not the ISO/IEC specification),
//! which violates the clean-room provenance requirement regardless of
//! those references' licensing. The crate will be re-implemented from
//! scratch against the staged ISO/IEC 11172-3 / 13818-3 Layer III
//! specification (numeric tables read only from the standard) in a
//! future clean-room round.
//!
//! Every public API currently returns [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

/// Crate-local error type. Until the clean-room rebuild lands every
/// public API path returns [`Error::NotImplemented`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The crate has been reset to a scaffold pending clean-room
    /// rebuild; no decoder or encoder functionality is wired up yet.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "oxideav-mp3: orphan-rebuild scaffold — no codec wired up"
        )
    }
}

impl std::error::Error for Error {}

/// No-op codec registration — the orphan-rebuild scaffold registers
/// nothing into the runtime context.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("mp3", register);
