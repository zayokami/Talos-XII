//! Panic recovery and resilience utilities
//!
//! Provides a custom panic hook that routes panic messages through the `log`
//! crate (so they respect `env_logger` filters and formatting) and helpers for
//! recovering from poisoned `std::sync::RwLock`s without crashing the main loop.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Install a custom panic hook that logs the panic info and backtrace via
/// `log::error!` before invoking the default hook.
///
/// Call this once at application startup, before spawning any threads.
pub fn install() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic payload");
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown location".to_string());
        log::error!(
            target: "panic",
            "Thread panicked at {}: {}",
            location,
            payload
        );
        default_hook(info);
    }));
}

/// Read from a shared `RwLock`, recovering gracefully if the lock is poisoned.
///
/// Background training threads may panic while holding a write lock. Rather
/// than crashing the application on the next read, we log a warning and
/// return the guard anyway. Neural-network weights are always in a valid
/// state (they are just `Vec<f64>`), so this recovery is safe.
pub fn read_shared<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    match lock.read() {
        Ok(guard) => guard,
        Err(poison) => {
            log::warn!(
                target: "resilience",
                "Shared model lock was poisoned (background thread panicked). Recovering data."
            );
            poison.into_inner()
        }
    }
}

/// Write to a shared `RwLock`, recovering gracefully if the lock is poisoned.
///
/// Same rationale as `read_shared`: the data inside is always valid, so we
/// can safely proceed after logging the incident.
#[allow(dead_code)]
pub fn write_shared<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    match lock.write() {
        Ok(guard) => guard,
        Err(poison) => {
            log::warn!(
                target: "resilience",
                "Shared model lock was poisoned during write. Recovering data."
            );
            poison.into_inner()
        }
    }
}
