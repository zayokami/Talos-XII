use crate::config::Config;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_STACK_SIZE_BYTES: usize = 4 * 1024 * 1024;
const MIN_STACK_SIZE_BYTES: usize = 64 * 1024;
const MAX_STACK_SIZE_BYTES: usize = 512 * 1024 * 1024;

fn stack_size_bytes_from_mb(worker_stack_size_mb: usize) -> usize {
    if worker_stack_size_mb == 0 {
        return DEFAULT_STACK_SIZE_BYTES;
    }

    let bytes = match worker_stack_size_mb
        .checked_mul(1024)
        .and_then(|v| v.checked_mul(1024))
    {
        Some(v) => v,
        None => {
            eprintln!(
                "[Config Warning] worker_stack_size_mb={} overflows, using default {} MB",
                worker_stack_size_mb,
                DEFAULT_STACK_SIZE_BYTES / (1024 * 1024)
            );
            return DEFAULT_STACK_SIZE_BYTES;
        }
    };

    if bytes < MIN_STACK_SIZE_BYTES {
        MIN_STACK_SIZE_BYTES
    } else if bytes > MAX_STACK_SIZE_BYTES {
        eprintln!(
            "[Config Warning] worker_stack_size_mb={} is too large, clamping to {} MB",
            worker_stack_size_mb,
            MAX_STACK_SIZE_BYTES / (1024 * 1024)
        );
        MAX_STACK_SIZE_BYTES
    } else {
        bytes
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Platform: Windows thread priority and CPU affinity
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(windows)]
#[allow(dead_code)]
mod win_platform {
    use std::ffi::c_void;

    type Handle = *mut c_void;
    type DwordPtr = usize;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> Handle;
        fn SetThreadPriority(hThread: Handle, nPriority: i32) -> i32;
        fn SetThreadAffinityMask(hThread: Handle, dwThreadAffinityMask: DwordPtr) -> DwordPtr;
        fn GetCurrentProcess() -> Handle;
        fn GetProcessAffinityMask(
            hProcess: Handle,
            lpProcessAffinityMask: *mut DwordPtr,
            lpSystemAffinityMask: *mut DwordPtr,
        ) -> i32;
    }

    const THREAD_PRIORITY_NORMAL: i32 = 0;
    const THREAD_PRIORITY_ABOVE_NORMAL: i32 = 1;
    const THREAD_PRIORITY_HIGHEST: i32 = 2;
    const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
    const THREAD_PRIORITY_LOWEST: i32 = -2;
    const THREAD_PRIORITY_IDLE: i32 = -15;
    const THREAD_PRIORITY_TIME_CRITICAL: i32 = 15;

    #[allow(dead_code)]
    pub unsafe fn set_current_thread_priority(level: i32) {
        let handle = GetCurrentThread();
        SetThreadPriority(handle, level);
    }

    pub fn priority_from_str(s: &str) -> i32 {
        let level = match s {
            "time_critical" => THREAD_PRIORITY_TIME_CRITICAL,
            "highest" => THREAD_PRIORITY_HIGHEST,
            "above_normal" => THREAD_PRIORITY_ABOVE_NORMAL,
            "normal" => THREAD_PRIORITY_NORMAL,
            "below_normal" => THREAD_PRIORITY_BELOW_NORMAL,
            "lowest" => THREAD_PRIORITY_LOWEST,
            "idle" => THREAD_PRIORITY_IDLE,
            _ => THREAD_PRIORITY_ABOVE_NORMAL,
        };
        if level >= THREAD_PRIORITY_TIME_CRITICAL {
            eprintln!(
                "\x1b[33m[Warning]\x1b[0m time_critical priority can crash the system, clamping to highest"
            );
            THREAD_PRIORITY_HIGHEST
        } else {
            level
        }
    }

    /// Pin the current thread to a specific logical CPU core.
    #[allow(dead_code)]
    pub unsafe fn pin_to_core(core_id: usize) {
        let handle = GetCurrentThread();
        let mask: DwordPtr = 1 << core_id;
        SetThreadAffinityMask(handle, mask);
    }

    /// Detect available cores and return a bitmask of the process affinity.
    pub fn get_process_affinity() -> u64 {
        unsafe {
            let mut proc_mask: DwordPtr = 0;
            let mut sys_mask: DwordPtr = 0;
            GetProcessAffinityMask(GetCurrentProcess(), &mut proc_mask, &mut sys_mask);
            proc_mask as u64
        }
    }

    /// Pre-touch stack pages to avoid page faults during hot execution.
    /// Uses recursion so each page-sized buffer lives in its own stack frame,
    /// guaranteeing we never write outside the current thread's stack.
    #[allow(dead_code)]
    pub fn warmup_stack(stack_bytes: usize) {
        const PAGE_SIZE: usize = 4096;
        let target = stack_bytes.min(512 * 1024);

        #[inline(never)]
        fn probe(remaining: usize) {
            let mut page = [0u8; PAGE_SIZE];
            std::hint::black_box(&mut page);
            if remaining > PAGE_SIZE {
                probe(remaining.saturating_sub(PAGE_SIZE));
            }
        }

        probe(target);
    }
}

#[cfg(not(windows))]
mod win_platform {
    #[allow(dead_code)]
    pub unsafe fn set_current_thread_priority(_level: i32) {}
    pub fn priority_from_str(_s: &str) -> i32 {
        0
    }
    #[allow(dead_code)]
    pub unsafe fn pin_to_core(_core_id: usize) {}
    pub fn get_process_affinity() -> u64 {
        u64::MAX
    }
    #[allow(dead_code)]
    pub fn warmup_stack(_stack_bytes: usize) {}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Worker Pool Metrics
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Default)]
struct WorkerMetrics {
    tasks_completed: AtomicU64,
    tasks_failed: AtomicU64,
    total_exec_ns: AtomicU64,
}

#[allow(dead_code)]
pub struct WorkerStats {
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub avg_exec_us: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
//  GoodJobWorker
// ═══════════════════════════════════════════════════════════════════════════

/// Managed Rayon thread pool with platform-specific optimizations.
pub struct GoodJobWorker {
    pool: ThreadPool,
    num_threads: usize,
    metrics: Arc<WorkerMetrics>,
}

impl GoodJobWorker {
    #[allow(dead_code)]
    pub fn new(requested_threads: usize) -> Result<Self, String> {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let num_threads = if requested_threads == 0 {
            cores.saturating_sub(1).max(1)
        } else {
            requested_threads
        };
        Self::build_pool(num_threads, DEFAULT_STACK_SIZE_BYTES, None, true)
    }

    pub fn new_with_config(config: &Config) -> Result<Self, String> {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let mut num_threads = if cores > config.worker_reserve_cores {
            cores - config.worker_reserve_cores
        } else {
            1
        };
        if config.worker_max_threads > 0 && num_threads > config.worker_max_threads {
            num_threads = config.worker_max_threads;
        }
        let hard_cap = cores.saturating_mul(4).max(64);
        if num_threads > hard_cap {
            num_threads = hard_cap;
        }
        let stack_size = stack_size_bytes_from_mb(config.worker_stack_size_mb);
        let priority = Some(config.worker_priority.clone());
        Self::build_pool(num_threads, stack_size, priority, true)
    }

    fn build_pool(
        num_threads: usize,
        stack_size: usize,
        priority: Option<String>,
        pin_cores: bool,
    ) -> Result<Self, String> {
        let _priority_level =
            win_platform::priority_from_str(priority.as_deref().unwrap_or("normal"));

        // Determine which logical cores are available and build an affinity map.
        // On hybrid architectures (e.g. Alder/Raptor Lake), the OS scheduler
        // typically assigns lower core IDs to P-Cores. By pinning worker threads
        // to consecutive cores starting from 0, we maximize the chance of landing
        // on P-Cores where AVX2/FMA throughput is highest.
        let affinity_mask = win_platform::get_process_affinity();
        let available_cores: Vec<usize> =
            (0..64).filter(|&i| (affinity_mask >> i) & 1 == 1).collect();

        let _warmup_bytes = (stack_size / 2).min(512 * 1024);
        let cores_for_info = available_cores.clone();

        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("gjw-{}", i))
            .stack_size(stack_size)
            .start_handler(move |_idx| {
                // Disabled: start_handler may cause stack corruption or allocation issues
                // under high concurrency on Windows.
            })
            .panic_handler(|err| {
                log::error!("[GJW] Worker thread panicked: {:?}", err);
            })
            .build()
            .map_err(|e| format!("Failed to build worker pool: {}", e))?;

        let pinned_info = if pin_cores && !cores_for_info.is_empty() {
            let used: Vec<usize> = (0..num_threads)
                .map(|i| cores_for_info[i % cores_for_info.len()])
                .collect();
            format!(" pinned={:?}", used)
        } else {
            String::new()
        };

        eprintln!(
            "[GJW-DEBUG] Pool ready: {} threads, {} bytes ({})MB stack, pri={}{}",
            num_threads,
            stack_size,
            stack_size / (1024 * 1024),
            priority.as_deref().unwrap_or("normal"),
            pinned_info
        );

        Ok(Self {
            pool,
            num_threads,
            metrics: Arc::new(WorkerMetrics::default()),
        })
    }

    /// Execute a closure on the worker pool, measuring latency and tracking metrics.
    pub fn execute<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        let start = Instant::now();
        let result = self
            .pool
            .install(|| panic::catch_unwind(AssertUnwindSafe(f)));

        let elapsed_ns = start.elapsed().as_nanos() as u64;

        match result {
            Ok(val) => {
                self.metrics.tasks_completed.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .total_exec_ns
                    .fetch_add(elapsed_ns, Ordering::Relaxed);
                Ok(val)
            }
            Err(err) => {
                self.metrics.tasks_failed.fetch_add(1, Ordering::Relaxed);
                let msg = if let Some(s) = err.downcast_ref::<&str>() {
                    format!("Task panicked: {}", s)
                } else if let Some(s) = err.downcast_ref::<String>() {
                    format!("Task panicked: {}", s)
                } else {
                    "Task panicked with unknown error".to_string()
                };
                Err(msg)
            }
        }
    }

    /// Execute multiple independent closures in parallel and collect results.
    #[allow(dead_code)]
    pub fn execute_batch<F, R>(&self, tasks: Vec<F>) -> Vec<Result<R, String>>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        use rayon::prelude::*;
        let metrics = Arc::clone(&self.metrics);

        self.pool.install(|| {
            tasks
                .into_par_iter()
                .map(|f| {
                    let start = Instant::now();
                    let result = panic::catch_unwind(AssertUnwindSafe(f));
                    let elapsed_ns = start.elapsed().as_nanos() as u64;

                    match result {
                        Ok(val) => {
                            metrics.tasks_completed.fetch_add(1, Ordering::Relaxed);
                            metrics
                                .total_exec_ns
                                .fetch_add(elapsed_ns, Ordering::Relaxed);
                            Ok(val)
                        }
                        Err(err) => {
                            metrics.tasks_failed.fetch_add(1, Ordering::Relaxed);
                            let msg = if let Some(s) = err.downcast_ref::<&str>() {
                                format!("Task panicked: {}", s)
                            } else if let Some(s) = err.downcast_ref::<String>() {
                                format!("Task panicked: {}", s)
                            } else {
                                "Task panicked with unknown error".to_string()
                            };
                            Err(msg)
                        }
                    }
                })
                .collect()
        })
    }

    pub fn thread_count(&self) -> usize {
        self.num_threads
    }

    /// Snapshot of pool utilization metrics.
    #[allow(dead_code)]
    pub fn stats(&self) -> WorkerStats {
        let completed = self.metrics.tasks_completed.load(Ordering::Relaxed);
        let failed = self.metrics.tasks_failed.load(Ordering::Relaxed);
        let total_ns = self.metrics.total_exec_ns.load(Ordering::Relaxed);
        let avg_us = if completed > 0 {
            (total_ns as f64 / completed as f64) / 1000.0
        } else {
            0.0
        };
        WorkerStats {
            tasks_completed: completed,
            tasks_failed: failed,
            avg_exec_us: avg_us,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_size_mb_zero_uses_default() {
        assert_eq!(stack_size_bytes_from_mb(0), DEFAULT_STACK_SIZE_BYTES);
    }

    #[test]
    fn stack_size_overflow_falls_back_to_default() {
        assert_eq!(
            stack_size_bytes_from_mb(usize::MAX),
            DEFAULT_STACK_SIZE_BYTES
        );
    }

    #[test]
    fn stack_size_is_clamped_to_max() {
        let max_mb = (MAX_STACK_SIZE_BYTES / (1024 * 1024)).saturating_add(1);
        assert_eq!(stack_size_bytes_from_mb(max_mb), MAX_STACK_SIZE_BYTES);
    }
}
