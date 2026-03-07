use crate::config::Config;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_STACK_SIZE_BYTES: usize = 4 * 1024 * 1024;

// ═══════════════════════════════════════════════════════════════════════════
//  Platform: Windows thread priority and CPU affinity
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(windows)]
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

    pub unsafe fn set_current_thread_priority(level: i32) {
        let handle = GetCurrentThread();
        SetThreadPriority(handle, level);
    }

    pub fn priority_from_str(s: &str) -> i32 {
        match s {
            "time_critical" => THREAD_PRIORITY_TIME_CRITICAL,
            "highest" => THREAD_PRIORITY_HIGHEST,
            "above_normal" => THREAD_PRIORITY_ABOVE_NORMAL,
            "normal" => THREAD_PRIORITY_NORMAL,
            "below_normal" => THREAD_PRIORITY_BELOW_NORMAL,
            "lowest" => THREAD_PRIORITY_LOWEST,
            "idle" => THREAD_PRIORITY_IDLE,
            _ => THREAD_PRIORITY_HIGHEST,
        }
    }

    /// Pin the current thread to a specific logical CPU core.
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
    /// Walks `stack_bytes` worth of stack in 4KB increments.
    pub fn warmup_stack(stack_bytes: usize) {
        let pages = stack_bytes / 4096;
        let mut dummy: u8 = 0;
        for i in 0..pages {
            // Volatile write to force stack page allocation without optimization
            unsafe {
                let stack_probe: *mut u8 = (&mut dummy as *mut u8).sub(i * 4096);
                std::ptr::write_volatile(stack_probe, 0);
            }
        }
        std::hint::black_box(dummy);
    }
}

#[cfg(not(windows))]
mod win_platform {
    pub unsafe fn set_current_thread_priority(_level: i32) {}
    pub fn priority_from_str(_s: &str) -> i32 {
        0
    }
    pub unsafe fn pin_to_core(_core_id: usize) {}
    pub fn get_process_affinity() -> u64 {
        u64::MAX
    }
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

pub struct GoodJobWorker {
    pool: ThreadPool,
    num_threads: usize,
    metrics: Arc<WorkerMetrics>,
}

impl GoodJobWorker {
    #[allow(dead_code)]
    pub fn new(requested_threads: usize) -> Self {
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

    pub fn new_with_config(config: &Config) -> Self {
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
        let stack_size = if config.worker_stack_size_mb == 0 {
            DEFAULT_STACK_SIZE_BYTES
        } else {
            config.worker_stack_size_mb * 1024 * 1024
        };
        let priority = Some(config.worker_priority.clone());
        Self::build_pool(num_threads, stack_size, priority, true)
    }

    fn build_pool(
        num_threads: usize,
        stack_size: usize,
        priority: Option<String>,
        pin_cores: bool,
    ) -> Self {
        let priority_level =
            win_platform::priority_from_str(priority.as_deref().unwrap_or("normal"));

        // Determine which logical cores are available and build an affinity map.
        // On hybrid architectures (e.g. Alder/Raptor Lake), the OS scheduler
        // typically assigns lower core IDs to P-Cores. By pinning worker threads
        // to consecutive cores starting from 0, we maximize the chance of landing
        // on P-Cores where AVX2/FMA throughput is highest.
        let affinity_mask = win_platform::get_process_affinity();
        let available_cores: Vec<usize> =
            (0..64).filter(|&i| (affinity_mask >> i) & 1 == 1).collect();

        let warmup_bytes = (stack_size / 2).min(512 * 1024);
        let cores_for_info = available_cores.clone();

        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("gjw-{}", i))
            .stack_size(stack_size)
            .start_handler(move |idx| {
                unsafe {
                    win_platform::set_current_thread_priority(priority_level);
                }

                if pin_cores && !available_cores.is_empty() {
                    let core_id = available_cores[idx % available_cores.len()];
                    unsafe {
                        win_platform::pin_to_core(core_id);
                    }
                }

                win_platform::warmup_stack(warmup_bytes);
            })
            .panic_handler(|err| {
                eprintln!("[GJW] Worker thread panicked: {:?}", err);
            })
            .build()
            .expect("Failed to build worker pool");

        let pinned_info = if pin_cores && !cores_for_info.is_empty() {
            let used: Vec<usize> = (0..num_threads)
                .map(|i| cores_for_info[i % cores_for_info.len()])
                .collect();
            format!(" pinned={:?}", used)
        } else {
            String::new()
        };

        println!(
            "[GJW] Pool ready: {} threads, {}MB stack, pri={}{}",
            num_threads,
            stack_size / (1024 * 1024),
            priority.as_deref().unwrap_or("normal"),
            pinned_info
        );

        Self {
            pool,
            num_threads,
            metrics: Arc::new(WorkerMetrics::default()),
        }
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
