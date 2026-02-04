use crate::config::Config;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

#[cfg(windows)]
mod win_priority {
    use std::ffi::c_void;

    type HANDLE = *mut c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> HANDLE;
        fn SetThreadPriority(hThread: HANDLE, nPriority: i32) -> i32;
    }

    const THREAD_PRIORITY_NORMAL: i32 = 0;
    const THREAD_PRIORITY_ABOVE_NORMAL: i32 = 1;
    const THREAD_PRIORITY_HIGHEST: i32 = 2;

    pub unsafe fn set_current_thread_priority(level: i32) {
        let handle = GetCurrentThread();
        SetThreadPriority(handle, level);
    }

    pub fn priority_from_str(s: &str) -> i32 {
        match s {
            "highest" => THREAD_PRIORITY_HIGHEST,
            "normal" => THREAD_PRIORITY_NORMAL,
            _ => THREAD_PRIORITY_ABOVE_NORMAL,
        }
    }
}

pub struct GoodJobWorker {
    pool: Arc<ThreadPool>,
    num_threads: usize,
}

impl GoodJobWorker {
    #[allow(dead_code)]
    pub fn new(requested_threads: usize) -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let num_threads = if requested_threads == 0 {
            if cores > 2 {
                cores - 1
            } else {
                cores
            }
        } else {
            requested_threads
        };
        Self::build_pool(num_threads, 4 * 1024 * 1024, None)
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
            4 * 1024 * 1024
        } else {
            config.worker_stack_size_mb * 1024 * 1024
        };
        let priority = Some(config.worker_priority.clone());
        Self::build_pool(num_threads, stack_size, priority)
    }

    fn build_pool(num_threads: usize, stack_size: usize, priority: Option<String>) -> Self {
        #[cfg(windows)]
        let priority_level =
            win_priority::priority_from_str(priority.as_deref().unwrap_or("normal"));
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("worker-thread-{}", i))
            .stack_size(stack_size)
            .start_handler(move |_| {
                #[cfg(windows)]
                {
                    unsafe {
                        win_priority::set_current_thread_priority(priority_level);
                    }
                }
            })
            .panic_handler(|err| {
                eprintln!("Worker thread panicked: {:?}", err);
            })
            .build()
            .expect("Failed to build worker pool");

        println!("Worker initialized with {} threads.", num_threads);

        Self {
            pool: Arc::new(pool),
            num_threads,
        }
    }

    pub fn execute<F, R>(&self, f: F) -> Result<R, String>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        let result = self
            .pool
            .install(|| panic::catch_unwind(AssertUnwindSafe(f)));

        match result {
            Ok(val) => Ok(val),
            Err(err) => {
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

    #[allow(dead_code)]
    pub fn thread_count(&self) -> usize {
        self.num_threads
    }
}
