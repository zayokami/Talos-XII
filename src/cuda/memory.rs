//! CUDA memory management
//!
//! Provides GPU memory allocation, deallocation, and CPU<->GPU data transfer.
#![allow(dead_code)]

#[cfg(cuda)]
use crate::cuda::bindings::{
    cuStreamSynchronize, cudaErrorNotReady, cudaEventCreateWithFlags, cudaEventDestroy,
    cudaEventDisableTiming, cudaEventQuery, cudaEventRecord, cudaEventSynchronize, cudaEvent_t,
    cudaFree, cudaFreeHost, cudaMalloc, cudaMallocHost, cudaMemcpy, cudaMemcpyAsync,
    cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost, cudaMemcpyHostToDevice, CUstream,
    CUDA_SUCCESS,
};
use crate::cuda::error::{CudaError, CudaResult};
#[cfg(cuda)]
use std::ffi::c_void;
#[cfg(cuda)]
use std::os::raw::c_int;
#[cfg(cuda)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Size-keyed GPU memory pool for reusing temporary allocations.
/// Keys are buffer size in bytes; values are lists of raw device pointers.
///
/// Correctness invariants:
/// - Buffers handed out by `alloc_pooled` contain stale data from their
///   previous use. Callers must fully overwrite them (GEMM with beta=0,
///   full-range kernels, `copy_h2d`/`copy_d2d`) or `fill` them first.
/// - Recycling is safe without an explicit device sync only because every
///   device-side operation in this crate runs either on the legacy default
///   stream (custom kernel launches) or on the process-wide *blocking*
///   transfer stream (async copies and cuBLAS, see
///   `stream::global_transfer_stream`). The legacy default stream implicitly
///   synchronizes with all blocking streams in both directions, so all device
///   work remains mutually ordered: writes into a recycled buffer are
///   stream-ordered after any earlier reads of it. This is equivalent to the
///   pre-async invariant where everything ran on the legacy default stream.
///   Before introducing any NON-blocking stream
///   (`cudaStreamNonBlocking`/`CU_STREAM_NON_BLOCKING`), per-buffer event
///   tracking must be implemented first, otherwise a pooled buffer released
///   on stream A could be rewritten on stream B while A is still reading it.
#[cfg(cuda)]
static GPU_BUFFER_POOL: std::sync::LazyLock<Mutex<std::collections::HashMap<usize, Vec<usize>>>> =
    std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

const MAX_POOL_ENTRIES_PER_SIZE: usize = 8;

/// Opaque CUDA memory pointer wrapper
#[derive(Clone)]
pub struct DevicePtr<T> {
    ptr: usize,
    size: usize,
    pooled: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> DevicePtr<T> {
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn as_raw(&self) -> usize {
        self.ptr
    }

    /// Create a zero-sized device pointer (no GPU allocation)
    pub fn zero_sized() -> Self {
        DevicePtr {
            ptr: 0,
            size: 0,
            pooled: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Drop for DevicePtr<T> {
    fn drop(&mut self) {
        #[cfg(cuda)]
        if self.ptr != 0 {
            if self.pooled {
                let size_bytes = self.size * std::mem::size_of::<T>();
                if let Ok(mut pool) = GPU_BUFFER_POOL.lock() {
                    let entry = pool.entry(size_bytes).or_default();
                    if entry.len() < MAX_POOL_ENTRIES_PER_SIZE {
                        entry.push(self.ptr);
                        self.ptr = 0;
                        return;
                    }
                }
            }
            unsafe {
                let result = cudaFree(self.ptr as *mut c_void);
                if result != CUDA_SUCCESS as c_int {
                    eprintln!("[CUDA] cudaFree failed during drop: {}", result);
                }
            }
        }
    }
}

/// Type-erased GPU buffer for mixed-dtype cache storage.
#[derive(Clone)]
pub enum CudaBuffer {
    BF16(DevicePtr<crate::dtype::bf16>),
    I8(DevicePtr<i8>),
    F32(DevicePtr<f32>),
    F64(DevicePtr<f64>),
}

impl CudaBuffer {
    pub fn len(&self) -> usize {
        match self {
            CudaBuffer::BF16(b) => b.len(),
            CudaBuffer::I8(b) => b.len(),
            CudaBuffer::F32(b) => b.len(),
            CudaBuffer::F64(b) => b.len(),
        }
    }

    pub fn as_raw(&self) -> usize {
        match self {
            CudaBuffer::BF16(b) => b.as_raw(),
            CudaBuffer::I8(b) => b.as_raw(),
            CudaBuffer::F32(b) => b.as_raw(),
            CudaBuffer::F64(b) => b.as_raw(),
        }
    }

    pub fn dtype(&self) -> crate::dtype::Dtype {
        match self {
            CudaBuffer::BF16(_) => crate::dtype::Dtype::BF16,
            CudaBuffer::I8(_) => crate::dtype::Dtype::I8,
            CudaBuffer::F32(_) => crate::dtype::Dtype::F32,
            CudaBuffer::F64(_) => crate::dtype::Dtype::F64,
        }
    }

    pub fn as_bf16(&self) -> Option<&DevicePtr<crate::dtype::bf16>> {
        match self {
            CudaBuffer::BF16(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_i8(&self) -> Option<&DevicePtr<i8>> {
        match self {
            CudaBuffer::I8(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&DevicePtr<f32>> {
        match self {
            CudaBuffer::F32(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&DevicePtr<f64>> {
        match self {
            CudaBuffer::F64(p) => Some(p),
            _ => None,
        }
    }
}

/// Allocate GPU memory for `count` elements of type T
#[cfg(cuda)]
pub fn alloc<T>(count: usize) -> CudaResult<DevicePtr<T>> {
    crate::cuda::init()?;
    if count == 0 {
        return Err(CudaError::InvalidInput {
            op: "cudaMalloc",
            message: "count must be greater than zero",
        });
    }
    let elem_size = std::mem::size_of::<T>();
    let size_bytes = count
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMalloc",
            count,
            elem_size,
        })?;
    let mut ptr: *mut c_void = std::ptr::null_mut();

    unsafe {
        let result = cudaMalloc(&mut ptr, size_bytes);
        if result != CUDA_SUCCESS as c_int {
            return Err(CudaError::Runtime {
                op: "cudaMalloc",
                code: result as u32,
            });
        }
    }

    Ok(DevicePtr {
        ptr: ptr as usize,
        size: count,
        pooled: false,
        _phantom: std::marker::PhantomData,
    })
}

/// Allocate GPU memory from pool if available, otherwise allocate fresh.
/// The returned buffer returns to the pool on Drop instead of being freed.
#[cfg(cuda)]
pub fn alloc_pooled<T>(count: usize) -> CudaResult<DevicePtr<T>> {
    crate::cuda::init()?;
    if count == 0 {
        return Err(CudaError::InvalidInput {
            op: "cudaMalloc",
            message: "count must be greater than zero",
        });
    }
    let elem_size = std::mem::size_of::<T>();
    let size_bytes = count
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMalloc",
            count,
            elem_size,
        })?;

    // Try pool first
    if let Ok(mut pool) = GPU_BUFFER_POOL.lock() {
        if let Some(vec) = pool.get_mut(&size_bytes) {
            if let Some(ptr) = vec.pop() {
                return Ok(DevicePtr {
                    ptr,
                    size: count,
                    pooled: true,
                    _phantom: std::marker::PhantomData,
                });
            }
        }
    }

    // Fall back to fresh allocation
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        let result = cudaMalloc(&mut ptr, size_bytes);
        if result != CUDA_SUCCESS as c_int {
            return Err(CudaError::Runtime {
                op: "cudaMalloc",
                code: result as u32,
            });
        }
    }

    Ok(DevicePtr {
        ptr: ptr as usize,
        size: count,
        pooled: true,
        _phantom: std::marker::PhantomData,
    })
}

/// Free GPU memory (automatically called when DevicePtr is dropped)
#[cfg(cuda)]
pub fn free<T>(_device: &DevicePtr<T>) -> CudaResult<()> {
    Ok(())
}

// =============================================================================
// Pinned staging + async transfers
// =============================================================================
//
// Transfer architecture: one global blocking stream + a pinned staging pool
// gated by CUDA events.
//
// Stream model. All transfers above `ASYNC_TRANSFER_THRESHOLD_BYTES` are
// enqueued with `cudaMemcpyAsync` on the process-wide *blocking* transfer
// stream (`stream::global_transfer_stream`, created with flag 0). Custom
// kernels keep launching on the legacy default stream, and every
// thread-local cuBLAS handle is bound to the transfer stream
// (`blas::new_thread_cublas`). Ordering argument:
// - copy vs. cuBLAS: same stream, ordered by stream order;
// - copy vs. custom kernel: the legacy default stream and blocking streams
//   implicitly synchronize in both directions. When work is enqueued on the
//   legacy stream, it waits for everything previously enqueued on blocking
//   streams, and vice versa. So a kernel launched after an H2D copy observes
//   the copied data, and a D2H copy enqueued after a kernel launch observes
//   the kernel's output.
//
// H2D (`copy_h2d`): host source -> CPU memcpy into a pinned staging buffer ->
// `cudaMemcpyAsync(H2D)` on the transfer stream -> `cudaEventRecord` on the
// same stream -> staging returns to the pool marked in-flight -> return
// WITHOUT waiting. The caller's host buffer may be freed immediately (its
// bytes already live in the staging buffer), and every consumer of the
// device buffer (GEMM on the transfer stream, kernels on the legacy stream)
// is stream-ordered after the copy. The CPU never reads device buffers
// directly (always via `copy_d2h`), so no host-side wait is needed.
//
// D2H (`copy_d2h` / `copy_d2h_raw`): `cudaMemcpyAsync(D2H)` into pinned
// staging on the transfer stream -> `cuStreamSynchronize(transfer stream)` ->
// CPU memcpy staging -> host destination -> staging returns to the pool idle
// (no event needed: the sync already proved completion). The synchronization
// is a semantic requirement: every D2H call site in this crate reads the
// host data immediately after the call returns. The pinned win here is
// bandwidth, not overlap. Note `cuStreamSynchronize` only waits on the
// transfer stream, but that is sufficient even for buffers written by
// kernels on the legacy default stream: when the D2H copy was enqueued on
// the blocking transfer stream it became ordered after all previously
// enqueued legacy-stream work (mutual legacy <-> blocking synchronization),
// so draining the transfer stream also drains the producing kernel.
//
// D2D (`copy_d2d`): `cudaMemcpyAsync(D2D)` on the transfer stream, no wait.
// All call sites follow up with kernels/GEMMs whose stream order is
// guaranteed by the same argument as above; none reads back on the host.
//
// Staging pool. Pinned buffers (`cudaMallocHost`) are size-bucketed like
// GPU_BUFFER_POOL, capped at `MAX_PINNED_ENTRIES_PER_SIZE` per bucket (pinned
// memory locks physical pages and must not grow unboundedly). Each buffer
// carries a lazily created, reusable `cudaEventDisableTiming` event. A
// buffer returned while its H2D copy is still in flight stores the recorded
// event; `acquire_pinned` only hands a buffer out again once
// `cudaEventQuery` reports completion.
//
// Degradation. If the global stream could not be created, or `cudaMallocHost`
// fails (pinned memory is a finite resource), or a transfer is smaller than
// `ASYNC_TRANSFER_THRESHOLD_BYTES`, the code falls back to the original
// synchronous `cudaMemcpy` on the legacy default stream. The fallback is
// globally consistent: without the global stream, cuBLAS handles also stay
// on the legacy default stream, restoring exactly the pre-async behavior.

/// Transfers strictly below this many bytes bypass the pinned/async path and
/// use the plain synchronous `cudaMemcpy`. Most transfers in this workload
/// are tiny and latency-bound; for them the staging memcpy, event recording
/// and pool locking cost more than the pageable copy they would replace.
/// 4 KiB is one page: below it the pinned bandwidth advantage is negligible.
#[cfg(cuda)]
const ASYNC_TRANSFER_THRESHOLD_BYTES: usize = 4096;

/// Upper bound on pooled pinned staging buffers per size bucket. Pinned
/// memory locks physical pages (a scarce resource), so excess buffers are
/// released back to the OS instead of being pooled.
#[cfg(cuda)]
const MAX_PINNED_ENTRIES_PER_SIZE: usize = 8;

#[cfg(cuda)]
#[inline]
fn exceeds_async_threshold(size_bytes: usize) -> bool {
    size_bytes >= ASYNC_TRANSFER_THRESHOLD_BYTES
}

/// Plain-old-data record for a pinned staging buffer. No `Drop`: ownership
/// and cleanup are managed by `PinnedBuffer` and the pool.
#[cfg(cuda)]
struct PinnedEntry {
    /// Host pointer returned by `cudaMallocHost` (stored as usize for Send).
    ptr: usize,
    /// Capacity in bytes; also the pool bucket key.
    capacity: usize,
    /// Cached `cudaEvent_t` (0 = not created yet). Created once with
    /// `cudaEventDisableTiming` and reused across recycles to avoid
    /// per-transfer create/destroy churn.
    event: usize,
    /// True when `event` was recorded after an async copy that reads this
    /// buffer and completion has not been observed yet.
    in_flight: bool,
}

/// Size-bucketed pool of pinned staging buffers, shared across threads.
/// Same locking pattern as GPU_BUFFER_POOL.
#[cfg(cuda)]
static PINNED_STAGING_POOL: std::sync::LazyLock<
    Mutex<std::collections::HashMap<usize, Vec<PinnedEntry>>>,
> = std::sync::LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

/// Warn-once flag for `cudaMallocHost` failures (fallback is per-call, so a
/// transient pinned-memory shortage does not permanently disable the path).
#[cfg(cuda)]
static PINNED_ALLOC_WARNED: AtomicBool = AtomicBool::new(false);

/// Reuse decision for a pooled pinned entry.
#[cfg(cuda)]
#[derive(Debug, PartialEq, Eq)]
enum PinnedEntryState {
    /// Idle, or its gating event has completed: safe to hand out.
    Ready,
    /// Async copy still in flight: leave it in the pool.
    Busy,
    /// `cudaEventQuery` reported a real error: the entry must be destroyed,
    /// not reused (the copy state is unknown).
    Poisoned,
}

/// Pure classification of `(in_flight, cudaEventQuery result)`; kept free of
/// FFI so the gating logic is unit-testable without a GPU.
#[cfg(cuda)]
fn classify_pinned_entry(in_flight: bool, query_code: c_int) -> PinnedEntryState {
    if !in_flight || query_code == CUDA_SUCCESS as c_int {
        PinnedEntryState::Ready
    } else if query_code == cudaErrorNotReady {
        PinnedEntryState::Busy
    } else {
        PinnedEntryState::Poisoned
    }
}

/// Synchronize (if needed) and free a pinned entry that is leaving the pool.
#[cfg(cuda)]
fn destroy_pinned_entry(entry: PinnedEntry) {
    if entry.in_flight && entry.event != 0 {
        let code = unsafe { cudaEventSynchronize(entry.event as cudaEvent_t) };
        if code != CUDA_SUCCESS as c_int {
            // The async copy may still be reading this buffer and we cannot
            // prove otherwise; freeing pinned memory under a live copy is
            // undefined behavior. Leak the buffer and its event — this only
            // happens when the CUDA context is already broken.
            return;
        }
    }
    if entry.event != 0 {
        unsafe {
            cudaEventDestroy(entry.event as cudaEvent_t);
        }
    }
    unsafe {
        cudaFreeHost(entry.ptr as *mut c_void);
    }
}

/// RAII handle to a pinned staging buffer. On drop the buffer returns to the
/// pool (carrying its in-flight event, if any) or is freed when the bucket is
/// full.
#[cfg(cuda)]
struct PinnedBuffer {
    entry: PinnedEntry,
}

#[cfg(cuda)]
impl PinnedBuffer {
    fn as_ptr(&self) -> *const u8 {
        self.entry.ptr as *const u8
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.entry.ptr as *mut u8
    }

    /// Mark this buffer as read by an async copy just enqueued on `stream`:
    /// lazily create the gating event and record it. If the event cannot be
    /// created or recorded, fall back to draining the stream right here so
    /// the buffer is provably idle when it returns to the pool.
    fn record_in_flight(&mut self, stream: CUstream) {
        if self.entry.event == 0 {
            let mut event: cudaEvent_t = std::ptr::null_mut();
            let code = unsafe { cudaEventCreateWithFlags(&mut event, cudaEventDisableTiming) };
            if code == CUDA_SUCCESS as c_int && !event.is_null() {
                self.entry.event = event as usize;
            }
        }
        if self.entry.event != 0 {
            let code = unsafe { cudaEventRecord(self.entry.event as cudaEvent_t, stream) };
            if code == CUDA_SUCCESS as c_int {
                self.entry.in_flight = true;
                return;
            }
        }
        // No usable event: wait for the copy now (degrades this transfer to
        // synchronous, but keeps the pool invariant intact).
        let code = unsafe { cuStreamSynchronize(stream) };
        if code != CUDA_SUCCESS {
            // Sync failed, so the in-flight copy may still be reading this
            // buffer and we cannot prove otherwise. Returning it to the pool
            // as idle would let the next user overwrite it mid-copy; leak it
            // in place instead (same rationale as `leak`).
            log::warn!(
                "[CUDA] cuStreamSynchronize failed (code {code}) while gating \
                 a pinned staging buffer; leaking the buffer"
            );
            self.entry.ptr = 0;
            self.entry.event = 0;
            self.entry.in_flight = false;
        }
    }

    /// Deliberately leak the buffer (and its event). Used when a stream
    /// synchronization failed and the copy state is unknown: neither pooling
    /// nor `cudaFreeHost` would be safe.
    fn leak(mut self) {
        self.entry.ptr = 0;
        self.entry.event = 0;
        self.entry.in_flight = false;
    }
}

#[cfg(cuda)]
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        let entry = std::mem::replace(
            &mut self.entry,
            PinnedEntry {
                ptr: 0,
                capacity: 0,
                event: 0,
                in_flight: false,
            },
        );
        if entry.ptr == 0 {
            return;
        }
        if let Ok(mut pool) = PINNED_STAGING_POOL.lock() {
            let bucket = pool.entry(entry.capacity).or_default();
            if bucket.len() < MAX_PINNED_ENTRIES_PER_SIZE {
                bucket.push(entry);
                return;
            }
        }
        destroy_pinned_entry(entry);
    }
}

/// Take a ready pinned staging buffer of exactly `size_bytes` from the pool,
/// or allocate a fresh one. Returns `None` when `cudaMallocHost` fails, in
/// which case the caller must fall back to the synchronous pageable path.
#[cfg(cuda)]
fn acquire_pinned(size_bytes: usize) -> Option<PinnedBuffer> {
    // Poisoned entries are collected under the lock but destroyed after it is
    // released: `destroy_pinned_entry` may block in `cudaEventSynchronize`,
    // and holding the pool mutex across that would stall every other thread
    // that needs a staging buffer. `cudaEventQuery` itself is non-blocking.
    let mut found: Option<PinnedEntry> = None;
    let mut poisoned: Vec<PinnedEntry> = Vec::new();
    if let Ok(mut pool) = PINNED_STAGING_POOL.lock() {
        if let Some(bucket) = pool.get_mut(&size_bytes) {
            let mut idx = 0;
            while idx < bucket.len() {
                let state = if bucket[idx].in_flight {
                    let code = unsafe { cudaEventQuery(bucket[idx].event as cudaEvent_t) };
                    classify_pinned_entry(true, code)
                } else {
                    PinnedEntryState::Ready
                };
                match state {
                    PinnedEntryState::Ready => {
                        found = Some(bucket.swap_remove(idx));
                        break;
                    }
                    PinnedEntryState::Busy => idx += 1,
                    PinnedEntryState::Poisoned => {
                        poisoned.push(bucket.swap_remove(idx));
                    }
                }
            }
        }
    }
    for entry in poisoned {
        destroy_pinned_entry(entry);
    }
    if let Some(mut entry) = found {
        entry.in_flight = false;
        return Some(PinnedBuffer { entry });
    }

    let mut ptr: *mut c_void = std::ptr::null_mut();
    let code = unsafe { cudaMallocHost(&mut ptr, size_bytes) };
    if code != CUDA_SUCCESS as c_int || ptr.is_null() {
        if !PINNED_ALLOC_WARNED.swap(true, Ordering::Relaxed) {
            log::warn!(
                "[CUDA] cudaMallocHost({size_bytes} bytes) failed (code {code}); \
                 falling back to synchronous pageable transfers when pinned \
                 staging is unavailable"
            );
        }
        return None;
    }
    Some(PinnedBuffer {
        entry: PinnedEntry {
            ptr: ptr as usize,
            capacity: size_bytes,
            event: 0,
            in_flight: false,
        },
    })
}

/// Synchronous `cudaMemcpy` core shared by all fallback paths.
///
/// # Safety
/// `dst`/`src` must be valid for `size_bytes` according to `kind`.
#[cfg(cuda)]
unsafe fn memcpy_sync(
    dst: *mut c_void,
    src: *const c_void,
    size_bytes: usize,
    kind: c_int,
    op: &'static str,
) -> CudaResult<()> {
    let result = cudaMemcpy(dst, src, size_bytes, kind);
    if result != CUDA_SUCCESS as c_int {
        return Err(CudaError::Runtime {
            op,
            code: result as u32,
        });
    }
    Ok(())
}

/// Host-to-device byte transfer: pinned staging + async copy when profitable,
/// synchronous `cudaMemcpy` otherwise. Returns as soon as the copy is
/// enqueued (async path) — see the module-level transfer notes for why this
/// is safe.
///
/// # Safety
/// `src` must be valid for `size_bytes` reads and `dst_device` must be a
/// device allocation of at least `size_bytes`.
#[cfg(cuda)]
unsafe fn h2d_bytes(dst_device: usize, src: *const u8, size_bytes: usize) -> CudaResult<()> {
    if exceeds_async_threshold(size_bytes) {
        if let Some(stream) = crate::cuda::stream::global_transfer_stream() {
            if let Some(mut staging) = acquire_pinned(size_bytes) {
                std::ptr::copy_nonoverlapping(src, staging.as_mut_ptr(), size_bytes);
                let code = cudaMemcpyAsync(
                    dst_device as *mut c_void,
                    staging.as_ptr().cast::<c_void>(),
                    size_bytes,
                    cudaMemcpyHostToDevice,
                    stream,
                );
                if code != CUDA_SUCCESS as c_int {
                    // Nothing was enqueued; `staging` is idle and returns to
                    // the pool through its Drop.
                    return Err(CudaError::Runtime {
                        op: "cudaMemcpyAsync(H2D)",
                        code: code as u32,
                    });
                }
                // Gate future reuse of the staging buffer on copy completion,
                // then return it to the pool (Drop) without waiting.
                staging.record_in_flight(stream);
                return Ok(());
            }
        }
    }
    memcpy_sync(
        dst_device as *mut c_void,
        src.cast::<c_void>(),
        size_bytes,
        cudaMemcpyHostToDevice,
        "cudaMemcpy(H2D)",
    )
}

/// Device-to-host byte transfer. Always complete on return (every call site
/// reads the host data immediately); the pinned path buys transfer bandwidth,
/// not overlap.
///
/// # Safety
/// `dst` must be valid for `size_bytes` writes and `src_device` must be a
/// device allocation of at least `size_bytes`.
#[cfg(cuda)]
unsafe fn d2h_bytes(dst: *mut u8, src_device: usize, size_bytes: usize) -> CudaResult<()> {
    if exceeds_async_threshold(size_bytes) {
        if let Some(stream) = crate::cuda::stream::global_transfer_stream() {
            if let Some(mut staging) = acquire_pinned(size_bytes) {
                let code = cudaMemcpyAsync(
                    staging.as_mut_ptr().cast::<c_void>(),
                    src_device as *const c_void,
                    size_bytes,
                    cudaMemcpyDeviceToHost,
                    stream,
                );
                if code != CUDA_SUCCESS as c_int {
                    // Nothing was enqueued; staging returns to the pool idle.
                    return Err(CudaError::Runtime {
                        op: "cudaMemcpyAsync(D2H)",
                        code: code as u32,
                    });
                }
                // Draining the (blocking) transfer stream is sufficient even
                // when the source buffer was written by a kernel on the
                // legacy default stream: enqueuing the copy on a blocking
                // stream ordered it after all previously enqueued legacy
                // work, so the sync below also waits for the producer.
                let sync_code = cuStreamSynchronize(stream);
                if sync_code != CUDA_SUCCESS {
                    // Copy state unknown: the device may still write into the
                    // staging buffer, so neither pooling nor freeing is safe.
                    staging.leak();
                    return Err(CudaError::Runtime {
                        op: "cuStreamSynchronize(D2H)",
                        code: sync_code,
                    });
                }
                std::ptr::copy_nonoverlapping(staging.as_ptr(), dst, size_bytes);
                // The sync proved completion; staging returns to the pool
                // idle (no event gating needed) through its Drop.
                return Ok(());
            }
        }
    }
    memcpy_sync(
        dst.cast::<c_void>(),
        src_device as *const c_void,
        size_bytes,
        cudaMemcpyDeviceToHost,
        "cudaMemcpy(D2H)",
    )
}

/// Device-to-device byte transfer: async on the transfer stream when
/// available (no host wait needed — every consumer is stream-ordered after
/// the copy), synchronous otherwise. No staging or events are involved, so
/// no small-size bypass is needed either.
///
/// # Safety
/// `dst_device` and `src_device` must be device allocations of at least
/// `size_bytes`.
#[cfg(cuda)]
unsafe fn d2d_bytes(dst_device: usize, src_device: usize, size_bytes: usize) -> CudaResult<()> {
    if let Some(stream) = crate::cuda::stream::global_transfer_stream() {
        let code = cudaMemcpyAsync(
            dst_device as *mut c_void,
            src_device as *const c_void,
            size_bytes,
            cudaMemcpyDeviceToDevice,
            stream,
        );
        if code != CUDA_SUCCESS as c_int {
            return Err(CudaError::Runtime {
                op: "cudaMemcpyAsync(D2D)",
                code: code as u32,
            });
        }
        return Ok(());
    }
    memcpy_sync(
        dst_device as *mut c_void,
        src_device as *const c_void,
        size_bytes,
        cudaMemcpyDeviceToDevice,
        "cudaMemcpy(D2D)",
    )
}

/// Copy data from host (CPU) to device (GPU).
///
/// Large transfers go through pinned staging + `cudaMemcpyAsync` on the
/// global transfer stream and return as soon as the copy is enqueued; `host`
/// may be dropped immediately after this returns (its bytes were copied into
/// the staging buffer first). Small transfers and degraded configurations
/// use the original synchronous `cudaMemcpy`.
#[cfg(cuda)]
pub fn copy_h2d<T: Copy>(device: &DevicePtr<T>, host: &[T]) -> CudaResult<()> {
    crate::cuda::init()?;
    if host.len() != device.size {
        return Err(CudaError::SizeMismatch {
            op: "cudaMemcpy(H2D)",
            expected: device.size,
            actual: host.len(),
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = host
        .len()
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMemcpy(H2D)",
            count: host.len(),
            elem_size,
        })?;
    unsafe { h2d_bytes(device.ptr, host.as_ptr().cast::<u8>(), size_bytes) }
}

/// Copy data from device (GPU) to host (CPU) - complete on return.
///
/// Large transfers stream through a pinned staging buffer for bandwidth, but
/// the call always synchronizes before returning: every call site reads the
/// host data immediately.
#[cfg(cuda)]
pub fn copy_d2h<T: Copy>(host: &mut [T], device: &DevicePtr<T>) -> CudaResult<()> {
    crate::cuda::init()?;
    if host.len() != device.size {
        return Err(CudaError::SizeMismatch {
            op: "cudaMemcpy(D2H)",
            expected: device.size,
            actual: host.len(),
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = host
        .len()
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMemcpy(D2H)",
            count: host.len(),
            elem_size,
        })?;
    unsafe { d2h_bytes(host.as_mut_ptr().cast::<u8>(), device.ptr, size_bytes) }
}

/// Copy data from device (GPU) to device (GPU).
///
/// Enqueued asynchronously on the global transfer stream when available; no
/// host-side wait is needed because every consumer (kernel, GEMM, `copy_d2h`)
/// is stream-ordered after the copy.
#[cfg(cuda)]
pub fn copy_d2d<T: Copy>(dst: &DevicePtr<T>, src: &DevicePtr<T>) -> CudaResult<()> {
    crate::cuda::init()?;
    if dst.size != src.size {
        return Err(CudaError::SizeMismatch {
            op: "cudaMemcpy(D2D)",
            expected: dst.size,
            actual: src.size,
        });
    }

    let elem_size = std::mem::size_of::<T>();
    let size_bytes = dst
        .size
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMemcpy(D2D)",
            count: dst.size,
            elem_size,
        })?;
    unsafe { d2d_bytes(dst.ptr, src.ptr, size_bytes) }
}

/// Copy data from device (GPU) to host (CPU) using raw pointers.
/// Does NOT free the GPU memory (caller manages lifetime).
///
/// Shares the `copy_d2h` transfer path so both interfaces keep identical
/// semantics: the copy is always complete when this returns.
///
/// # Safety
/// `host` must be valid for `count` writes of `T`, `device_ptr` must point to
/// a device allocation of at least `count` elements, and CUDA must already be
/// initialized.
#[cfg(cuda)]
pub unsafe fn copy_d2h_raw<T: Copy>(
    host: *mut T,
    device_ptr: usize,
    count: usize,
) -> CudaResult<()> {
    if host.is_null() {
        return Err(CudaError::InvalidInput {
            op: "cudaMemcpy(D2H)",
            message: "host pointer must not be null",
        });
    }
    if device_ptr == 0 {
        return Err(CudaError::InvalidInput {
            op: "cudaMemcpy(D2H)",
            message: "device pointer must not be zero",
        });
    }
    let elem_size = std::mem::size_of::<T>();
    let size_bytes = count
        .checked_mul(elem_size)
        .ok_or(CudaError::SizeOverflow {
            op: "cudaMemcpy(D2H)",
            count,
            elem_size,
        })?;
    d2h_bytes(host.cast::<u8>(), device_ptr, size_bytes)
}

// =============================================================================
// Stub implementations for non-CUDA builds
// =============================================================================

#[cfg(not(cuda))]
#[derive(Clone)]
pub struct DevicePtr<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(not(cuda))]
impl<T> DevicePtr<T> {
    pub fn len(&self) -> usize {
        0
    }
    pub fn as_raw(&self) -> usize {
        0
    }
    pub fn zero_sized() -> Self {
        DevicePtr {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(not(cuda))]
pub fn alloc<T>(_count: usize) -> CudaResult<DevicePtr<T>> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::alloc",
    })
}

#[cfg(not(cuda))]
pub fn alloc_pooled<T>(_count: usize) -> CudaResult<DevicePtr<T>> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::alloc_pooled",
    })
}

#[cfg(not(cuda))]
pub fn free<T>(_device: &DevicePtr<T>) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::free",
    })
}

#[cfg(not(cuda))]
pub fn copy_h2d<T: Copy>(_device: &DevicePtr<T>, _host: &[T]) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::copy_h2d",
    })
}

#[cfg(not(cuda))]
pub fn copy_d2h<T: Copy>(_host: &mut [T], _device: &DevicePtr<T>) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::copy_d2h",
    })
}

#[cfg(not(cuda))]
pub fn copy_d2d<T: Copy>(_dst: &DevicePtr<T>, _src: &DevicePtr<T>) -> CudaResult<()> {
    Err(CudaError::UnsupportedBuild {
        op: "cuda::memory::copy_d2d",
    })
}

#[cfg(all(test, cuda))]
mod tests {
    use super::*;

    #[test]
    fn small_transfers_bypass_async_path() {
        assert!(!exceeds_async_threshold(0));
        assert!(!exceeds_async_threshold(1));
        assert!(!exceeds_async_threshold(ASYNC_TRANSFER_THRESHOLD_BYTES - 1));
        assert!(exceeds_async_threshold(ASYNC_TRANSFER_THRESHOLD_BYTES));
        assert!(exceeds_async_threshold(ASYNC_TRANSFER_THRESHOLD_BYTES + 1));
    }

    #[test]
    fn pinned_entry_classification_gates_reuse() {
        // Idle entries are always reusable; the query result is irrelevant.
        assert_eq!(
            classify_pinned_entry(false, 0),
            PinnedEntryState::Ready,
            "idle entry must be reusable"
        );
        // In-flight entry whose event completed.
        assert_eq!(
            classify_pinned_entry(true, CUDA_SUCCESS as c_int),
            PinnedEntryState::Ready
        );
        // cudaErrorNotReady is the normal "still running" answer, not an
        // error: the entry stays in the pool.
        assert_eq!(
            classify_pinned_entry(true, cudaErrorNotReady),
            PinnedEntryState::Busy
        );
        // Any real error means the copy state is unknown.
        assert_eq!(classify_pinned_entry(true, 1), PinnedEntryState::Poisoned);
        assert_eq!(classify_pinned_entry(true, 700), PinnedEntryState::Poisoned);
    }

    /// Requires a real GPU; skipped (early return) when CUDA init fails.
    #[test]
    fn h2d_d2h_roundtrip_above_and_below_threshold() {
        if crate::cuda::init().is_err() {
            return;
        }

        // Above the async threshold: 4096 f32 = 16 KiB.
        let count = 4096;
        let device = alloc::<f32>(count).expect("device alloc");
        let host: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();
        copy_h2d(&device, &host).expect("h2d (async path)");
        let mut out = vec![0.0f32; count];
        copy_d2h(&mut out, &device).expect("d2h (async path)");
        assert_eq!(host, out);

        // Below the threshold: 8 f32 = 32 bytes (sync bypass).
        let count = 8;
        let device_small = alloc::<f32>(count).expect("device alloc");
        let host_small: Vec<f32> = (0..count).map(|i| -(i as f32)).collect();
        copy_h2d(&device_small, &host_small).expect("h2d (sync bypass)");
        let mut out_small = vec![0.0f32; count];
        copy_d2h(&mut out_small, &device_small).expect("d2h (sync bypass)");
        assert_eq!(host_small, out_small);
    }

    /// Requires a real GPU; skipped (early return) when CUDA init fails.
    #[test]
    fn d2d_copy_roundtrip() {
        if crate::cuda::init().is_err() {
            return;
        }

        let count = 4096;
        let src = alloc::<f32>(count).expect("device alloc src");
        let dst = alloc::<f32>(count).expect("device alloc dst");
        let host: Vec<f32> = (0..count).map(|i| (i % 97) as f32).collect();
        copy_h2d(&src, &host).expect("h2d");
        copy_d2d(&dst, &src).expect("d2d");
        let mut out = vec![0.0f32; count];
        copy_d2h(&mut out, &dst).expect("d2h");
        assert_eq!(host, out);
    }

    /// Requires a real GPU; skipped (early return) when CUDA init fails.
    /// Uses a bucket size no other test touches so parallel test execution
    /// cannot steal the pooled entry.
    #[test]
    fn pinned_pool_recycles_completed_staging() {
        if crate::cuda::init().is_err() {
            return;
        }

        let size_bytes = ASYNC_TRANSFER_THRESHOLD_BYTES * 3;
        let first_ptr;
        {
            let staging = acquire_pinned(size_bytes).expect("pinned alloc");
            first_ptr = staging.entry.ptr;
            assert!(!staging.entry.in_flight);
            // Dropped idle -> returns straight to the pool.
        }
        let staging = acquire_pinned(size_bytes).expect("pinned realloc");
        assert_eq!(
            staging.entry.ptr, first_ptr,
            "idle staging buffer must be recycled from the pool"
        );
    }
}
