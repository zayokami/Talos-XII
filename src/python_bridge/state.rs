use crate::autograd::Tensor;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock, Weak};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TensorKey {
    data: usize,
    grad: usize,
}

#[derive(Clone)]
struct VersionDependency {
    counter: Arc<AtomicU64>,
    expected: u64,
}

pub(super) struct PythonAutogradState {
    requires_grad: AtomicBool,
    is_leaf: bool,
    retain_grad: AtomicBool,
    grad_ready: AtomicBool,
    graph_consumed: AtomicBool,
    version: Arc<AtomicU64>,
    dependencies: Vec<VersionDependency>,
}

static STATE_REGISTRY: OnceLock<RwLock<HashMap<TensorKey, Weak<PythonAutogradState>>>> =
    OnceLock::new();

fn registry() -> &'static RwLock<HashMap<TensorKey, Weak<PythonAutogradState>>> {
    STATE_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

fn tensor_key(tensor: &Tensor) -> TensorKey {
    TensorKey {
        data: tensor.data.id(),
        grad: tensor.grad.id(),
    }
}

fn lookup(tensor: &Tensor) -> Option<Arc<PythonAutogradState>> {
    registry()
        .read()
        .ok()?
        .get(&tensor_key(tensor))
        .and_then(Weak::upgrade)
}

fn register(tensor: &Tensor, state: &Arc<PythonAutogradState>) {
    if let Ok(mut states) = registry().write() {
        if states.len() > 4096 {
            states.retain(|_, state| state.strong_count() > 0);
        }
        states.insert(tensor_key(tensor), Arc::downgrade(state));
    }
}

impl PythonAutogradState {
    pub(super) fn leaf(tensor: &Tensor, requires_grad: bool) -> Arc<Self> {
        let state = Arc::new(Self {
            requires_grad: AtomicBool::new(requires_grad),
            is_leaf: true,
            retain_grad: AtomicBool::new(false),
            grad_ready: AtomicBool::new(false),
            graph_consumed: AtomicBool::new(false),
            version: Arc::new(AtomicU64::new(0)),
            dependencies: Vec::new(),
        });
        register(tensor, &state);
        state
    }

    pub(super) fn operation(tensor: &mut Tensor, grad_enabled: bool) -> Arc<Self> {
        let parent_states = collect_registered_ancestors(tensor);
        let requires_grad =
            grad_enabled && parent_states.iter().any(|(_, state)| state.requires_grad());

        let version = parent_states
            .iter()
            .find(|(key, _)| key.data == tensor.data.id())
            .map(|(_, state)| state.version.clone())
            .unwrap_or_else(|| Arc::new(AtomicU64::new(0)));

        let mut dependencies = Vec::new();
        let mut seen = HashSet::new();
        if requires_grad {
            for (_, parent) in &parent_states {
                if !parent.requires_grad() {
                    continue;
                }
                push_dependency(
                    &mut dependencies,
                    &mut seen,
                    parent.version.clone(),
                    parent.version.load(Ordering::Acquire),
                );
                for dependency in &parent.dependencies {
                    push_dependency(
                        &mut dependencies,
                        &mut seen,
                        dependency.counter.clone(),
                        dependency.expected,
                    );
                }
            }
        } else {
            tensor.clear_graph();
        }

        let state = Arc::new(Self {
            requires_grad: AtomicBool::new(requires_grad),
            is_leaf: !requires_grad,
            retain_grad: AtomicBool::new(false),
            grad_ready: AtomicBool::new(false),
            graph_consumed: AtomicBool::new(false),
            version,
            dependencies,
        });
        register(tensor, &state);
        state
    }

    pub(super) fn alias(
        tensor: &Tensor,
        source: &Arc<Self>,
        requires_grad: bool,
        is_leaf: bool,
    ) -> Arc<Self> {
        let state = Arc::new(Self {
            requires_grad: AtomicBool::new(requires_grad),
            is_leaf,
            retain_grad: AtomicBool::new(false),
            grad_ready: AtomicBool::new(false),
            graph_consumed: AtomicBool::new(false),
            version: source.version.clone(),
            dependencies: if requires_grad {
                source.dependencies.clone()
            } else {
                Vec::new()
            },
        });
        register(tensor, &state);
        state
    }

    pub(super) fn requires_grad(&self) -> bool {
        self.requires_grad.load(Ordering::Acquire)
    }

    pub(super) fn set_requires_grad(&self, value: bool) {
        self.requires_grad.store(value, Ordering::Release);
        if !value {
            self.grad_ready.store(false, Ordering::Release);
        }
    }

    pub(super) fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    pub(super) fn retain_grad(&self) -> bool {
        self.retain_grad.load(Ordering::Acquire)
    }

    pub(super) fn set_retain_grad(&self, value: bool) {
        self.retain_grad.store(value, Ordering::Release);
    }

    pub(super) fn grad_ready(&self) -> bool {
        self.grad_ready.load(Ordering::Acquire)
    }

    pub(super) fn set_grad_ready(&self, value: bool) {
        self.grad_ready.store(value, Ordering::Release);
    }

    pub(super) fn graph_consumed(&self) -> bool {
        self.graph_consumed.load(Ordering::Acquire)
    }

    pub(super) fn set_graph_consumed(&self) {
        self.graph_consumed.store(true, Ordering::Release);
    }

    pub(super) fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    pub(super) fn increment_version(&self) {
        self.version.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn check_versions(&self) -> Result<(), String> {
        for dependency in &self.dependencies {
            let actual = dependency.counter.load(Ordering::Acquire);
            if actual != dependency.expected {
                return Err(format!(
                    "one of the variables needed for gradient computation was modified by an in-place operation: expected version {}, found version {}",
                    dependency.expected, actual
                ));
            }
        }
        Ok(())
    }
}

fn collect_registered_ancestors(tensor: &Tensor) -> Vec<(TensorKey, Arc<PythonAutogradState>)> {
    let mut states = Vec::new();
    let mut visited = HashSet::new();
    let mut stack: Vec<Tensor> = tensor
        ._ctx
        .as_ref()
        .map(|context| context.parents.clone())
        .unwrap_or_default();
    while let Some(parent) = stack.pop() {
        let key = tensor_key(&parent);
        if !visited.insert(key) {
            continue;
        }
        if let Some(state) = lookup(&parent) {
            states.push((key, state));
            continue;
        }
        if let Some(context) = &parent._ctx {
            stack.extend(context.parents.iter().cloned());
        }
    }
    states
}

fn push_dependency(
    dependencies: &mut Vec<VersionDependency>,
    seen: &mut HashSet<usize>,
    counter: Arc<AtomicU64>,
    expected: u64,
) {
    let id = Arc::as_ptr(&counter) as usize;
    if seen.insert(id) {
        dependencies.push(VersionDependency { counter, expected });
    }
}

pub(super) fn mark_gradients_ready(root: &Tensor) {
    let mut visited = HashSet::new();
    let mut stack = vec![root.clone()];
    while let Some(tensor) = stack.pop() {
        if !visited.insert(tensor_key(&tensor)) {
            continue;
        }
        if let Some(state) = lookup(&tensor) {
            if state.requires_grad() && (state.is_leaf() || state.retain_grad()) {
                state.set_grad_ready(true);
            }
        }
        if let Some(context) = &tensor._ctx {
            stack.extend(context.parents.iter().cloned());
        }
    }
}
