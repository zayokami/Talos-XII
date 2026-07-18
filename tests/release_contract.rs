use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn section_version(contents: &str, section: &str) -> Option<String> {
    let mut in_section = false;
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_section = trimmed == format!("[{section}]");
            continue;
        }
        if in_section {
            if let Some(value) = trimmed.strip_prefix("version = ") {
                return Some(value.trim_matches('"').to_string());
            }
        }
    }
    None
}

#[test]
fn rust_and_python_package_versions_stay_in_sync() {
    let version = env!("CARGO_PKG_VERSION");
    let binding_manifest =
        std::fs::read_to_string(repo_path("bindings/python/Cargo.toml")).unwrap();
    let pyproject = std::fs::read_to_string(repo_path("pyproject.toml")).unwrap();

    assert_eq!(
        section_version(&binding_manifest, "package").as_deref(),
        Some(version)
    );
    assert_eq!(
        section_version(&pyproject, "project").as_deref(),
        Some(version)
    );
}

#[test]
fn release_packaging_contains_runtime_contract_files() {
    for relative in [
        "LICENSE",
        "README.md",
        "data/config.json",
        "data/pools.json",
        "docs/REQUIREMENTS.md",
        "docs/PYTORCH_COMPATIBILITY.md",
        "docs/USAGE.md",
        "bindings/python/python/talos_xii/__init__.pyi",
        "bindings/python/python/talos_xii/py.typed",
        "bindings/python/tests/test_tensor_contract.py",
        "bindings/python/tests/test_torch_differential.py",
        "scripts/package-release.ps1",
    ] {
        assert!(repo_path(relative).is_file(), "missing {relative}");
    }
}

#[test]
fn cargo_package_globs_exclude_generated_python_artifacts() {
    let manifest = std::fs::read_to_string(repo_path("Cargo.toml")).unwrap();

    assert!(manifest.contains(r#""/examples/**/*.py""#));
    assert!(manifest.contains(r#""/bindings/python/python/**/*.pyi""#));
    assert!(!manifest.contains(r#""/examples/**""#));
    assert!(!manifest.contains(r#""/bindings/python/python/**""#));
}
