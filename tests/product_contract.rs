use serde_json::Value;
use std::path::{Path, PathBuf};

fn repo_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn read(relative: &str) -> String {
    std::fs::read_to_string(repo_path(relative))
        .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"))
}

fn shipped_pool_data() -> Value {
    serde_json::from_str(&read("data/pools.json")).expect("data/pools.json must be valid JSON")
}

#[test]
fn product_docs_match_the_shipped_active_pool_contract() {
    let pools = shipped_pool_data();
    let active_id = pools["active_pool"]
        .as_str()
        .expect("active_pool must be a string");
    let active_pool = pools["pools"]
        .as_array()
        .expect("pools must be an array")
        .iter()
        .find(|pool| pool["id"].as_str() == Some(active_id))
        .expect("active_pool must reference an existing pool");
    let up_rate = active_pool["up_rate"]
        .as_f64()
        .expect("active pool up_rate must be numeric");
    let percent = up_rate * 100.0;
    let readme = read("README.md");

    let english_contract = format!(
        "Shipped active pool contract: `{active_id}` uses **{percent:.0}%** (`up_rate = {up_rate}`)."
    );
    let chinese_contract = format!(
        "随附激活卡池约定：`{active_id}` 使用 **{percent:.0}%**（`up_rate = {up_rate}`）。"
    );

    assert!(
        readme.contains(&english_contract),
        "README must state the active pool value derived from data/pools.json: {english_contract}"
    );
    assert!(
        readme.contains(&chinese_contract),
        "README must state the active pool value derived from data/pools.json: {chinese_contract}"
    );
}

#[test]
fn external_pool_file_has_no_duplicate_active_pool_fields() {
    let pools = shipped_pool_data();

    for field in [
        "pool_name",
        "pool_name_en",
        "up_six",
        "up_six_en",
        "up_rate",
        "prob_6_base",
        "prob_5_base",
        "prob_4_base",
        "soft_pity_start",
        "small_pity_guarantee",
        "big_pity_cumulative",
        "up_pity_soft",
        "five_star_pity",
        "always_5_star",
        "big_pity_requires_not_up",
        "six_stars",
        "six_stars_en",
        "five_stars",
        "five_stars_en",
        "four_stars",
        "four_stars_en",
    ] {
        assert!(
            pools.get(field).is_none(),
            "data/pools.json must define {field} only inside pools[]"
        );
    }
}

#[test]
fn product_docs_do_not_freeze_volatile_counts_or_claim_stale_behavior() {
    let readme = read("README.md");
    let usage = read("docs/USAGE.md");
    let agents = read("AGENTS.md");

    for stale_claim in [
        "a DBN (Deep Belief Network) fits the environment noise distribution",
        "训练 DBN 建模环境噪声",
        "**probability** — pure dice roll per config probability table",
        "**probability** — 纯概率，按配置的概率表投骰",
        "decides \"pull or wait\"",
        "决定\"抽还是不抽\"",
        "continuous pull strategy",
        "连续抽卡策略",
        "should I pull or wait?",
        "continuous pull-strategy distribution",
        "pure dice roll against the config probability table",
        "discrete \"pull vs wait\" decisions",
        "caches stay valid across pool updates",
        "default 2048 in `data/config.json`",
        "All settings centralized here: pool definitions",
        "PPO (20k steps by default)",
        "PPO（默认 20k 步）",
        "all 7 experiments, 3 trials",
        "默认 3 次试验",
        "calibrate models with collected data",
        "用采集数据校准模型",
    ] {
        for (path, contents) in [("README.md", &readme), ("docs/USAGE.md", &usage)] {
            assert!(
                !contents.contains(stale_claim),
                "{path} contains stale product behavior: {stale_claim}"
            );
        }
    }

    assert!(
        !readme.contains("## Testing ("),
        "README must not hard-code the changing test count"
    );
    assert!(
        !readme.contains("## 测试（"),
        "README must not hard-code the changing test count"
    );
    assert!(
        !agents.contains("tests)"),
        "AGENTS.md command descriptions must not hard-code the changing test count"
    );
}
