use serde_json::{json, Value};
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

fn pool_by_id<'a>(pools: &'a Value, id: &str) -> &'a Value {
    pools["pools"]
        .as_array()
        .expect("pools must be an array")
        .iter()
        .find(|pool| pool["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("pool {id} must exist"))
}

#[test]
fn product_docs_match_the_shipped_active_pool_contract() {
    let pools = shipped_pool_data();
    let active_id = pools["active_pool"]
        .as_str()
        .expect("active_pool must be a string");
    let active_pool = pool_by_id(&pools, active_id);
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
fn shipped_v1_4_pools_match_announced_rosters() {
    let pools = shipped_pool_data();
    assert_eq!(pools["active_pool"].as_str(), Some("char_up_20260716"));

    let phase_one = pool_by_id(&pools, "char_up_20260716");
    assert_eq!(phase_one["name"].as_str(), Some("临渊望北"));
    assert_eq!(
        phase_one["name_en"].as_str(),
        Some("North Yearns The Rift Vigile")
    );
    assert_eq!(phase_one["up_six"], json!(["诀"]));
    assert_eq!(phase_one["up_six_en"], json!(["Arcane"]));
    assert_eq!(
        phase_one["six_stars"],
        json!([
            "诀",
            "卡缪",
            "弭弗",
            "余烬",
            "黎风",
            "艾尔黛拉",
            "别礼",
            "骏卫"
        ])
    );

    let phase_two = pool_by_id(&pools, "char_up_20260809");
    assert_eq!(phase_two["name"].as_str(), Some("晨星于此闪耀"));
    assert_eq!(
        phase_two["name_en"].as_str(),
        Some("Good Morning From Your Dawnstar")
    );
    assert_eq!(phase_two["up_six"], json!(["梨诺"]));
    assert_eq!(phase_two["up_six_en"], json!(["Liino"]));
    assert_eq!(
        phase_two["six_stars"],
        json!([
            "梨诺",
            "诀",
            "卡缪",
            "弭弗",
            "余烬",
            "黎风",
            "艾尔黛拉",
            "别礼",
            "骏卫"
        ])
    );

    let weapon_phase_one = pool_by_id(&pools, "weapon_up_20260716");
    assert_eq!(weapon_phase_one["name"].as_str(), Some("军列申领"));
    assert_eq!(
        weapon_phase_one["name_en"].as_str(),
        Some("Military Grade Issue")
    );
    assert_eq!(
        weapon_phase_one["up_six"],
        json!(["四二式·肃阵（施术单元）"])
    );
    assert_eq!(
        weapon_phase_one["up_six_en"],
        json!(["Type 42: Solemn Phalanx (Arts Unit)"])
    );
    assert_eq!(
        weapon_phase_one["six_stars"],
        json!([
            "四二式·肃阵（施术单元）",
            "昔日精品（双手剑）",
            "楔子（手铳）",
            "显赫声名（单手剑）",
            "J.E.T.（长柄武器）",
            "爆破单元（施术单元）",
            "遗忘（施术单元）"
        ])
    );

    let weapon_phase_two = pool_by_id(&pools, "weapon_up_20260809");
    assert_eq!(weapon_phase_two["name"].as_str(), Some("明耀申领"));
    assert_eq!(
        weapon_phase_two["name_en"].as_str(),
        Some("Bedazzled Issue")
    );
    assert_eq!(
        weapon_phase_two["up_six"],
        json!(["曜夜的首演（长柄武器）"])
    );
    assert_eq!(
        weapon_phase_two["up_six_en"],
        json!(["Bedazzling Night Debut (Polearm)"])
    );
    assert_eq!(
        weapon_phase_two["six_stars"],
        json!([
            "曜夜的首演（长柄武器）",
            "破碎君王（双手剑）",
            "同类相食（手铳）",
            "骁勇（长柄武器）",
            "J.E.T.（长柄武器）",
            "爆破单元（施术单元）",
            "热熔切割器（单手剑）"
        ])
    );

    for pool in [phase_one, phase_two] {
        assert_eq!(pool["up_rate"].as_f64(), Some(0.5));
        assert_eq!(pool["small_pity_guarantee"].as_u64(), Some(80));
        assert_eq!(pool["big_pity_cumulative"].as_u64(), Some(120));
        assert_ne!(pool["is_archived"].as_bool(), Some(true));
    }

    for pool in [weapon_phase_one, weapon_phase_two] {
        assert_eq!(pool["up_rate"].as_f64(), Some(0.5));
        assert_eq!(pool["prob_6_base"].as_f64(), Some(0.04));
        assert_eq!(pool["small_pity_guarantee"].as_u64(), Some(40));
        assert_eq!(pool["big_pity_cumulative"].as_u64(), Some(180));
        assert_eq!(pool["up_pity_soft"].as_u64(), Some(80));
        assert_eq!(pool["always_5_star"].as_bool(), Some(true));
        assert_ne!(pool["is_archived"].as_bool(), Some(true));
    }

    for id in [
        "char_up_20260605",
        "char_up_20260626",
        "weapon_up_20260605",
        "weapon_up_20260626",
    ] {
        assert_eq!(
            pool_by_id(&pools, id)["is_archived"].as_bool(),
            Some(true),
            "previous-version pool {id} must be archived"
        );
    }
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
