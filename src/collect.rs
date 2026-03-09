//! Player data collection: record real player gacha pulls for Bayesian calibration.

#![allow(dead_code)] // Module API for future CLI integration (collect add/stats subcommands)

use crate::config::Config;
use crate::i18n::Language;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Write};

/// A single pull result from a real player.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerPullRecord {
    pub rarity: u8,
    pub is_up: bool,
}

/// One player's pull session on a specific pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerSession {
    pub player_id: String,
    pub pool_id: String,
    pub pulls: Vec<PlayerPullRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_jade_spent: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_pulls_used: Option<u32>,
    pub timestamp: String,
}

/// Persistent storage for all collected player data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlayerDatabase {
    pub sessions: Vec<PlayerSession>,
}

/// Per-pool aggregate statistics computed from collected data.
pub struct PoolEmpiricalStats {
    pub pool_id: String,
    pub pool_name: String,
    pub total_pulls: usize,
    pub total_six_star: usize,
    pub total_up: usize,
    pub session_count: usize,
    pub pity_hits: Vec<usize>,
    pub observed_base_rate: f64,
    pub observed_up_rate: f64,
    pub avg_pulls_per_six: f64,
}

impl PlayerDatabase {
    pub fn load(path: &str) -> Self {
        if let Ok(data) = std::fs::read_to_string(path) {
            match serde_json::from_str(&data) {
                Ok(db) => return db,
                Err(e) => log::warn!("[Collect] Failed to parse {}: {}", path, e),
            }
        }
        let alt = format!("../../{}", path);
        if let Ok(data) = std::fs::read_to_string(&alt) {
            match serde_json::from_str(&data) {
                Ok(db) => return db,
                Err(e) => log::warn!("[Collect] Failed to parse {}: {}", alt, e),
            }
        }
        PlayerDatabase::default()
    }

    pub fn save(&self, path: &str) -> bool {
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(self) {
            Ok(json) => std::fs::write(path, json).is_ok(),
            Err(_) => false,
        }
    }

    pub fn add_session(&mut self, session: PlayerSession) {
        self.sessions.push(session);
    }

    pub fn total_pulls(&self) -> usize {
        self.sessions.iter().map(|s| s.pulls.len()).sum()
    }

    pub fn player_count(&self) -> usize {
        let mut ids = std::collections::HashSet::new();
        for s in &self.sessions {
            ids.insert(&s.player_id);
        }
        ids.len()
    }

    /// Compute per-pool empirical statistics.
    pub fn compute_pool_stats(&self, config: &Config) -> Vec<PoolEmpiricalStats> {
        let mut pool_map: HashMap<String, Vec<&PlayerSession>> = HashMap::new();
        for session in &self.sessions {
            pool_map
                .entry(session.pool_id.clone())
                .or_default()
                .push(session);
        }

        let mut stats_list = Vec::new();
        for (pool_id, sessions) in &pool_map {
            let pool_name = config
                .pools
                .iter()
                .find(|p| &p.id == pool_id)
                .map(|p| p.name.clone())
                .unwrap_or_else(|| pool_id.clone());

            let pool_cfg = config.pools.iter().find(|p| &p.id == pool_id);
            let guarantee = pool_cfg.map(|p| p.small_pity_guarantee).unwrap_or(80);

            let mut total_pulls = 0usize;
            let mut total_six = 0usize;
            let mut total_up = 0usize;
            let mut pity_hits = vec![0usize; guarantee + 1];

            for session in sessions {
                total_pulls += session.pulls.len();
                let mut pity = 0usize;
                for pull in &session.pulls {
                    pity += 1;
                    if pull.rarity == 6 {
                        total_six += 1;
                        if pull.is_up {
                            total_up += 1;
                        }
                        if pity <= guarantee {
                            pity_hits[pity] += 1;
                        }
                        pity = 0;
                    }
                }
            }

            let observed_base_rate = if total_pulls > 0 {
                total_six as f64 / total_pulls as f64
            } else {
                0.0
            };
            let observed_up_rate = if total_six > 0 {
                total_up as f64 / total_six as f64
            } else {
                0.0
            };
            let avg_pulls_per_six = if total_six > 0 {
                total_pulls as f64 / total_six as f64
            } else {
                0.0
            };

            stats_list.push(PoolEmpiricalStats {
                pool_id: pool_id.clone(),
                pool_name,
                total_pulls,
                total_six_star: total_six,
                total_up,
                session_count: sessions.len(),
                pity_hits,
                observed_base_rate,
                observed_up_rate,
                avg_pulls_per_six,
            });
        }
        stats_list.sort_by(|a, b| b.total_pulls.cmp(&a.total_pulls));
        stats_list
    }
}

/// Reconstruct a full pull sequence from simplified "6-star positions" input.
///
/// Input: comma-separated positions where 6-stars appeared (e.g. "78,145,200").
/// The positions are 1-indexed absolute pull numbers.
/// Fills in 4-star pulls between 6-star positions.
///
/// `up_flags`: for each 6-star position, whether it was UP (e.g. "y,n,y").
pub fn reconstruct_from_six_star_positions(
    positions_str: &str,
    up_flags_str: &str,
    total_pulls: usize,
) -> Vec<PlayerPullRecord> {
    let raw_positions: Vec<usize> = positions_str
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    let raw_up_flags: Vec<bool> = up_flags_str
        .split(',')
        .map(|s| {
            let t = s.trim().to_lowercase();
            t == "y" || t == "yes" || t == "1" || t == "true"
        })
        .collect();

    let mut paired: Vec<(usize, bool)> = raw_positions
        .into_iter()
        .zip(raw_up_flags.into_iter().chain(std::iter::repeat(false)))
        .collect();
    paired.sort_unstable_by_key(|&(pos, _)| pos);

    let positions: Vec<usize> = paired.iter().map(|&(p, _)| p).collect();
    let up_flags: Vec<bool> = paired.iter().map(|&(_, u)| u).collect();

    let mut pulls = Vec::with_capacity(total_pulls);
    let mut six_idx = 0;

    for i in 1..=total_pulls {
        if six_idx < positions.len() && i == positions[six_idx] {
            let is_up = up_flags.get(six_idx).copied().unwrap_or(false);
            pulls.push(PlayerPullRecord { rarity: 6, is_up });
            six_idx += 1;
        } else {
            pulls.push(PlayerPullRecord {
                rarity: 4,
                is_up: false,
            });
        }
    }
    pulls
}

fn read_line_trimmed() -> String {
    let mut buf = String::new();
    let _ = io::stdin().read_line(&mut buf);
    buf.trim().to_string()
}

fn prompt(msg: &str) -> String {
    print!("{}", msg);
    let _ = io::stdout().flush();
    read_line_trimmed()
}

/// Interactive session for adding player data.
pub fn add_session_interactive(config: &Config, lang: Language) -> Option<PlayerSession> {
    let cn = lang == Language::Cn;

    if config.pools.is_empty() {
        println!(
            "{}",
            if cn {
                "[数据采集] 没有可用的池子配置。"
            } else {
                "[Collect] No pool configurations available."
            }
        );
        return None;
    }

    println!(
        "{}",
        if cn {
            "\n═══ 数据采集: 录入玩家抽卡记录 ═══"
        } else {
            "\n═══ Data Collection: Record Player Pulls ═══"
        }
    );

    println!(
        "{}",
        if cn {
            "请选择池子:"
        } else {
            "Select pool:"
        }
    );
    for (i, pool) in config.pools.iter().enumerate() {
        println!("  {}. {} ({})", i + 1, pool.name, pool.pool_type);
    }
    let pool_input = prompt("> ");
    let pool_idx = pool_input.parse::<usize>().unwrap_or(1).saturating_sub(1);
    if pool_idx >= config.pools.len() {
        println!(
            "{}",
            if cn {
                "无效选择。"
            } else {
                "Invalid selection."
            }
        );
        return None;
    }
    let pool = &config.pools[pool_idx];

    let player_id = prompt(if cn {
        "玩家ID (匿名标识): "
    } else {
        "Player ID (anonymous label): "
    });
    let player_id = if player_id.is_empty() {
        format!("player_{}", chrono_timestamp())
    } else {
        player_id
    };

    println!(
        "{}",
        if cn {
            "录入模式:\n  1. 逐抽录入 (格式: 星级,是否UP  例: 4,n / 6,y)\n  2. 简化录入 (只输入6★出现位置)"
        } else {
            "Input mode:\n  1. Full (format: rarity,isUP  e.g. 4,n / 6,y)\n  2. Simplified (only 6-star positions)"
        }
    );
    let mode = prompt(if cn {
        "选择模式 [1/2]: "
    } else {
        "Select mode [1/2]: "
    });

    let pulls = if mode.trim() == "2" {
        let total_str = prompt(if cn { "总抽数: " } else { "Total pulls: " });
        let total: usize = total_str.parse().unwrap_or(0);
        if total == 0 {
            println!(
                "{}",
                if cn {
                    "无效输入。"
                } else {
                    "Invalid input."
                }
            );
            return None;
        }

        let positions_str = prompt(if cn {
            "6★出现位置 (逗号分隔, 如 78,145): "
        } else {
            "6-star positions (comma-separated, e.g. 78,145): "
        });
        let up_str = prompt(if cn {
            "每个6★是否UP (y/n逗号分隔, 如 y,n): "
        } else {
            "UP for each 6-star (y/n comma-separated, e.g. y,n): "
        });
        reconstruct_from_six_star_positions(&positions_str, &up_str, total)
    } else {
        let mut pulls = Vec::new();
        println!(
            "{}",
            if cn {
                "逐抽录入 (输入 done 结束, undo 撤销):"
            } else {
                "Enter pulls (type 'done' to finish, 'undo' to undo):"
            }
        );
        loop {
            let input = prompt(&format!(
                "  {} #{} > ",
                if cn { "抽" } else { "Pull" },
                pulls.len() + 1
            ));
            if input.eq_ignore_ascii_case("done") {
                break;
            }
            if input.eq_ignore_ascii_case("undo") {
                if pulls.pop().is_some() {
                    println!("  ↩ {}", if cn { "已撤销" } else { "Undone" });
                }
                continue;
            }
            let parts: Vec<&str> = input.split(',').map(|s| s.trim()).collect();
            if parts.len() < 2 {
                println!(
                    "  {}",
                    if cn {
                        "格式: 星级,是否UP (如 4,n / 6,y)"
                    } else {
                        "Format: rarity,isUP (e.g. 4,n / 6,y)"
                    }
                );
                continue;
            }
            let rarity: u8 = match parts[0].parse() {
                Ok(r) if r == 4 || r == 5 || r == 6 => r,
                _ => {
                    println!(
                        "  {}",
                        if cn {
                            "星级必须为 4/5/6"
                        } else {
                            "Rarity must be 4/5/6"
                        }
                    );
                    continue;
                }
            };
            let is_up = parts[1].eq_ignore_ascii_case("y")
                || parts[1].eq_ignore_ascii_case("yes")
                || parts[1] == "1";
            pulls.push(PlayerPullRecord { rarity, is_up });
        }
        pulls
    };

    if pulls.is_empty() {
        println!(
            "{}",
            if cn {
                "没有录入任何数据。"
            } else {
                "No data entered."
            }
        );
        return None;
    }

    let jade_input = prompt(if cn {
        "总花费源石 (可选, 回车跳过): "
    } else {
        "Total jade spent (optional, Enter to skip): "
    });
    let total_jade_spent = jade_input.parse::<u32>().ok();

    let free_input = prompt(if cn {
        "使用免费抽数 (可选, 回车跳过): "
    } else {
        "Free pulls used (optional, Enter to skip): "
    });
    let free_pulls_used = free_input.parse::<u32>().ok();

    let six_count = pulls.iter().filter(|p| p.rarity == 6).count();
    let up_count = pulls.iter().filter(|p| p.is_up).count();

    println!(
        "\n✓ {}: {} {}, {} {}, {} 6★ ({} UP)",
        if cn { "已录入" } else { "Recorded" },
        player_id,
        pool.name,
        pulls.len(),
        if cn { "抽" } else { "pulls" },
        six_count,
        up_count,
    );

    Some(PlayerSession {
        player_id,
        pool_id: pool.id.clone(),
        pulls,
        total_jade_spent,
        free_pulls_used,
        timestamp: chrono_timestamp(),
    })
}

/// Import sessions from a JSON file.
pub fn import_from_json(path: &str) -> Result<Vec<PlayerSession>, String> {
    let data =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    let sessions: Vec<PlayerSession> =
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse JSON: {}", e))?;
    Ok(sessions)
}

/// Print collected data statistics.
pub fn print_stats(db: &PlayerDatabase, config: &Config, lang: Language) {
    let cn = lang == Language::Cn;

    println!(
        "\n{}",
        if cn {
            "═══ 已采集数据统计 ═══"
        } else {
            "═══ Collected Data Statistics ═══"
        }
    );

    println!(
        "  {}: {}    {}: {}    {}: {}",
        if cn { "会话数" } else { "Sessions" },
        db.sessions.len(),
        if cn { "总抽数" } else { "Total pulls" },
        db.total_pulls(),
        if cn { "玩家数" } else { "Players" },
        db.player_count(),
    );

    if db.sessions.is_empty() {
        println!(
            "\n  {}",
            if cn {
                "暂无数据。使用 collect add 录入玩家数据。"
            } else {
                "No data yet. Use 'collect add' to record player data."
            }
        );
        return;
    }

    let pool_stats = db.compute_pool_stats(config);

    println!(
        "\n  {:<24} {:>8} {:>6} {:>6} {:>10} {:>10}",
        if cn { "池子" } else { "Pool" },
        if cn { "样本抽" } else { "Pulls" },
        if cn { "6★数" } else { "6★" },
        if cn { "UP数" } else { "UP" },
        if cn { "实测6★率" } else { "6★ Rate" },
        if cn { "实测UP率" } else { "UP Rate" },
    );
    println!("  {}", "-".repeat(70));

    for ps in &pool_stats {
        println!(
            "  {:<24} {:>8} {:>6} {:>6} {:>9.3}% {:>9.2}%",
            ps.pool_name,
            ps.total_pulls,
            ps.total_six_star,
            ps.total_up,
            ps.observed_base_rate * 100.0,
            ps.observed_up_rate * 100.0,
        );
    }

    for ps in &pool_stats {
        if ps.total_six_star == 0 {
            continue;
        }
        let pool_cfg = config.pools.iter().find(|p| p.id == ps.pool_id);
        let soft_start = pool_cfg.map(|p| p.soft_pity_start).unwrap_or(65);
        let guarantee = pool_cfg.map(|p| p.small_pity_guarantee).unwrap_or(80);

        let pre_soft: usize = ps.pity_hits.iter().take(soft_start).sum();
        let in_soft: usize = ps
            .pity_hits
            .iter()
            .skip(soft_start)
            .take(guarantee - soft_start)
            .sum();
        let at_guarantee = if guarantee < ps.pity_hits.len() {
            ps.pity_hits[guarantee]
        } else {
            0
        };

        println!(
            "\n  {} ({}):",
            if cn {
                "Pity 分布"
            } else {
                "Pity distribution"
            },
            ps.pool_name
        );
        println!(
            "    0-{} {}: {} {}",
            soft_start - 1,
            if cn { "抽出6★" } else { "pulls → 6★" },
            pre_soft,
            if cn {
                "次 (基础概率区间)"
            } else {
                "(base rate range)"
            },
        );
        println!(
            "    {}-{} {}: {} {}",
            soft_start,
            guarantee - 1,
            if cn { "抽出6★" } else { "pulls → 6★" },
            in_soft,
            if cn {
                "次 (软保底区间)"
            } else {
                "(soft pity range)"
            },
        );
        println!(
            "    {} {}: {} {}",
            guarantee,
            if cn { "抽出6★" } else { "pulls → 6★" },
            at_guarantee,
            if cn { "次 (硬保底)" } else { "(hard pity)" },
        );
    }

    println!();
}

fn chrono_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_basic() {
        let pulls = reconstruct_from_six_star_positions("3,7", "y,n", 10);
        assert_eq!(pulls.len(), 10);
        assert_eq!(pulls[2].rarity, 6);
        assert!(pulls[2].is_up);
        assert_eq!(pulls[6].rarity, 6);
        assert!(!pulls[6].is_up);
        assert_eq!(pulls[0].rarity, 4);
    }

    #[test]
    fn reconstruct_empty_positions() {
        let pulls = reconstruct_from_six_star_positions("", "", 5);
        assert_eq!(pulls.len(), 5);
        assert!(pulls.iter().all(|p| p.rarity == 4));
    }

    #[test]
    fn database_round_trip() {
        let mut db = PlayerDatabase::default();
        db.add_session(PlayerSession {
            player_id: "test".to_string(),
            pool_id: "pool_a".to_string(),
            pulls: vec![
                PlayerPullRecord {
                    rarity: 4,
                    is_up: false,
                },
                PlayerPullRecord {
                    rarity: 6,
                    is_up: true,
                },
            ],
            total_jade_spent: Some(1000),
            free_pulls_used: None,
            timestamp: "0".to_string(),
        });
        let json = serde_json::to_string(&db).unwrap();
        let db2: PlayerDatabase = serde_json::from_str(&json).unwrap();
        assert_eq!(db2.sessions.len(), 1);
        assert_eq!(db2.sessions[0].pulls.len(), 2);
    }
}
