use crate::config::Config;
use colored::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Language {
    En,
    Cn,
}

impl Language {
    pub fn from_config(config: &Config) -> Self {
        if let Some(lang_str) = &config.language {
            let lower = lang_str.to_lowercase();
            if lower.contains("cn") || lower.contains("zh") {
                return Language::Cn;
            } else if lower.contains("en") {
                return Language::En;
            }
        }
        Self::from_env()
    }

    pub fn from_env() -> Self {
        if let Ok(lang) = std::env::var("LANG") {
            let lower = lang.to_lowercase();
            if lower.contains("cn") || lower.contains("zh") {
                return Language::Cn;
            }
        }
        Language::En
    }
}

pub struct I18n;

impl I18n {
    pub fn get(lang: Language, key: &str) -> String {
        match (lang, key) {
            // === Explainability Report ===
            (Language::En, "insight_header") => format!(
                "\n{}",
                "[Model Insight] Linear Manifold Analysis:".cyan().bold()
            ),
            (Language::Cn, "insight_header") => {
                format!("\n{}", "[模型洞察] 线性流形分析:".cyan().bold())
            }

            (Language::En, "feat_pity") => "Pity Progress (0-1)".to_string(),
            (Language::Cn, "feat_pity") => "保底进度 (0-1)".to_string(),

            (Language::En, "feat_total_norm") => "Total Pulls Norm".to_string(),
            (Language::Cn, "feat_total_norm") => "总抽数归一化".to_string(),

            (Language::En, "feat_env_noise") => "Env Noise".to_string(),
            (Language::Cn, "feat_env_noise") => "环境噪声".to_string(),

            (Language::En, "feat_loss_norm") => "Loss Streak Norm".to_string(),
            (Language::Cn, "feat_loss_norm") => "歪保底连胜归一化".to_string(),

            (Language::En, "feat_streak_4") => "4-Star Streak Norm".to_string(),
            (Language::Cn, "feat_streak_4") => "4星连抽归一化".to_string(),

            (Language::En, "feat_env_bias") => "Env Bias".to_string(),
            (Language::Cn, "feat_env_bias") => "环境偏差".to_string(),

            (Language::En, "feat_pity_loss") => "Pity * Loss (Interaction)".to_string(),
            (Language::Cn, "feat_pity_loss") => "保底 * 歪 (交互项)".to_string(),

            (Language::En, "feat_total_sq") => "Total Pulls (Quadratic)".to_string(),
            (Language::Cn, "feat_total_sq") => "总抽数 (二次方)".to_string(),

            (Language::En, "impact_neutral") => "Neutral".dimmed().to_string(),
            (Language::Cn, "impact_neutral") => "中性".dimmed().to_string(),

            (Language::En, "impact_boost") => "Boost Luck".green().to_string(),
            (Language::Cn, "impact_boost") => "提升运势".green().to_string(),

            (Language::En, "impact_reduce") => "Reduce Luck".red().to_string(),
            (Language::Cn, "impact_reduce") => "降低运势".red().to_string(),

            (Language::En, "impact_base") => "[Bias]".to_string(),
            (Language::Cn, "impact_base") => "[偏差]".to_string(),

            (Language::En, "lbl_base_bias") => "Base Bias".to_string(),
            (Language::Cn, "lbl_base_bias") => "基础偏差".to_string(),

            // === Interaction Mode ===
            (Language::En, "prompt_ppo") => format!(
                "{}",
                "[System] Use PPO (Transformer) Brain for simulation? (y/n): ".yellow()
            ),
            (Language::Cn, "prompt_ppo") => format!(
                "{}",
                "[系统] 是否使用 PPO (Transformer) 决策大脑进行模拟？(y/n): ".yellow()
            ),

            // === Header ===
            (Language::En, "header_title") => {
                format!("{}", "=== Talos-XII v0.4.0 by zayoka ===".purple().bold())
            }
            (Language::Cn, "header_title") => {
                format!("{}", "=== Talos-XII v0.4.0 by zayoka ===".purple().bold())
            }

            (Language::En, "header_pool") => "Pool Name: {}".to_string(),
            (Language::Cn, "header_pool") => "卡池名称: {}".to_string(),
            (Language::En, "header_pool_type") => "Pool Type: {}".to_string(),
            (Language::Cn, "header_pool_type") => "卡池类型: {}".to_string(),

            (Language::En, "header_up") => "UP Operator(s): {}".to_string(),
            (Language::Cn, "header_up") => "UP 干员: {}".to_string(),

            (Language::En, "header_prob") => {
                "Probabilities: 6-Star {:.1}% (Soft Pity start at {}, +5%/pull)".to_string()
            }
            (Language::Cn, "header_prob") => {
                "概率: 6星 {:.1}% (软保底从 {} 抽开始, 每抽 +5%)".to_string()
            }

            (Language::En, "header_rules") => {
                "Rules: {} Pulls Guarantee 6* (UP {}%, No Guarantee on Loss)".to_string()
            }
            (Language::Cn, "header_rules") => {
                "规则: {} 抽必出6星 (UP率 {}%, 歪了不保底)".to_string()
            }

            (Language::En, "header_up_pity") => "UP Pity: {} pulls guarantee UP".to_string(),
            (Language::Cn, "header_up_pity") => "UP 保底: {} 抽必出 UP".to_string(),
            (Language::En, "header_up_pity_off") => "UP Pity: Off".to_string(),
            (Language::Cn, "header_up_pity_off") => "UP 保底: 关闭".to_string(),
            (Language::En, "header_five_star_rule") => "5-Star Rule: {}".to_string(),
            (Language::Cn, "header_five_star_rule") => "五星规则: {}".to_string(),
            (Language::En, "rule_5star_every") => "Every pull guarantees 5-Star".to_string(),
            (Language::Cn, "rule_5star_every") => "每抽必出五星".to_string(),
            (Language::En, "rule_5star_pity") => "Guarantee at {} pulls".to_string(),
            (Language::Cn, "rule_5star_pity") => "{} 抽必出五星".to_string(),
            (Language::En, "rule_5star_off") => "No guarantee".to_string(),
            (Language::Cn, "rule_5star_off") => "无保底".to_string(),
            (Language::En, "label_yes") => "Yes".to_string(),
            (Language::Cn, "label_yes") => "是".to_string(),
            (Language::En, "label_no") => "No".to_string(),
            (Language::Cn, "label_no") => "否".to_string(),
            (Language::En, "header_big_pity_on") => {
                "Big Pity: Cumulative {} pulls guarantee UP (Require not UP: {})".to_string()
            }
            (Language::Cn, "header_big_pity_on") => {
                "大保底: 累计 {} 抽必出 UP (要求未出UP: {})".to_string()
            }
            (Language::En, "header_big_pity_off") => "Big Pity: Off".to_string(),
            (Language::Cn, "header_big_pity_off") => "大保底: 关闭".to_string(),

            (Language::En, "header_economy") => {
                "Economy: {} Jade/Pull | ~{} Free Pulls (Welfare)".to_string()
            }
            (Language::Cn, "header_economy") => {
                "经济: {} 合成玉/抽 | ~{} 免费抽 (福利)".to_string()
            }

            (Language::En, "header_neural") => {
                "Neural Core: Online (Evolved for Luck Balancing)".to_string()
            }
            (Language::Cn, "header_neural") => "神经核心: 在线 (进化以平衡运势)".to_string(),

            (Language::En, "sys_prng") => {
                format!("{}", "[System] PRNG Initialized: xoshiro256**".blue())
            }
            (Language::Cn, "sys_prng") => {
                format!("{}", "[系统] 伪随机数生成器已初始化: xoshiro256**".blue())
            }

            (Language::En, "sys_bench") => format!(
                "{}",
                "[System] Benchmarking simulation throughput...".blue()
            ),
            (Language::Cn, "sys_bench") => format!("{}", "[系统] 正在基准测试模拟吞吐量...".blue()),

            // === F2P Analysis ===
            (Language::En, "f2p_header") => format!(
                "\n{}",
                "=== F2P Welfare Analysis ({} Free Pulls) ==="
                    .purple()
                    .bold()
            ),
            (Language::Cn, "f2p_header") => {
                format!("\n{}", "=== F2P 福利分析 ({} 免费抽) ===".purple().bold())
            }

            (Language::En, "sys_run_prob") => format!(
                "{}",
                "[System] Running {} simulations for probability...".blue()
            ),
            (Language::Cn, "sys_run_prob") => {
                format!("{}", "[系统] 正在运行 {} 次模拟以计算概率...".blue())
            }

            (Language::En, "progress") => "Progress: {:>3}%".to_string(),
            (Language::Cn, "progress") => "进度: {:>3}%".to_string(),

            (Language::En, "expected_up") => "Expected UP Count: {:.2}".to_string(),
            (Language::Cn, "expected_up") => "期望 UP 数量: {:.2}".to_string(),

            (Language::En, "time_taken") => "Time taken: {:.2?}".to_string(),
            (Language::Cn, "time_taken") => "耗时: {:.2?}".to_string(),

            (Language::En, "throughput") => "Throughput: {:.0} sims/sec".to_string(),
            (Language::Cn, "throughput") => "吞吐量: {:.0} 模拟/秒".to_string(),

            (Language::En, "calc_cost") => {
                "\nCalculating average EXTRA cost for F2P players to get UP...".to_string()
            }
            (Language::Cn, "calc_cost") => {
                "\n正在计算 F2P 玩家获取 UP 的平均额外成本...".to_string()
            }

            (Language::En, "sys_run_cost") => format!(
                "{}",
                "[System] Running {} simulations for cost analysis...".blue()
            ),
            (Language::Cn, "sys_run_cost") => {
                format!("{}", "[系统] 正在运行 {} 次模拟以计算成本...".blue())
            }

            (Language::En, "total_value") => {
                "Total Value ~ 41000 Jade (Expected Cost for First UP)".to_string()
            }
            (Language::Cn, "total_value") => {
                "总价值 ~ 41000 合成玉 (首次获取 UP 的期望成本)".to_string()
            }

            // === Interactive Loop ===
            (Language::En, "prompt_pulls") => format!(
                "{}",
                "\nEnter pulls (default {}, 'h' for help, 'q' to quit): ".yellow()
            ),
            (Language::Cn, "prompt_pulls") => {
                format!("{}", "\n输入抽数 (默认 {}, 输入 'h' 查看指令, 'q' 退出): ".yellow())
            }

            (Language::En, "exit_msg") => "Exiting. Goodbye!".to_string(),
            (Language::Cn, "exit_msg") => "正在退出中。再见！".to_string(),

            (Language::En, "input_too_large") => {
                "Input too large, capped at 1,000,000 to prevent memory issues."
                    .red()
                    .to_string()
            }
            (Language::Cn, "input_too_large") => "输入过大，已限制为 1,000,000 以防止内存问题。"
                .red()
                .to_string(),

            (Language::En, "invalid_input") => "Invalid input. Please try again.".red().to_string(),
            (Language::Cn, "invalid_input") => "无效输入。请重试。".red().to_string(),

            (Language::En, "prompt_welfare") => format!(
                "{}",
                "Use Welfare Resources ({} pulls)? (y/n, default {}): ".yellow()
            ),
            (Language::Cn, "prompt_welfare") => {
                format!("{}", "使用福利资源 ({} 抽)? (y/n, 默认 {}): ".yellow())
            }

            (Language::En, "prompt_sim_count") => format!(
                "{}",
                "Enter simulation count (default {}, max 1M): ".yellow()
            ),
            (Language::Cn, "prompt_sim_count") => {
                format!("{}", "输入模拟次数 (默认 {}, 最大 1M): ".yellow())
            }
            (Language::En, "init_pool_list") => "Available pools:".to_string(),
            (Language::Cn, "init_pool_list") => "可用卡池:".to_string(),
            (Language::En, "init_pool_item") => "{}. {} ({})".to_string(),
            (Language::Cn, "init_pool_item") => "{}. {} ({})".to_string(),
            (Language::En, "pool_archived_tag") => " [Archived]".to_string(),
            (Language::Cn, "pool_archived_tag") => " [往期]".to_string(),
            (Language::En, "prompt_pool_select") => format!(
                "{}",
                "Select pool number(s) (e.g. 1 2 or 1,2, all; default {}): ".yellow()
            ),
            (Language::Cn, "prompt_pool_select") => format!(
                "{}",
                "选择卡池编号 (如 1 2 或 1,2, all，默认 {}): ".yellow()
            ),

            (Language::En, "sim_count_too_large") => {
                "Simulation count too large, capped at 1,000,000 to prevent CPU hang."
                    .red()
                    .to_string()
            }
            (Language::Cn, "sim_count_too_large") => {
                "模拟次数过大，已限制为 1,000,000 以防止 CPU 卡死。"
                    .red()
                    .to_string()
            }

            (Language::En, "sim_result_stats") => {
                "\n{} simulations of {}-pulls: Avg 6-Star {:.3} | UP {:.3}".to_string()
            }
            (Language::Cn, "sim_result_stats") => {
                "\n{} 次 {} 抽模拟: 平均 6星 {:.3} | UP {:.3}".to_string()
            }

            (Language::En, "single_sim_result") => {
                "\nSingle {}-pull result (Time: {:.2?}):".to_string()
            }
            (Language::Cn, "single_sim_result") => "\n单次 {} 抽结果 (耗时: {:.2?}):".to_string(),

            (Language::En, "cmd_help") => {
                "\nCommands:\n  h/help        Show commands\n  p <n>         Set default pulls\n  s <n>         Set default sims\n  w             Toggle welfare default\n  ppo           Toggle PPO\n  pool list     List pools\n  pool <id>     Switch pool\n  pool multi a,b|a b Use multiple pools\n  pool all      Use all pools\n  q             Quit\n(Enter a number to run a one-off pull count)\n"
                    .to_string()
            }
            (Language::Cn, "cmd_help") => {
                "\n指令:\n  h/help        查看指令\n  p <n>         设置默认抽数\n  s <n>         设置默认模拟次数\n  w             切换福利默认值\n  ppo           切换 PPO\n  pool list     查看卡池列表\n  pool <id>     切换卡池\n  pool multi a,b|a b 多卡池并行\n  pool all      全部卡池并行\n  q             退出\n(直接输入数字表示本次抽数)\n"
                    .to_string()
            }
            (Language::En, "cmd_ppo_on") => "PPO enabled for simulation.".to_string(),
            (Language::Cn, "cmd_ppo_on") => "已启用 PPO 模拟。".to_string(),
            (Language::En, "cmd_ppo_off") => "PPO disabled for simulation.".to_string(),
            (Language::Cn, "cmd_ppo_off") => "已关闭 PPO 模拟。".to_string(),
            (Language::En, "cmd_welfare_on") => "Welfare default set to ON.".to_string(),
            (Language::Cn, "cmd_welfare_on") => "福利默认值已设为开启。".to_string(),
            (Language::En, "cmd_welfare_off") => "Welfare default set to OFF.".to_string(),
            (Language::Cn, "cmd_welfare_off") => "福利默认值已设为关闭。".to_string(),
            (Language::En, "cmd_set_default_pulls") => "Default pulls set to {}.".to_string(),
            (Language::Cn, "cmd_set_default_pulls") => "默认抽数已设为 {}。".to_string(),
            (Language::En, "cmd_set_default_sims") => "Default sims set to {}.".to_string(),
            (Language::Cn, "cmd_set_default_sims") => "默认模拟次数已设为 {}。".to_string(),
            (Language::En, "cmd_pool_list") => "Pools: {}".to_string(),
            (Language::Cn, "cmd_pool_list") => "卡池列表: {}".to_string(),
            (Language::En, "cmd_pool_switched") => "Switched to pool: {}.".to_string(),
            (Language::Cn, "cmd_pool_switched") => "已切换到卡池: {}。".to_string(),
            (Language::En, "cmd_pool_not_found") => "Pool not found: {}.".to_string(),
            (Language::Cn, "cmd_pool_not_found") => "未找到卡池: {}。".to_string(),
            (Language::En, "cmd_pool_multi_set") => "Multi pools set: {}".to_string(),
            (Language::Cn, "cmd_pool_multi_set") => "多卡池已设置: {}。".to_string(),
            (Language::En, "cmd_pool_multi_empty") => "No valid pools selected.".to_string(),
            (Language::Cn, "cmd_pool_multi_empty") => "未选择有效卡池。".to_string(),
            (Language::En, "cmd_pool_all_set") => "All pools selected.".to_string(),
            (Language::Cn, "cmd_pool_all_set") => "已选择全部卡池。".to_string(),
            (Language::En, "sim_pool_header") => "\nPool: {}".to_string(),
            (Language::Cn, "sim_pool_header") => "\n卡池: {}".to_string(),
            (Language::En, "cmd_invalid_command") => "Invalid command. Type 'h' for help.".to_string(),
            (Language::Cn, "cmd_invalid_command") => "无效指令，输入 'h' 查看帮助。".to_string(),

            (Language::En, "single_stats") => "6-Star: {} | UP: {}".to_string(),
            (Language::Cn, "single_stats") => "6星: {} | UP: {}".to_string(),

            (Language::En, "unit_star") => "Star".to_string(),
            (Language::Cn, "unit_star") => "星".to_string(),

            (Language::En, "bench_fast") => {
                "[Bench] simulate_fast: {} sims of {} pulls in {:.2?} ({:.0} sims/sec)".to_string()
            }
            (Language::Cn, "bench_fast") => {
                "[基准] simulate_fast: {} 次模拟 {} 抽，耗时 {:.2?} ({:.0} 模拟/秒)".to_string()
            }

            (Language::En, "bench_one") => {
                "[Bench] simulate_one: {} sims of {} pulls in {:.2?} ({:.0} sims/sec)".to_string()
            }
            (Language::Cn, "bench_one") => {
                "[基准] simulate_one: {} 次模拟 {} 抽，耗时 {:.2?} ({:.0} 模拟/秒)".to_string()
            }

            (Language::En, "batch_sim_header") => "Simulations: {}, Pulls per sim: {}".to_string(),
            (Language::Cn, "batch_sim_header") => "模拟次数: {}, 每次抽数: {}".to_string(),

            (Language::En, "avg_6_star") => "Avg 6-Star: {:.4}".to_string(),
            (Language::Cn, "avg_6_star") => "平均 6星: {:.4}".to_string(),

            (Language::En, "avg_up") => "Avg UP: {:.4}".to_string(),
            (Language::Cn, "avg_up") => "平均 UP: {:.4}".to_string(),

            // Default fallback
            (_, k) => k.to_string(),
        }
    }
}
