use crate::config::Config;
use colored::*;

/// Supported UI languages.
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

/// Internationalization provider for UI strings.
pub struct I18n;

impl I18n {
    /// Get a localized string by key for the given language.
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
                format!("{}", format!("=== Talos-XII v{} by zayoka ===", env!("CARGO_PKG_VERSION")).purple().bold())
            }
            (Language::Cn, "header_title") => {
                format!("{}", format!("=== Talos-XII v{} by zayoka ===", env!("CARGO_PKG_VERSION")).purple().bold())
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
                "See above for expected cost analysis based on current pool configuration.".to_string()
            }
            (Language::Cn, "total_value") => {
                "以上为基于当前池配置的期望成本分析。".to_string()
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

            (Language::En, "cmd_help") => format!(
                "\n{}\n  {}\n    {}          {}\n    {}        {}\n    {}        {}\n    {}              {}\n    {}            {}\n  {}\n    {}    {}\n    {}      {}\n    {}      {}\n  {}\n    {}         {}\n    {}           {}\n    {}       {}\n  {}\n    {}         {}\n    {}              {}\n{}\n",
                "═══ Commands ═══".cyan().bold(),
                "[Simulation]".yellow().bold(),
                "<number>".white().bold(), "Pull count for this run",
                "p <n>".white().bold(), "Set default pulls",
                "s <n>".white().bold(), "Set default sims",
                "w".white().bold(), "Toggle welfare",
                "ppo".white().bold(), "Toggle PPO brain",
                "[Pool]".yellow().bold(),
                "pool list".white().bold(), "List pools",
                "pool <id>".white().bold(), "Switch pool",
                "pool all".white().bold(), "Use all pools",
                "[Info]".yellow().bold(),
                "status".white().bold(), "Current state",
                "info".white().bold(), "Pool details",
                "history".white().bold(), "Sim history",
                "[Other]".yellow().bold(),
                "h/help".white().bold(), "Show this help",
                "q".white().bold(), "Quit",
                "═════════════════".cyan(),
            ),
            (Language::Cn, "cmd_help") => format!(
                "\n{}\n  {}\n    {}          {}\n    {}        {}\n    {}        {}\n    {}              {}\n    {}            {}\n  {}\n    {}    {}\n    {}      {}\n    {}      {}\n  {}\n    {}         {}\n    {}           {}\n    {}       {}\n  {}\n    {}         {}\n    {}              {}\n{}\n",
                "═══ 指令帮助 ═══".cyan().bold(),
                "[模拟]".yellow().bold(),
                "<数字>".white().bold(), "本次抽数",
                "p <n>".white().bold(), "设置默认抽数",
                "s <n>".white().bold(), "设置默认模拟次数",
                "w".white().bold(), "切换福利开关",
                "ppo".white().bold(), "切换 PPO 大脑",
                "[卡池]".yellow().bold(),
                "pool list".white().bold(), "查看卡池列表",
                "pool <id>".white().bold(), "切换卡池",
                "pool all".white().bold(), "全部卡池并行",
                "[信息]".yellow().bold(),
                "status".white().bold(), "查看当前状态",
                "info".white().bold(), "查看卡池详情",
                "history".white().bold(), "查看模拟历史",
                "[其他]".yellow().bold(),
                "h/help".white().bold(), "显示此帮助",
                "q".white().bold(), "退出",
                "═════════════════".cyan(),
            ),
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

            // === Pull Result Detail ===
            (Language::En, "pull_list_omitted") => "... ({} more omitted)".dimmed().to_string(),
            (Language::Cn, "pull_list_omitted") => "... (还有 {} 条省略)".dimmed().to_string(),

            // === Consumption Block ===
            (Language::En, "consumption_header") => {
                format!("{}", "═══ Consumption ═══".cyan())
            }
            (Language::Cn, "consumption_header") => {
                format!("{}", "═══ 消耗详情 ═══".cyan())
            }
            (Language::En, "consumption_free") => "  Free Pulls Used: {}".to_string(),
            (Language::Cn, "consumption_free") => "  使用免费抽: {}".to_string(),
            (Language::En, "consumption_jade") => "  Jade Spent: {} ({} pulls)".to_string(),
            (Language::Cn, "consumption_jade") => "  合成玉消耗: {} ({} 抽)".to_string(),
            (Language::En, "big_pity_triggered") => {
                format!("  {}", "Big Pity Triggered!".yellow().bold())
            }
            (Language::Cn, "big_pity_triggered") => {
                format!("  {}", "大保底已触发！".yellow().bold())
            }
            (Language::En, "consumption_footer") => {
                format!("{}", "════════════════════".cyan())
            }
            (Language::Cn, "consumption_footer") => {
                format!("{}", "════════════════════".cyan())
            }

            // === Prompt with Status ===
            (Language::En, "prompt_pulls_status") => format!(
                "{}",
                "\n[{} pulls | {} sims | welfare:{}] Enter pulls (h help, q quit): ".yellow()
            ),
            (Language::Cn, "prompt_pulls_status") => format!(
                "{}",
                "\n[{}抽 | {}次模拟 | 福利:{}] 输入抽数 (h 帮助, q 退出): ".yellow()
            ),
            (Language::En, "label_on") => "ON".to_string(),
            (Language::Cn, "label_on") => "开".to_string(),
            (Language::En, "label_off") => "OFF".to_string(),
            (Language::Cn, "label_off") => "关".to_string(),

            // === Status Command ===
            (Language::En, "cmd_status_header") => {
                format!("{}", "═══ Current State ═══".cyan().bold())
            }
            (Language::Cn, "cmd_status_header") => {
                format!("{}", "═══ 当前状态 ═══".cyan().bold())
            }
            (Language::En, "cmd_status_pool") => "  Pool: {}".to_string(),
            (Language::Cn, "cmd_status_pool") => "  卡池: {}".to_string(),
            (Language::En, "cmd_status_pulls") => "  Default Pulls: {}".to_string(),
            (Language::Cn, "cmd_status_pulls") => "  默认抽数: {}".to_string(),
            (Language::En, "cmd_status_sims") => "  Default Sims: {}".to_string(),
            (Language::Cn, "cmd_status_sims") => "  默认模拟次数: {}".to_string(),
            (Language::En, "cmd_status_welfare") => "  Welfare: {}".to_string(),
            (Language::Cn, "cmd_status_welfare") => "  福利: {}".to_string(),
            (Language::En, "cmd_status_ppo") => "  PPO Brain: {}".to_string(),
            (Language::Cn, "cmd_status_ppo") => "  PPO 大脑: {}".to_string(),
            (Language::En, "cmd_status_footer") => {
                format!("{}", "══════════════════════".cyan())
            }
            (Language::Cn, "cmd_status_footer") => {
                format!("{}", "══════════════════════".cyan())
            }

            // === History Command ===
            (Language::En, "cmd_history_header") => {
                format!("{}", "═══ Simulation History ═══".cyan().bold())
            }
            (Language::Cn, "cmd_history_header") => {
                format!("{}", "═══ 模拟历史 ═══".cyan().bold())
            }
            (Language::En, "cmd_history_item") => {
                "  #{}: {} | {} pulls x{} | Avg 6*: {:.3} UP: {:.3} | {}ms".to_string()
            }
            (Language::Cn, "cmd_history_item") => {
                "  #{}: {} | {} 抽 x{} | 均6星: {:.3} UP: {:.3} | {}ms".to_string()
            }
            (Language::En, "cmd_history_empty") => "  No simulation history yet.".dimmed().to_string(),
            (Language::Cn, "cmd_history_empty") => "  暂无模拟历史。".dimmed().to_string(),
            (Language::En, "cmd_history_footer") => {
                format!("{}", "═══════════════════════════".cyan())
            }
            (Language::Cn, "cmd_history_footer") => {
                format!("{}", "═══════════════════════════".cyan())
            }

            // === Calibration ===
            (Language::En, "cal_header") => "\n═══ Bayesian Calibration Analysis ═══".to_string(),
            (Language::Cn, "cal_header") => "\n═══ 贝叶斯校准分析 ═══".to_string(),
            (Language::En, "cal_total_samples") => "Total samples".to_string(),
            (Language::Cn, "cal_total_samples") => "总样本".to_string(),
            (Language::En, "cal_sessions") => "Sessions".to_string(),
            (Language::Cn, "cal_sessions") => "会话数".to_string(),
            (Language::En, "cal_unit_pulls") => "pulls".to_string(),
            (Language::Cn, "cal_unit_pulls") => "抽".to_string(),
            (Language::En, "cal_insufficient") => "Insufficient data, showing stats only".to_string(),
            (Language::Cn, "cal_insufficient") => "数据量不足，仅显示统计信息".to_string(),
            (Language::En, "cal_col_param") => "Parameter".to_string(),
            (Language::Cn, "cal_col_param") => "参数".to_string(),
            (Language::En, "cal_col_official") => "Official".to_string(),
            (Language::Cn, "cal_col_official") => "公示值".to_string(),
            (Language::En, "cal_col_calibrated") => "Calibrated".to_string(),
            (Language::Cn, "cal_col_calibrated") => "校准值".to_string(),
            (Language::En, "cal_col_ci") => "95% CI".to_string(),
            (Language::Cn, "cal_col_ci") => "95% 置信区间".to_string(),
            (Language::En, "cal_col_sig") => "Sig.".to_string(),
            (Language::Cn, "cal_col_sig") => "差异".to_string(),
            (Language::En, "cal_base_rate") => "Base 6★ rate".to_string(),
            (Language::Cn, "cal_base_rate") => "基础6★概率".to_string(),
            (Language::En, "cal_slope") => "Soft pity slope".to_string(),
            (Language::Cn, "cal_slope") => "软保底斜率".to_string(),
            (Language::En, "cal_up_rate") => "UP rate".to_string(),
            (Language::Cn, "cal_up_rate") => "UP 率".to_string(),
            (Language::En, "cal_no_data") => "No data to calibrate. Use 'collect add' to record player data first.".to_string(),
            (Language::Cn, "cal_no_data") => "无可校准数据。请先使用 collect add 录入玩家数据。".to_string(),
            (Language::En, "cal_sample_hint") => "Tip: UP rate CI still wide. Need".to_string(),
            (Language::Cn, "cal_sample_hint") => "提示: 当前UP率置信区间较宽。还需".to_string(),
            (Language::En, "cal_sample_hint_more") => "more".to_string(),
            (Language::Cn, "cal_sample_hint_more") => "个".to_string(),
            (Language::En, "cal_sample_hint_narrow") => "samples to narrow to".to_string(),
            (Language::Cn, "cal_sample_hint_narrow") => "样本可将区间收窄到".to_string(),
            (Language::En, "cal_sample_hint_acc") => "accuracy".to_string(),
            (Language::Cn, "cal_sample_hint_acc") => "精度".to_string(),
            (Language::En, "cal_sig_yes") => "Yes".to_string(),
            (Language::Cn, "cal_sig_yes") => "是".to_string(),
            (Language::En, "cal_sig_no") => "No".to_string(),
            (Language::Cn, "cal_sig_no") => "否".to_string(),

            // === Data Collection ===
            (Language::En, "collect_no_pools") => "[Collect] No pool configurations available.".to_string(),
            (Language::Cn, "collect_no_pools") => "[数据采集] 没有可用的池子配置。".to_string(),
            (Language::En, "collect_header") => "\n═══ Data Collection: Record Player Pulls ═══".to_string(),
            (Language::Cn, "collect_header") => "\n═══ 数据采集: 录入玩家抽卡记录 ═══".to_string(),
            (Language::En, "collect_select_pool") => "Select pool:".to_string(),
            (Language::Cn, "collect_select_pool") => "请选择池子:".to_string(),
            (Language::En, "collect_invalid_selection") => "Invalid selection.".to_string(),
            (Language::Cn, "collect_invalid_selection") => "无效选择。".to_string(),
            (Language::En, "collect_player_id") => "Player ID (anonymous label): ".to_string(),
            (Language::Cn, "collect_player_id") => "玩家ID (匿名标识): ".to_string(),
            (Language::En, "collect_input_mode") => "Input mode:\n  1. Full (format: rarity,isUP  e.g. 4,n / 6,y)\n  2. Simplified (only 6-star positions)".to_string(),
            (Language::Cn, "collect_input_mode") => "录入模式:\n  1. 逐抽录入 (格式: 星级,是否UP  例: 4,n / 6,y)\n  2. 简化录入 (只输入6★出现位置)".to_string(),
            (Language::En, "collect_select_mode") => "Select mode [1/2]: ".to_string(),
            (Language::Cn, "collect_select_mode") => "选择模式 [1/2]: ".to_string(),
            (Language::En, "collect_total_pulls") => "Total pulls: ".to_string(),
            (Language::Cn, "collect_total_pulls") => "总抽数: ".to_string(),
            (Language::En, "collect_invalid_input") => "Invalid input.".to_string(),
            (Language::Cn, "collect_invalid_input") => "无效输入。".to_string(),
            (Language::En, "collect_six_positions") => "6-star positions (comma-separated, e.g. 78,145): ".to_string(),
            (Language::Cn, "collect_six_positions") => "6★出现位置 (逗号分隔, 如 78,145): ".to_string(),
            (Language::En, "collect_up_flags") => "UP for each 6-star (y/n comma-separated, e.g. y,n): ".to_string(),
            (Language::Cn, "collect_up_flags") => "每个6★是否UP (y/n逗号分隔, 如 y,n): ".to_string(),
            (Language::En, "collect_full_mode_hint") => "Enter pulls (type 'done' to finish, 'undo' to undo):".to_string(),
            (Language::Cn, "collect_full_mode_hint") => "逐抽录入 (输入 done 结束, undo 撤销):".to_string(),
            (Language::En, "collect_pull_prompt") => "Pull".to_string(),
            (Language::Cn, "collect_pull_prompt") => "抽".to_string(),
            (Language::En, "collect_undone") => "Undone".to_string(),
            (Language::Cn, "collect_undone") => "已撤销".to_string(),
            (Language::En, "collect_format_hint") => "Format: rarity,isUP (e.g. 4,n / 6,y)".to_string(),
            (Language::Cn, "collect_format_hint") => "格式: 星级,是否UP (如 4,n / 6,y)".to_string(),
            (Language::En, "collect_rarity_hint") => "Rarity must be 4/5/6".to_string(),
            (Language::Cn, "collect_rarity_hint") => "星级必须为 4/5/6".to_string(),
            (Language::En, "collect_no_data") => "No data entered.".to_string(),
            (Language::Cn, "collect_no_data") => "没有录入任何数据。".to_string(),
            (Language::En, "collect_jade_prompt") => "Total jade spent (optional, Enter to skip): ".to_string(),
            (Language::Cn, "collect_jade_prompt") => "总花费源石 (可选, 回车跳过): ".to_string(),
            (Language::En, "collect_free_prompt") => "Free pulls used (optional, Enter to skip): ".to_string(),
            (Language::Cn, "collect_free_prompt") => "使用免费抽数 (可选, 回车跳过): ".to_string(),
            (Language::En, "collect_recorded") => "Recorded".to_string(),
            (Language::Cn, "collect_recorded") => "已录入".to_string(),
            (Language::En, "collect_unit_pulls") => "pulls".to_string(),
            (Language::Cn, "collect_unit_pulls") => "抽".to_string(),
            (Language::En, "stats_header") => "\n═══ Collected Data Statistics ═══".to_string(),
            (Language::Cn, "stats_header") => "\n═══ 已采集数据统计 ═══".to_string(),
            (Language::En, "stats_sessions") => "Sessions".to_string(),
            (Language::Cn, "stats_sessions") => "会话数".to_string(),
            (Language::En, "stats_total_pulls") => "Total pulls".to_string(),
            (Language::Cn, "stats_total_pulls") => "总抽数".to_string(),
            (Language::En, "stats_players") => "Players".to_string(),
            (Language::Cn, "stats_players") => "玩家数".to_string(),
            (Language::En, "stats_no_data") => "No data yet. Use 'collect add' to record player data.".to_string(),
            (Language::Cn, "stats_no_data") => "暂无数据。使用 collect add 录入玩家数据。".to_string(),
            (Language::En, "stats_col_pool") => "Pool".to_string(),
            (Language::Cn, "stats_col_pool") => "池子".to_string(),
            (Language::En, "stats_col_pulls") => "Pulls".to_string(),
            (Language::Cn, "stats_col_pulls") => "样本抽".to_string(),
            (Language::En, "stats_col_six") => "6★".to_string(),
            (Language::Cn, "stats_col_six") => "6★数".to_string(),
            (Language::En, "stats_col_up") => "UP".to_string(),
            (Language::Cn, "stats_col_up") => "UP数".to_string(),
            (Language::En, "stats_col_six_rate") => "6★ Rate".to_string(),
            (Language::Cn, "stats_col_six_rate") => "实测6★率".to_string(),
            (Language::En, "stats_col_up_rate") => "UP Rate".to_string(),
            (Language::Cn, "stats_col_up_rate") => "实测UP率".to_string(),
            (Language::En, "stats_pity_dist") => "Pity distribution".to_string(),
            (Language::Cn, "stats_pity_dist") => "Pity 分布".to_string(),
            (Language::En, "stats_pity_six") => "pulls → 6★".to_string(),
            (Language::Cn, "stats_pity_six") => "抽出6★".to_string(),
            (Language::En, "stats_base_range") => "(base rate range)".to_string(),
            (Language::Cn, "stats_base_range") => "次 (基础概率区间)".to_string(),
            (Language::En, "stats_soft_range") => "(soft pity range)".to_string(),
            (Language::Cn, "stats_soft_range") => "次 (软保底区间)".to_string(),
            (Language::En, "stats_hard_pity") => "(hard pity)".to_string(),
            (Language::Cn, "stats_hard_pity") => "次 (硬保底)".to_string(),

            // Default fallback
            (_, k) => k.to_string(),
        }
    }
}
