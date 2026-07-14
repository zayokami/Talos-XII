use plotters::prelude::*;

#[derive(Clone, Debug)]
pub enum ChartFormat {
    Svg,
    Png,
}

const PALETTE: &[RGBColor] = &[
    RGBColor(31, 119, 180),
    RGBColor(255, 127, 14),
    RGBColor(44, 160, 44),
    RGBColor(214, 39, 40),
    RGBColor(148, 103, 189),
    RGBColor(140, 86, 75),
    RGBColor(227, 119, 194),
    RGBColor(127, 127, 127),
];

pub type CiPoint = (f64, f64, f64, f64);
pub type CiSeries<'a> = (&'a str, &'a [CiPoint]);

fn color(idx: usize) -> RGBColor {
    PALETTE[idx % PALETTE.len()]
}

#[allow(dead_code)]
pub fn draw_line_chart(
    path: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[(&str, &[(f64, f64)])],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if series.is_empty() || series.iter().all(|(_, d)| d.is_empty()) {
        return Ok(());
    }

    let (x_min, x_max, y_min, y_max) = compute_bounds(series.iter().flat_map(|(_, d)| d.iter()));

    if path.ends_with(".svg") {
        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        draw_line_chart_on(
            &root, title, x_label, y_label, series, x_min, x_max, y_min, y_max,
        )?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
        draw_line_chart_on(
            &root, title, x_label, y_label, series, x_min, x_max, y_min, y_max,
        )?;
        root.present()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn draw_line_chart_on<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[(&str, &[(f64, f64)])],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .label_style(("sans-serif", 12))
        .draw()?;

    for (idx, (name, data)) in series.iter().enumerate() {
        let c = color(idx);
        chart
            .draw_series(LineSeries::new(
                data.iter().copied(),
                ShapeStyle::from(c).stroke_width(2),
            ))?
            .label(*name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], c.stroke_width(2)));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()?;

    Ok(())
}

pub fn draw_line_chart_with_ci(
    path: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[CiSeries<'_>],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if series.is_empty() || series.iter().all(|(_, data)| data.is_empty()) {
        return Ok(());
    }
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for (_, data) in series {
        for &(x, mean, low, high) in *data {
            if x.is_finite() && mean.is_finite() && low.is_finite() && high.is_finite() {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
                y_min = y_min.min(low.min(mean));
                y_max = y_max.max(high.max(mean));
            }
        }
    }
    if !x_min.is_finite() || !x_max.is_finite() || !y_min.is_finite() || !y_max.is_finite() {
        return Ok(());
    }
    if x_min == x_max {
        x_max = x_min + 1.0;
    }
    if y_min == y_max {
        y_max = y_min + 1.0;
    }
    let y_margin = (y_max - y_min) * 0.05;
    y_min -= y_margin;
    y_max += y_margin;

    if path.ends_with(".svg") {
        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        draw_line_chart_with_ci_on(
            &root, title, x_label, y_label, series, x_min, x_max, y_min, y_max,
        )?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
        draw_line_chart_with_ci_on(
            &root, title, x_label, y_label, series, x_min, x_max, y_min, y_max,
        )?;
        root.present()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_line_chart_with_ci_on<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[CiSeries<'_>],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .label_style(("sans-serif", 12))
        .draw()?;

    for (index, (name, data)) in series.iter().enumerate() {
        let series_color = color(index);
        chart.draw_series(data.iter().filter(|&&(_, _, low, high)| high > low).map(
            |&(x, _, low, high)| {
                PathElement::new(
                    vec![(x, low), (x, high)],
                    series_color.mix(0.35).stroke_width(1),
                )
            },
        ))?;
        chart
            .draw_series(LineSeries::new(
                data.iter().map(|&(x, mean, _, _)| (x, mean)),
                series_color.stroke_width(2),
            ))?
            .label(*name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], series_color.stroke_width(2))
            });
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()?;
    Ok(())
}

#[allow(dead_code)]
pub fn draw_bar_chart(
    path: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    bars: &[(&str, f64)],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if bars.is_empty() {
        return Ok(());
    }

    let y_max = bars
        .iter()
        .map(|(_, v)| *v)
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max)
        * 1.15;
    let y_max = if y_max <= 0.0 { 1.0 } else { y_max };
    let n = bars.len();

    if path.ends_with(".svg") {
        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        draw_bar_chart_on(&root, title, x_label, y_label, bars, y_max, n)?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
        draw_bar_chart_on(&root, title, x_label, y_label, bars, y_max, n)?;
        root.present()?;
    }
    Ok(())
}

#[allow(dead_code)]
fn draw_bar_chart_on<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    y_label: &str,
    bars: &[(&str, f64)],
    y_max: f64,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((0..n).into_segmented(), 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_labels(n)
        .x_label_formatter(&|v| {
            if let SegmentValue::CenterOf(idx) = v {
                if *idx < bars.len() {
                    return bars[*idx].0.to_string();
                }
            }
            String::new()
        })
        .label_style(("sans-serif", 11))
        .draw()?;

    chart.draw_series((0..n).map(|i| {
        let c = color(i);
        Rectangle::new(
            [
                (SegmentValue::Exact(i), 0.0),
                (SegmentValue::Exact(i + 1), bars[i].1),
            ],
            c.filled(),
        )
    }))?;

    for (i, (_, val)) in bars.iter().enumerate() {
        let x_pos = SegmentValue::CenterOf(i);
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.0}", val),
            (x_pos, *val + y_max * 0.02),
            ("sans-serif", 11).into_font().color(&BLACK),
        )))?;
    }

    Ok(())
}

pub fn draw_bar_chart_with_ci(
    path: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    bars: &[(&str, f64, f64, f64)], // (label, mean, ci_low, ci_high)
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if bars.is_empty() {
        return Ok(());
    }
    let y_max = bars
        .iter()
        .map(|(_, _, _, high)| *high)
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max)
        * 1.20;
    let y_max = if y_max <= 0.0 { 1.0 } else { y_max };
    let n = bars.len();

    if path.ends_with(".svg") {
        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        draw_bar_chart_with_ci_on(&root, title, x_label, y_label, bars, y_max, n)?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
        draw_bar_chart_with_ci_on(&root, title, x_label, y_label, bars, y_max, n)?;
        root.present()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_bar_chart_with_ci_on<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    y_label: &str,
    bars: &[(&str, f64, f64, f64)],
    y_max: f64,
    n: usize,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 18))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((0..n).into_segmented(), 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_labels(n)
        .x_label_formatter(&|v| {
            if let SegmentValue::CenterOf(idx) = v {
                if *idx < bars.len() {
                    return bars[*idx].0.to_string();
                }
            }
            String::new()
        })
        .label_style(("sans-serif", 11))
        .draw()?;

    // Draw bars
    chart.draw_series((0..n).map(|i| {
        let c = color(i);
        Rectangle::new(
            [
                (SegmentValue::Exact(i), 0.0),
                (SegmentValue::Exact(i + 1), bars[i].1),
            ],
            c.filled(),
        )
    }))?;

    // Draw error bars and value labels
    for (i, (_, mean, ci_low, ci_high)) in bars.iter().enumerate() {
        let x_pos = SegmentValue::CenterOf(i);
        if ci_high > ci_low {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (x_pos.clone(), (*ci_low).max(0.0)),
                    (x_pos.clone(), *ci_high),
                ],
                BLACK.stroke_width(2),
            )))?;
        }
        // Value label
        chart.draw_series(std::iter::once(Text::new(
            format!("{:.0}", mean),
            (x_pos, ci_high.max(*mean) + y_max * 0.02),
            ("sans-serif", 10).into_font().color(&BLACK),
        )))?;
    }

    Ok(())
}

pub fn draw_box_plot(
    path: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    stats: &[(&str, [f64; 5])],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if stats.is_empty() {
        return Ok(());
    }

    let y_max = stats
        .iter()
        .map(|(_, q)| q[4])
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max)
        * 1.15;
    let y_max = if y_max <= 0.0 { 1.0 } else { y_max };

    if path.ends_with(".svg") {
        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        draw_box_plot_on(&root, title, x_label, y_label, stats, y_max)?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
        draw_box_plot_on(&root, title, x_label, y_label, stats, y_max)?;
        root.present()?;
    }
    Ok(())
}

fn draw_box_plot_on<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    y_label: &str,
    stats: &[(&str, [f64; 5])],
    y_max: f64,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let n = stats.len();

    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((0..n).into_segmented(), 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_labels(n)
        .x_label_formatter(&|v| {
            if let SegmentValue::CenterOf(idx) = v {
                if *idx < stats.len() {
                    return stats[*idx].0.to_string();
                }
            }
            String::new()
        })
        .label_style(("sans-serif", 11))
        .draw()?;

    for (i, (_, q)) in stats.iter().enumerate() {
        let c = color(i);
        let x = SegmentValue::CenterOf(i);

        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x.clone(), q[0]), (x.clone(), q[4])],
            c.stroke_width(1),
        )))?;

        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (SegmentValue::Exact(i), q[1]),
                (SegmentValue::Exact(i + 1), q[3]),
            ],
            c.mix(0.6).filled(),
        )))?;

        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (SegmentValue::Exact(i), q[2]),
                (SegmentValue::Exact(i + 1), q[2]),
            ],
            c.stroke_width(2),
        )))?;
    }

    Ok(())
}

#[allow(dead_code, clippy::too_many_arguments)]
pub fn draw_dual_axis(
    path: &str,
    title: &str,
    x_label: &str,
    y1_label: &str,
    y2_label: &str,
    series1: &[(&str, &[(f64, f64)])],
    series2: &[(&str, &[(f64, f64)])],
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if (series1.is_empty() && series2.is_empty())
        || (series1.iter().all(|(_, d)| d.is_empty()) && series2.iter().all(|(_, d)| d.is_empty()))
    {
        return Ok(());
    }

    let all_x: Vec<f64> = series1
        .iter()
        .chain(series2.iter())
        .flat_map(|(_, d)| d.iter().map(|(x, _)| *x))
        .filter(|v| v.is_finite())
        .collect();
    if all_x.is_empty() {
        return Ok(());
    }
    let x_min = all_x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = all_x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let (_, _, y1_min, y1_max) = compute_bounds(series1.iter().flat_map(|(_, d)| d.iter()));
    let (_, _, y2_min, y2_max) = compute_bounds(series2.iter().flat_map(|(_, d)| d.iter()));

    if path.ends_with(".svg") {
        let root = SVGBackend::new(path, (width, height)).into_drawing_area();
        draw_dual_axis_on(
            &root, title, x_label, y1_label, y2_label, series1, series2, x_min, x_max, y1_min,
            y1_max, y2_min, y2_max,
        )?;
        root.present()?;
    } else {
        let root = BitMapBackend::new(path, (width, height)).into_drawing_area();
        draw_dual_axis_on(
            &root, title, x_label, y1_label, y2_label, series1, series2, x_min, x_max, y1_min,
            y1_max, y2_min, y2_max,
        )?;
        root.present()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, dead_code)]
fn draw_dual_axis_on<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    title: &str,
    x_label: &str,
    y1_label: &str,
    _y2_label: &str,
    series1: &[(&str, &[(f64, f64)])],
    series2: &[(&str, &[(f64, f64)])],
    x_min: f64,
    x_max: f64,
    y1_min: f64,
    y1_max: f64,
    _y2_min: f64,
    _y2_max: f64,
) -> Result<(), Box<dyn std::error::Error>>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(root)
        .caption(title, ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .right_y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y1_min..y1_max)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y1_label)
        .label_style(("sans-serif", 12))
        .draw()?;

    for (idx, (name, data)) in series1.iter().enumerate() {
        let c = color(idx);
        chart
            .draw_series(LineSeries::new(data.iter().copied(), c.stroke_width(2)))?
            .label(*name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], c.stroke_width(2)));
    }
    for (idx, (name, data)) in series2.iter().enumerate() {
        let c = color(series1.len() + idx);
        chart
            .draw_series(DashedLineSeries::new(
                data.iter().copied(),
                5,
                3,
                c.stroke_width(2),
            ))?
            .label(*name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], c.stroke_width(2)));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 12))
        .draw()?;

    Ok(())
}

fn compute_bounds<'a>(points: impl Iterator<Item = &'a (f64, f64)>) -> (f64, f64, f64, f64) {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for (x, y) in points {
        if x.is_finite() && y.is_finite() {
            x_min = x_min.min(*x);
            x_max = x_max.max(*x);
            y_min = y_min.min(*y);
            y_max = y_max.max(*y);
        }
    }

    if !x_min.is_finite() || !x_max.is_finite() {
        x_min = 0.0;
        x_max = 1.0;
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    if x_min == x_max {
        x_max = x_min + 1.0;
    }
    if y_min == y_max {
        y_max = y_min + 1.0;
    }

    let y_margin = (y_max - y_min) * 0.05;
    (x_min, x_max, y_min - y_margin, y_max + y_margin)
}
