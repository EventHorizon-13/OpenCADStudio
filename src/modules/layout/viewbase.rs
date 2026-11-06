//! A lightweight, non-associative VIEWBASE implementation.
//!
//! The command projects B-rep drawing edges and view-dependent silhouettes into
//! editable 2D linework. The first-angle arrangement follows common mechanical
//! drawing practice: top below front, left view to the right, and an isometric
//! view at the far right.

use std::collections::HashMap;

use glam::{DVec2, DVec3};

use crate::command::{CadCommand, CmdOption, CmdResult};
use crate::modules::{IconKind, ModuleEvent, ToolDef};
use crate::scene::model::wire_model::WireModel;

#[derive(Clone, Copy, Debug)]
pub struct ProjectionSegment {
    pub start: DVec3,
    pub end: DVec3,
    pub hidden: bool,
}

struct BodyGeometry {
    mesh: kernel::brep::mesh::Mesh,
    drawing_edges: Vec<Vec<DVec3>>,
    silhouette: kernel::brep::mesh::SilhouetteSource,
}

#[derive(Clone, Copy)]
struct ViewBasis {
    right: DVec3,
    up: DVec3,
    toward_viewer: DVec3,
}

impl ViewBasis {
    fn new(toward_viewer: DVec3, up_hint: DVec3) -> Self {
        let toward_viewer = if toward_viewer.length_squared() > 1e-18 {
            toward_viewer.normalize()
        } else {
            DVec3::Y
        };
        let right = up_hint.cross(toward_viewer).normalize_or_zero();
        let up = toward_viewer.cross(right).normalize_or_zero();
        Self {
            right,
            up,
            toward_viewer,
        }
    }

    fn from_axes(right: DVec3, up: DVec3, toward_viewer: DVec3) -> Self {
        Self {
            right: right.normalize_or_zero(),
            up: up.normalize_or_zero(),
            toward_viewer: toward_viewer.normalize_or_zero(),
        }
    }

    fn project(self, point: DVec3) -> (DVec2, f64) {
        (
            DVec2::new(point.dot(self.right), point.dot(self.up)),
            point.dot(self.toward_viewer),
        )
    }
}

#[derive(Clone, Copy)]
struct ProjectedTriangle {
    a: DVec2,
    b: DVec2,
    c: DVec2,
    da: f64,
    db: f64,
    dc: f64,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
    area: f64,
}

impl ProjectedTriangle {
    fn new(points: [DVec3; 3], basis: ViewBasis) -> Option<Self> {
        let (a, da) = basis.project(points[0]);
        let (b, db) = basis.project(points[1]);
        let (c, dc) = basis.project(points[2]);
        let area = (b - a).perp_dot(c - a);
        if !area.is_finite() || area.abs() <= 1e-12 {
            return None;
        }
        Some(Self {
            a,
            b,
            c,
            da,
            db,
            dc,
            min_x: a.x.min(b.x).min(c.x),
            min_y: a.y.min(b.y).min(c.y),
            max_x: a.x.max(b.x).max(c.x),
            max_y: a.y.max(b.y).max(c.y),
            area,
        })
    }

    fn depth_at(self, point: DVec2) -> Option<f64> {
        if point.x < self.min_x - 1e-9
            || point.x > self.max_x + 1e-9
            || point.y < self.min_y - 1e-9
            || point.y > self.max_y + 1e-9
        {
            return None;
        }
        let v = (point - self.a).perp_dot(self.c - self.a) / self.area;
        let w = (self.b - self.a).perp_dot(point - self.a) / self.area;
        let u = 1.0 - v - w;
        const EDGE_EPSILON: f64 = 1e-8;
        if u < -EDGE_EPSILON
            || v < -EDGE_EPSILON
            || w < -EDGE_EPSILON
            || u > 1.0 + EDGE_EPSILON
            || v > 1.0 + EDGE_EPSILON
            || w > 1.0 + EDGE_EPSILON
        {
            return None;
        }
        Some(u * self.da + v * self.db + w * self.dc)
    }
}

/// Project one or more restored kernel bodies into four first-angle views.
///
/// The returned segments share one projected model origin and are already
/// arranged on the sheet. Hidden portions are marked so the command can render
/// them as dashed, editable line entities at the chosen scale.
pub fn project_bodies(
    bodies: &[kernel::brep::Body],
) -> Option<(Vec<ProjectionSegment>, f64)> {
    let mut prepared = Vec::new();
    let mut low = DVec3::splat(f64::INFINITY);
    let mut high = DVec3::splat(f64::NEG_INFINITY);

    for body in bodies {
        let tessellation = kernel::brep::mesh::tessellate(
            body,
            kernel::brep::mesh::TessellationTolerance::new(
                kernel::tessellation::DEFAULT_ANGLE,
                1e-9,
            ),
        );
        if !tessellation.missing_faces.is_empty() {
            return None;
        }
        if tessellation.mesh.is_empty() {
            continue;
        }

        for point in &tessellation.mesh.positions {
            let point = DVec3::from_array(*point);
            low = low.min(point);
            high = high.max(point);
        }

        let drawing_edges = tessellation
            .drawing_edges
            .iter()
            .filter_map(|edge| {
                let points = edge
                    .positions
                    .iter()
                    .copied()
                    .map(DVec3::from_array)
                    .collect::<Vec<_>>();
                (points.len() >= 2).then_some(points)
            })
            .collect::<Vec<_>>();

        let silhouette = tessellation.silhouette_source();
        prepared.push(BodyGeometry {
            mesh: tessellation.mesh,
            drawing_edges,
            silhouette,
        });
    }

    if prepared.is_empty() || !low.is_finite() || !high.is_finite() {
        return None;
    }

    let model_diagonal = (high - low).length().max(1e-6);
    let model_centre = (low + high) * 0.5;
    let triangles = prepared
        .iter()
        .flat_map(|body| {
            body.mesh.triangles.iter().map(|triangle| {
                [
                    DVec3::from_array(body.mesh.positions[triangle[0]]),
                    DVec3::from_array(body.mesh.positions[triangle[1]]),
                    DVec3::from_array(body.mesh.positions[triangle[2]]),
                ]
            })
        })
        .collect::<Vec<_>>();

    let bases = [
        // Front: +X to the right, +Z up, viewer on -Y.
        ViewBasis::new(-DVec3::Y, DVec3::Z),
        // First-angle top: front (-Y) is at the top, next to the front view.
        ViewBasis::from_axes(DVec3::X, -DVec3::Y, DVec3::Z),
        // First-angle left view: the front (-Y) edge is on the left.
        ViewBasis::from_axes(DVec3::Y, DVec3::Z, -DVec3::X),
        // Isometric: viewer on the +X / -Y / +Z diagonal.
        ViewBasis::new(DVec3::new(1.0, -1.0, 1.0), DVec3::Z),
    ];

    let mut views = Vec::with_capacity(4);
    for basis in bases {
        let projected_origin = basis.project(model_centre).0;
        let projected_triangles = triangles
            .iter()
            .filter_map(|triangle| ProjectedTriangle::new(*triangle, basis))
            .collect::<Vec<_>>();

        let mut segments = Vec::new();
        let mut seen = HashMap::new();
        let depth_tolerance = model_diagonal * 1e-7;

        for body in &prepared {
            for edge in &body.drawing_edges {
                for pair in edge.windows(2) {
                    append_projected_edge(
                        pair[0],
                        pair[1],
                        basis,
                        projected_origin,
                        &projected_triangles,
                        depth_tolerance,
                        model_diagonal,
                        &mut seen,
                        &mut segments,
                    );
                }
            }

            // The kernel computes smooth-face silhouettes analytically where
            // possible and from the tessellated face normals elsewhere.
            let silhouette = kernel::brep::mesh::silhouette(
                &body.silhouette,
                basis.toward_viewer.to_array(),
            );
            for pair in silhouette.chunks_exact(2) {
                append_projected_edge(
                    DVec3::from_array(pair[0]),
                    DVec3::from_array(pair[1]),
                    basis,
                    projected_origin,
                    &projected_triangles,
                    depth_tolerance,
                    model_diagonal,
                    &mut seen,
                    &mut segments,
                );
            }
        }

        let Some(bounds) = segment_bounds(&segments) else {
            return None;
        };
        views.push((segments, bounds));
    }

    let gap = model_diagonal * 0.12;
    let front_bounds = views[0].1;
    let top_bounds = views[1].1;
    let left_bounds = views[2].1;
    let iso_bounds = views[3].1;

    // Preserve the shared projection axes while unfolding the views into a
    // first-angle sheet arrangement: top below front, left to the right.
    let front_centre_y = (front_bounds.0.y + front_bounds.1.y) * 0.5;
    let iso_centre_y = (iso_bounds.0.y + iso_bounds.1.y) * 0.5;
    let offsets = [
        DVec2::ZERO,
        DVec2::new(0.0, front_bounds.0.y - gap - top_bounds.1.y),
        DVec2::new(front_bounds.1.x + gap - left_bounds.0.x, 0.0),
        DVec2::new(
            left_bounds.1.x + gap - iso_bounds.0.x,
            front_centre_y - iso_centre_y,
        ),
    ];

    let mut output = Vec::new();
    for ((segments, _), offset) in views.into_iter().zip(offsets) {
        output.extend(segments.into_iter().map(|mut segment| {
            segment.start.x += offset.x;
            segment.start.y += offset.y;
            segment.end.x += offset.x;
            segment.end.y += offset.y;
            segment
        }));
    }

    (!output.is_empty()).then_some((output, model_diagonal))
}

fn append_projected_edge(
    start: DVec3,
    end: DVec3,
    basis: ViewBasis,
    projected_origin: DVec2,
    triangles: &[ProjectedTriangle],
    depth_tolerance: f64,
    model_diagonal: f64,
    seen: &mut HashMap<((i64, i64), (i64, i64)), usize>,
    output: &mut Vec<ProjectionSegment>,
) {
    let length = start.distance(end);
    if !length.is_finite() || length <= model_diagonal * 1e-12 {
        return;
    }

    // Short subdivisions allow an edge to switch between visible and hidden
    // without requiring a full curve-intersection solver.
    let subdivisions = ((length / (model_diagonal * 0.08)).ceil() as usize).clamp(1, 8);
    for index in 0..subdivisions {
        let t0 = index as f64 / subdivisions as f64;
        let t1 = (index + 1) as f64 / subdivisions as f64;
        let a = start.lerp(end, t0);
        let b = start.lerp(end, t1);
        let middle = (a + b) * 0.5;
        let hidden = is_occluded(middle, basis, triangles, depth_tolerance);
        let (a2, _) = basis.project(a);
        let (b2, _) = basis.project(b);
        let a2 = a2 - projected_origin;
        let b2 = b2 - projected_origin;
        if a2.distance(b2) <= model_diagonal * 1e-12 {
            continue;
        }

        let tolerance = (model_diagonal * 1e-8).max(1e-9);
        let mut ka = (
            (a2.x / tolerance).round() as i64,
            (a2.y / tolerance).round() as i64,
        );
        let mut kb = (
            (b2.x / tolerance).round() as i64,
            (b2.y / tolerance).round() as i64,
        );
        if kb < ka {
            std::mem::swap(&mut ka, &mut kb);
        }
        let key = (ka, kb);
        if let Some(existing) = seen.get(&key).copied() {
            // If coincident projected edges include a visible one, keep the
            // visible representation instead of hiding it behind a duplicate.
            if !hidden {
                output[existing].hidden = false;
            }
            continue;
        }

        seen.insert(key, output.len());
        output.push(ProjectionSegment {
            start: DVec3::new(a2.x, a2.y, 0.0),
            end: DVec3::new(b2.x, b2.y, 0.0),
            hidden,
        });
    }
}

fn is_occluded(
    point: DVec3,
    basis: ViewBasis,
    triangles: &[ProjectedTriangle],
    depth_tolerance: f64,
) -> bool {
    let (projected, depth) = basis.project(point);
    triangles.iter().any(|triangle| {
        triangle
            .depth_at(projected)
            .is_some_and(|other_depth| other_depth > depth + depth_tolerance)
    })
}

fn segment_bounds(segments: &[ProjectionSegment]) -> Option<(DVec2, DVec2)> {
    let mut low = DVec2::splat(f64::INFINITY);
    let mut high = DVec2::splat(f64::NEG_INFINITY);
    for segment in segments {
        low = low.min(segment.start.truncate()).min(segment.end.truncate());
        high = high.max(segment.start.truncate()).max(segment.end.truncate());
    }
    (low.is_finite() && high.is_finite()).then_some((low, high))
}

pub fn tool() -> ToolDef {
    ToolDef {
        id: "VIEWBASE",
        label: "Base View",
        icon: IconKind::Svg(include_bytes!("../../../assets/icons/model/flatshot.svg")),
        event: ModuleEvent::Command("VIEWBASE".to_string()),
    }
}

pub struct ViewBaseCommand {
    segments: Vec<ProjectionSegment>,
    model_diagonal: f64,
    scale: f64,
    awaiting_scale: bool,
}

impl ViewBaseCommand {
    pub fn new(segments: Vec<ProjectionSegment>, model_diagonal: f64) -> Self {
        Self {
            segments,
            model_diagonal,
            scale: 1.0,
            awaiting_scale: false,
        }
    }

    fn dash_lengths(&self) -> (f64, f64) {
        let dash = (self.model_diagonal * self.scale * 0.008).max(1e-9);
        (dash, dash * 0.65)
    }

    fn append_preview_segment(
        &self,
        points: &mut Vec<[f64; 3]>,
        start: DVec2,
        end: DVec2,
        z: f64,
        hidden: bool,
    ) {
        let delta = end - start;
        let length = delta.length();
        if length <= 1e-9 {
            return;
        }
        if hidden {
            let (dash, gap) = self.dash_lengths();
            let direction = delta / length;
            let mut offset = 0.0;
            while offset < length {
                let finish = (offset + dash).min(length);
                let a = start + direction * offset;
                let b = start + direction * finish;
                points.push([a.x, a.y, z]);
                points.push([b.x, b.y, z]);
                points.push([f64::NAN; 3]);
                offset = finish + gap;
            }
        } else {
            points.push([start.x, start.y, z]);
            points.push([end.x, end.y, z]);
            points.push([f64::NAN; 3]);
        }
    }

    fn append_entities(&self, anchor: DVec3) -> Vec<codec::EntityType> {
        use codec::types::Vector3;
        use codec::Line;

        let mut entities = Vec::new();
        let (dash, gap) = self.dash_lengths();

        for segment in &self.segments {
            let start = DVec2::new(
                anchor.x + segment.start.x * self.scale,
                anchor.y + segment.start.y * self.scale,
            );
            let end = DVec2::new(
                anchor.x + segment.end.x * self.scale,
                anchor.y + segment.end.y * self.scale,
            );
            let delta = end - start;
            let length = delta.length();
            if length <= 1e-9 {
                continue;
            }

            let mut emit = |a: DVec2, b: DVec2| {
                if a.distance(b) <= 1e-9 {
                    return;
                }
                let line = Line::from_points(
                    Vector3::new(a.x, a.y, anchor.z),
                    Vector3::new(b.x, b.y, anchor.z),
                );
                entities.push(codec::EntityType::Line(line));
            };

            if segment.hidden {
                let direction = delta / length;
                let mut offset = 0.0;
                while offset < length {
                    let finish = (offset + dash).min(length);
                    emit(start + direction * offset, start + direction * finish);
                    offset = finish + gap;
                }
            } else {
                emit(start, end);
            }
        }
        entities
    }
}

impl CadCommand for ViewBaseCommand {
    fn name(&self) -> &'static str {
        "VIEWBASE"
    }

    fn prompt(&self) -> String {
        if self.awaiting_scale {
            "VIEWBASE  Enter scale factor (e.g. 1:2 or 0.5):".to_string()
        } else {
            format!(
                "VIEWBASE  Specify front-view placement point [Scale] <1:1> (factor {:.4}):",
                self.scale
            )
        }
    }

    fn options(&self) -> Vec<CmdOption> {
        if self.awaiting_scale {
            Vec::new()
        } else {
            vec![CmdOption::new("Scale", "S")]
        }
    }

    fn wants_text_input(&self) -> bool {
        true
    }

    fn on_text_input(&mut self, text: &str) -> Option<CmdResult> {
        let value = text.trim();
        if !self.awaiting_scale && matches!(value.to_ascii_uppercase().as_str(), "S" | "SCALE") {
            self.awaiting_scale = true;
            return Some(CmdResult::NeedPoint);
        }
        if self.awaiting_scale {
            if let Some(scale) = parse_scale(value) {
                self.scale = scale;
                self.awaiting_scale = false;
            }
            return Some(CmdResult::NeedPoint);
        }
        None
    }

    fn on_point(&mut self, point: DVec3) -> CmdResult {
        let entities = self.append_entities(point);
        if entities.is_empty() {
            CmdResult::Cancel
        } else {
            CmdResult::CommitEntitiesAndExit(entities)
        }
    }

    fn on_enter(&mut self) -> CmdResult {
        if self.awaiting_scale {
            self.awaiting_scale = false;
            CmdResult::NeedPoint
        } else {
            CmdResult::Cancel
        }
    }

    fn on_preview_wires(&mut self, point: DVec3) -> Vec<WireModel> {
        let mut points = Vec::new();
        for segment in &self.segments {
            let start = DVec2::new(
                point.x + segment.start.x * self.scale,
                point.y + segment.start.y * self.scale,
            );
            let end = DVec2::new(
                point.x + segment.end.x * self.scale,
                point.y + segment.end.y * self.scale,
            );
            self.append_preview_segment(&mut points, start, end, point.z, segment.hidden);
        }
        if points.is_empty() {
            return Vec::new();
        }
        vec![WireModel::solid_f64(
            "VIEWBASE-PREVIEW".to_string(),
            points,
            WireModel::CYAN,
            false,
        )]
    }
}

fn parse_scale(text: &str) -> Option<f64> {
    let text = text.trim();
    if let Some((numerator, denominator)) = text
        .split_once(':')
        .or_else(|| text.split_once('/'))
    {
        let numerator = numerator.trim().parse::<f64>().ok()?;
        let denominator = denominator.trim().parse::<f64>().ok()?;
        if numerator.is_finite()
            && denominator.is_finite()
            && numerator > 0.0
            && denominator > 0.0
        {
            let scale = numerator / denominator;
            return (scale.is_finite() && scale > 0.0).then_some(scale);
        }
        return None;
    }
    let scale = text.parse::<f64>().ok()?;
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_angle_view_bases_keep_expected_axes() {
        let front = ViewBasis::new(-DVec3::Y, DVec3::Z);
        let (front_x, _) = front.project(DVec3::X);
        let (front_z, _) = front.project(DVec3::Z);
        assert!(front_x.x > 0.0 && front_x.y.abs() < 1e-12);
        assert!(front_z.y > 0.0 && front_z.x.abs() < 1e-12);

        let top = ViewBasis::from_axes(DVec3::X, -DVec3::Y, DVec3::Z);
        let (top_front, _) = top.project(-DVec3::Y);
        let (top_back, _) = top.project(DVec3::Y);
        assert!(top_front.y > 0.0 && top_back.y < 0.0);

        let left = ViewBasis::from_axes(DVec3::Y, DVec3::Z, -DVec3::X);
        let (left_front, _) = left.project(-DVec3::Y);
        assert!(left_front.x < 0.0);
    }

    #[test]
    fn scale_parser_accepts_ratios_and_factors() {
        assert!((parse_scale("1:2").unwrap() - 0.5).abs() < 1e-12);
        assert!((parse_scale("1/4").unwrap() - 0.25).abs() < 1e-12);
        assert!((parse_scale("2").unwrap() - 2.0).abs() < 1e-12);
        assert!(parse_scale("0").is_none());
    }

    #[test]
    fn cuboid_produces_four_view_linework() {
        let body = kernel::brep::make::cuboid([0.0; 3], [10.0; 3]).expect("cuboid");
        let (segments, diagonal) = project_bodies(&[body]).expect("projected views");
        assert!(diagonal > 0.0);
        assert!(segments.len() > 12);
        assert!(segments.iter().all(|segment| {
            segment.start.z == 0.0
                && segment.end.z == 0.0
                && segment.start.is_finite()
                && segment.end.is_finite()
        }));
    }
}
