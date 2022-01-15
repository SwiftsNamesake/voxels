use lyon::math::{rect, Point};
use lyon::path::{builder::*, Winding};
use lyon::tessellation::{FillTessellator, StrokeTessellator, FillOptions, StrokeOptions, VertexBuffers};
use lyon::tessellation::geometry_builder::simple_builder;

pub fn rounded_rectangle_outlined(width: f32, height: f32, line_width: f32, border_radius: f32) -> lyon::lyon_tessellation::VertexBuffers<Point, u16> {
    let mut geometry: VertexBuffers<Point, u16> = VertexBuffers::new();
    let mut geometry_builder = simple_builder(&mut geometry);

    // let options = FillOptions::tolerance(0.001);
    // let mut tessellator = FillTessellator::new();

    let options = StrokeOptions::default().with_tolerance(0.001).with_line_width(line_width);
    let mut tessellator = StrokeTessellator::new();

    let mut builder = tessellator.builder(
        &options,
        &mut geometry_builder,
    );

    builder.add_rounded_rectangle(
        &rect(0.0, 0.0, width, height),
        &BorderRadii {
            top_left: border_radius,
            top_right: border_radius,
            bottom_left: border_radius,
            bottom_right: border_radius,
        },
        Winding::Positive
    );
    // builder.cubic_bezier_to(point(0.17, 0.67), point(0.83, 0.67), point(5.87, 7.46));
    // builder.close();
    builder.build().expect("Failed to build with tesselator builder");

    return geometry;
}

pub fn rounded_rectangle_filled(width: f32, height: f32, border_radius: f32) -> lyon::lyon_tessellation::VertexBuffers<Point, u16> {
    let mut geometry: VertexBuffers<Point, u16> = VertexBuffers::new();
    let mut geometry_builder = simple_builder(&mut geometry);

    let options = FillOptions::tolerance(0.001);
    let mut tessellator = FillTessellator::new();

    // let options = StrokeOptions::default().with_tolerance(0.001).with_line_width(line_width);
    // let mut tessellator = StrokeTessellator::new();

    let mut builder = tessellator.builder(
        &options,
        &mut geometry_builder,
    );

    builder.add_rounded_rectangle(
        &rect(0.0, 0.0, width, height),
        &BorderRadii {
            top_left: border_radius,
            top_right: border_radius,
            bottom_left: border_radius,
            bottom_right: border_radius,
        },
        Winding::Positive
    );
    // builder.cubic_bezier_to(point(0.17, 0.67), point(0.83, 0.67), point(5.87, 7.46));
    // builder.close();
    builder.build().expect("Failed to build with tesselator builder");

    return geometry;
}
