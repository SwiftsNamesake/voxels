use lyon::math::{rect, Point};
use lyon::path::{builder::*, Winding};
use lyon::tessellation::{FillTessellator, FillOptions, VertexBuffers};
use lyon::tessellation::geometry_builder::simple_builder;

pub fn rounded_rectangle() -> lyon::lyon_tessellation::VertexBuffers<Point, u16> {
    let mut geometry: VertexBuffers<Point, u16> = VertexBuffers::new();
    let mut geometry_builder = simple_builder(&mut geometry);
    let options = FillOptions::tolerance(0.001);
    let mut tessellator = FillTessellator::new();

    let mut builder = tessellator.builder(
        &options,
        &mut geometry_builder,
    );

    let w = 100.0;
    let h = 100.0;
    builder.add_rounded_rectangle(
        &rect(0.0, 0.0, w, h),
        &BorderRadii {
            top_left: 20.0,
            top_right: 20.0,
            bottom_left: 20.0,
            bottom_right: 20.0,
        },
        Winding::Positive
    );
    // builder.cubic_bezier_to(point(0.17, 0.67), point(0.83, 0.67), point(5.87, 7.46));
    // builder.close();
    builder.build().expect("Failed to build with tesselator builder");

    return geometry;
}