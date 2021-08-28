
function compile-plugins {
    rustc --crate-type cdylib ./bricks/src/dlc.rs
}
