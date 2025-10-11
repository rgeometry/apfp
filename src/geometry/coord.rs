/// 2D coordinate represented with ordinary floating-point inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coord {
    pub x: f64,
    pub y: f64,
}

impl Coord {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl From<(f64, f64)> for Coord {
    fn from(value: (f64, f64)) -> Self {
        Coord::new(value.0, value.1)
    }
}
