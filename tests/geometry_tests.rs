use apfp::{Coord, GeometryPredicateResult, incircle, orient2d};

#[test]
fn orient2d_detects_ccw() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    assert_eq!(orient2d(&a, &b, &c), GeometryPredicateResult::Positive);
}

#[test]
fn orient2d_detects_cw() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(0.0, 1.0);
    let c = Coord::new(1.0, 0.0);
    assert_eq!(orient2d(&a, &b, &c), GeometryPredicateResult::Negative);
}

#[test]
fn orient2d_detects_collinear() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 1.0);
    let c = Coord::new(2.0, 2.0);
    assert_eq!(orient2d(&a, &b, &c), GeometryPredicateResult::Zero);
}

#[test]
fn incircle_positive_for_point_inside() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    let d = Coord::new(0.25, 0.25);
    assert_eq!(incircle(&a, &b, &c, &d), GeometryPredicateResult::Positive);
}

#[test]
fn incircle_negative_for_point_outside() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    let d = Coord::new(2.0, 2.0);
    assert_eq!(incircle(&a, &b, &c, &d), GeometryPredicateResult::Negative);
}

#[test]
fn incircle_zero_on_circumference() {
    let a = Coord::new(0.0, 0.0);
    let b = Coord::new(1.0, 0.0);
    let c = Coord::new(0.0, 1.0);
    // Point on circumcircle: (1,1)
    let d = Coord::new(1.0, 1.0);
    assert_eq!(incircle(&a, &b, &c, &d), GeometryPredicateResult::Zero);
}
