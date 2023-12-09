use nalgebra::Affine2; 
use crate::type_aliases::{Float, SE3, Mat3, Point2, Point3, Vec3};

#[derive(PartialEq, Debug, Clone)]
pub struct Intrinsics {
    pub principal_point: (Float, Float), 
    pub focal: (Float, Float), 
}

impl Intrinsics {
    pub fn matrix(&self) -> Affine2<Float> {
        Affine2::from_matrix_unchecked(Mat3::new(
            self.focal.0, 0.0,    self.principal_point.0,
            0.0,          self.focal.1, self.principal_point.1,
            0.0,          0.0,          1.0,
        ))
    }

    pub fn project(&self, point: &Point3) -> Vec3 {
        // homogeneous point 
        Vec3::new(
            self.focal.0 * point[0] / point[2] + self.principal_point.0,
            self.focal.1 * point[1] / point[2] + self.principal_point.1,
            1.0,
        )
    }

    pub fn back_project(&self, point: &Point2, depth: Float) -> Point3 {
        let x = (point[0] - self.principal_point.0) * depth / self.focal.0;
        let y = (point[1] - self.principal_point.1) * depth / self.focal.1;
        Point3::new(x, y, depth)
    }

    pub fn get_fx(&self) -> Float {
        self.focal.0 
    }

    pub fn get_fy(&self) -> Float {
        self.focal.1 
    }

    pub fn get_cx(&self) -> Float {
        self.principal_point.0 
    }

    pub fn get_cy(&self) -> Float {
        self.principal_point.1 
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Camera {
    pub intrinsics: Intrinsics, 
    pub w_tr_c: SE3, 
}

impl Camera {
    pub fn new(intrinsics: Intrinsics, w_tr_c: SE3) -> Self {
        Self {
            intrinsics, 
            w_tr_c, 
        }
    }

    pub fn project(&self, point: &Point3) -> Vec3 {
        self.intrinsics
            .project(&extrinsics::project(&self.w_tr_c, point))
    }

    pub fn back_project(&self, point: &Point2, depth: Float) -> Point3 {
        extrinsics::back_project(&self.w_tr_c, &self.intrinsics.back_project(point, depth))
    }
}

pub mod extrinsics {
    use super::*;

    pub fn project(w_tr_c: &SE3, point: &Point3) -> Point3 {
        let c_tr_w = w_tr_c.inverse(); 
        c_tr_w.translation * (c_tr_w.rotation * point)
    }

    pub fn back_project(w_tr_c: &SE3, point: &Point3) -> Point3 {
        w_tr_c * point
    }
}