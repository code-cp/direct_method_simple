use nalgebra as na; 
use ndarray as nd;
use anyhow::Result;

use opencv as cv2;
use cv2::prelude::*;

use crate::type_aliases::{Float, Point2}; 

trait AsArray {
    fn try_as_array(&self) -> Result<nd::Array3<u8>>;
}

impl AsArray for cv2::core::Mat {
    fn try_as_array(&self) -> Result<nd::Array3<u8>> {
        let bytes = self.data_bytes()?;
        let size = self.size()?;
        let a = nd::ArrayView3::from_shape((size.height as usize, size.width as usize, 3), bytes)?;
        Ok(a.to_owned())
    }
}

pub fn matrix_to_cv_8u(a: &na::DMatrix<u8>) -> cv2::core::Mat {
    unsafe {
        cv2::core::Mat::new_rows_cols_with_data(
            a.shape().1 as i32,
            a.shape().0 as i32,
            cv2::core::CV_8U,
            std::mem::transmute(a.as_ptr()),
            cv2::core::Mat_AUTO_STEP,
        )
        .unwrap()
    }
}

pub fn visualize_tracked_points(na_mat: &na::DMatrix<u8>, reference: &Vec<Point2>, tracked: &Vec<Point2>) -> Result<nd::Array3<u8>> {
    let cv_mat_transposed = matrix_to_cv_8u(&na_mat); 
    let mut cv_mat = cv_mat_transposed.clone(); 
    cv2::core::transpose(&cv_mat_transposed, &mut cv_mat)?;
    // println!("image width {} height {}", cv_mat.size()?.width, cv_mat.size()?.height);

    let mut color_mat: cv2::core::Mat = cv_mat.to_owned(); 
    cv2::imgproc::cvt_color(&cv_mat, &mut color_mat, cv2::imgproc::COLOR_GRAY2RGB, 0)?; 
    for (ref_pixel, tracked_pixel) in reference.iter().zip(tracked.iter()) {
        let ref_p = cv2::core::Point {x: ref_pixel.x as i32, y: ref_pixel.y as i32}; 
        cv2::imgproc::draw_marker(&mut color_mat, ref_p, cv2::core::Scalar::new(0.0,255.0,0.0, 0.0), 1, 1, 1, 1)?;

        let tracked_p = cv2::core::Point {x: tracked_pixel.x as i32, y: tracked_pixel.y as i32}; 
        cv2::imgproc::draw_marker(&mut color_mat, tracked_p, cv2::core::Scalar::new(255.0,0.0,0.0, 0.0), 1, 1, 1, 1)?;

        cv2::imgproc::line(&mut color_mat, ref_p, tracked_p, cv2::core::Scalar::new(0.0,0.0,255.0, 0.0), 1, 1, 0)?;
    }

    color_mat.try_as_array()
}