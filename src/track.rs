use std::collections::binary_heap;

use nalgebra as na; 
use nalgebra::DMatrix;
use rand::Rng;

use crate::type_aliases::{Float, Point2, SE3, Mat26, Mat3, Mat6, Vec2, Vec3, Vec6}; 
use crate::dataset::DEPTH_SCALE; 
use crate::camera::Intrinsics; 

pub fn get_pixel_values(img_width: usize, img_height: usize) -> Vec<Point2> {
    let total_pixels = 500; 
    let mut pixels = Vec::with_capacity(total_pixels); 
    let border = 20; 
    
    let mut rng = rand::thread_rng();
    for _ in 0..total_pixels {
        let random_u = rng.gen_range(border, img_width - border);
        let random_v = rng.gen_range(border, img_height - border);

        pixels.push(Point2::new(random_u as f32, random_v as f32));
    }

    pixels 
} 

pub fn get_depth_values(depth_image: &DMatrix<u16>, pixels: &Vec<Point2>) -> Vec<Float> {
    let get_single_depth = |depth_image: &DMatrix<u16>, pixel: &Point2| {
        depth_image[(pixel.y as usize, pixel.x as usize)] as f32 / DEPTH_SCALE as f32
    };

    let depth_values = pixels
        .iter()
        .map(|p| get_single_depth(depth_image, p))
        .collect(); 

    depth_values
}

pub fn direct_pose_estimation_single_layer(
    img1: &DMatrix<u8>, 
    img2: &DMatrix<u8>, 
    intrinsics: &Intrinsics, 
    pixels: &Vec<Point2>, 
    depth_values: &Vec<Float>,
    t2_tr_t1: &mut SE3,  
) -> Vec<Point2> {
    let iterations = 10; 
    let mut cost = 0.0; 
    let mut last_cost = 0.0; 
    let mut projected_points = Vec::with_capacity(pixels.len()); 
    let tolerance = 1e-3;

    for _ in 0..iterations {
        let mut good_point_count = 0; 
        let mut hessian = Mat6::zeros(); 
        let mut bias: na::Matrix<f32, na::U6, na::U1, na::ArrayStorage<f32, na::U6, na::U1>> = Vec6::zeros();     

        projected_points.clear(); 

        for (pixel, depth) in pixels.iter().zip(depth_values.iter()) {
            projected_points.push(*pixel); 
            
            if *depth < 1e-3 {
                // ignore missing depth 
                continue; 
            }
            let point_ref = intrinsics.back_project(pixel, *depth); 
            let point_cur = *t2_tr_t1 * point_ref; 
            if point_cur.z < 1e-3 {
                // ignore invalid depth after transformation 
                continue; 
            }
    
            let uv = intrinsics.project(&point_cur); 
            let (u, v) = (uv[0], uv[1]); 
            if (u as usize) < 1 || (u as usize) > img2.ncols() - 1 ||
                (v as usize) < 1 || (v as usize) > img2.nrows() - 1 {
                    // ignore invalid pixel 
                    continue;  
            }
    
            let n = projected_points.len();
            projected_points[n - 1] = Point2::new(u, v); 
            good_point_count += 1; 
    
            let (x, y, z) = (point_cur.x, point_cur.y, point_cur.z); 
            let (z_inv, z2_inv) = (1.0 / z, 1.0 / (z * z));
    
            for i in -1..=1 {
                for j in -1..=1 {
                    let (du, dv) = (i as f32, j as f32); 
                    let residual = bilinear_interpolation(img1, pixel.x + du, pixel.y + dv) - 
                    bilinear_interpolation(img2, u + du, v + dv);
    
                    let mut j_pixel_xi = Mat26::zeros(); 
                    let mut j_img_pixel = Vec2::zeros(); 
    
                    j_pixel_xi[(0, 0)] = intrinsics.get_fx() * z_inv; 
                    j_pixel_xi[(0, 2)] = -intrinsics.get_fx() * x * z2_inv; 
                    j_pixel_xi[(0, 3)] = -intrinsics.get_fx() * x * y * z2_inv; 
                    j_pixel_xi[(0, 4)] = intrinsics.get_fx() + intrinsics.get_fx() * x * x * z2_inv;
                    j_pixel_xi[(0, 5)] = -intrinsics.get_fx() * y * z_inv; 
    
                    j_pixel_xi[(1, 1)] = intrinsics.get_fy() * z_inv; 
                    j_pixel_xi[(1, 2)] = -intrinsics.get_fy() * y * z2_inv; 
                    j_pixel_xi[(1, 3)] = -intrinsics.get_fy() - intrinsics.get_fy() * y * y * z2_inv; 
                    j_pixel_xi[(1, 4)] = intrinsics.get_fy() * x * y * z2_inv; 
                    j_pixel_xi[(1, 5)] = intrinsics.get_fy() * x * z_inv; 
    
                    j_img_pixel[0] = 0.5 * (bilinear_interpolation(img2, u + 1.0 + du, v + dv)
                        - bilinear_interpolation(img2, u - 1.0 + du, v + dv)
                    );
                    j_img_pixel[1] = 0.5 * (bilinear_interpolation(img2, u + du, v + 1.0 + dv)
                    - bilinear_interpolation(img2, u + du, v - 1.0 + dv)
                    );
    
                    let jacobian = -1.0 * (j_img_pixel.transpose() * j_pixel_xi).transpose();
    
                    hessian += jacobian * jacobian.transpose(); 
                    bias += -residual * jacobian; 
                    cost += residual * residual; 
                }
            }
        } 
        if good_point_count == 0 {
            continue; 
        }

        cost /= good_point_count as f32;

        if cost > last_cost {
            break; 
        }

        // Solve the normal equations: J^T * J * update = J^T * residual
        // svd solve Solves the system self * x = b where self is the decomposed matrix and x the unknown.
        let update = na::linalg::SVD::new(hessian, true, true)
            .solve(&bias, tolerance)
            .unwrap_or(Vec6::zeros());

        if update.norm() < 1e-3 {
            break; 
        }

        *t2_tr_t1 = exp_map(&update) * *t2_tr_t1;
        last_cost = cost; 
    }

    projected_points
}

/// slam book eq. 2.4 
fn skew(v: na::Vector3<Float>) -> Mat3 {
    let mut ss = Mat3::zeros();
    ss[(0, 1)] = -v[2];
    ss[(0, 2)] = v[1];
    ss[(1, 0)] = v[2];
    ss[(1, 2)] = -v[0];
    ss[(2, 0)] = -v[1];
    ss[(2, 1)] = v[0];
    ss
}

/// slam book eq. 3.27
fn calculate_j(theta: Float, n: &Vec3) -> Mat3 {
    let i3 = Mat3::identity(); 
    // need to check value of theta 
    // otherwise this function will be super slow 
    if theta > 1e-6 {
        (theta.sin() / theta) * i3 
        + (1.0 - theta.sin() / theta) * n * n.transpose() 
        + (1.0 - theta.cos()) / theta * skew(*n)
    } else {
        i3 
    }
}

/// slam book p. 65 
fn exp_map(v: &Vec6) -> SE3 {
    let rho = Vec3::new(v[0], v[1], v[2]); 
    let phi = Vec3::new(v[3], v[4], v[5]); 
    
    let theta = phi.norm();
    let n = phi / theta; 
    let exp_phi_skew = theta.cos() * Mat3::identity() 
        + (1.0 - theta.cos()) * n * n.transpose() 
        + theta.sin() * skew(n);  

    let j = calculate_j(theta, &n);   

    let t = na::Translation3::from(j * rho); 
    let r = na::Rotation3::from_matrix(&exp_phi_skew); 
    na::Isometry3::from_parts(t, r.into())
}

fn bilinear_interpolation(img: &DMatrix<u8>, x: f32, y: f32) -> f32 {
    // Boundary check
    let x = if x < 0.0 { 0.0 } else if x >= img.ncols() as f32 { img.ncols() as f32 - 1.0 } else { x };
    let y = if y < 0.0 { 0.0 } else if y >= img.nrows() as f32 { img.nrows() as f32 - 1.0 } else { y };

    let x_floor = x.floor() as usize;
    let y_floor = y.floor() as usize;

    let xx = x - x_floor as f32;
    let yy = y - y_floor as f32;

    let data00 = img[(y_floor, x_floor)] as f32;
    let data10 = img[(y_floor, x_floor + 1)] as f32;
    let data01 = img[(y_floor + 1, x_floor)] as f32;
    let data11 = img[(y_floor + 1, x_floor + 1)] as f32;

    let interpolated_value = (1.0 - xx) * (1.0 - yy) * data00
        + xx * (1.0 - yy) * data10
        + (1.0 - xx) * yy * data01
        + xx * yy * data11;

    interpolated_value
}