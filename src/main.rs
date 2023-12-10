use image; 
use nalgebra as na; 
use ndarray as nd; 
use direct_method_simple as dms;

use na::DMatrix;
use std::{env, error::Error, fs, io::BufReader, io::Read, path::Path, path::PathBuf};
use rerun::RecordingStreamBuilder;

use dms::dataset;
use dms::track::*; 
use dms::type_aliases::*; 
use dms::visualize::*; 

fn main() -> Result<(), Box<dyn Error>> {
    // visualization 
    let rec = RecordingStreamBuilder::new("direct_method_simple").save("./logs/my_recording.rrd")?;

    let associations_file_path = "/home/sean/workspace/rgbd_dataset_freiburg1_xyz/associations.txt"; 
    let associations = parse_associations(associations_file_path)?;

    let intrinsics = dataset::INTRINSICS_FR1; 
    let mut w_tr_c = SE3::identity(); 

    // If we log a pinhole camera model, the depth gets automatically back-projected to 3D
    rec.log(
        "world/camera/#0/image",
        &rerun::Pinhole::from_focal_length_and_resolution(
            intrinsics.focal,
            intrinsics.get_resolution(),
        ),
    )?;

    let mut pixels = Vec::with_capacity(500);
    let mut prev_depth_map = na::DMatrix::<u16>::zeros(intrinsics.get_resolution().1 as usize, intrinsics.get_resolution().0 as usize);
    let mut prev_img = na::DMatrix::<u8>::zeros(intrinsics.get_resolution().1 as usize, intrinsics.get_resolution().0 as usize);

    for (i, assoc) in associations.iter().enumerate() { 
        let (w, h, depth_map_vec_u16) = dataset::read_png_16bits(&assoc.depth_file_path)?;
        let cur_depth_map = DMatrix::from_row_slice(h, w, depth_map_vec_u16.as_slice());
        // println!("depth image width {w} height {h}");
        let cur_img = dataset::matrix_from_image(image::open(&assoc.color_file_path)?.to_luma());
        // println!("color image width {} height {}", cur_img.shape().1, cur_img.shape().0);

        let depth_nd = nd::Array::from_shape_vec((h, w), depth_map_vec_u16.to_vec()).unwrap();
        let depth_image = rerun::DepthImage::try_from(depth_nd.clone())?.with_meter(dataset::DEPTH_SCALE);
        rec.log("world/camera/depth", &depth_image)?;
        
        if i == 0 {
            pixels = get_pixel_values(cur_img.ncols(), cur_img.nrows());
            continue; 
        }

        let depth_values = get_depth_values(&prev_depth_map, &pixels);
        
        let mut t2_tr_t1 = SE3::identity(); 
        let projected_points = direct_pose_estimation_single_layer(
            &prev_img, 
            &cur_img,
            &intrinsics, 
            &pixels, 
            &depth_values, 
            &mut t2_tr_t1, 
        );
        w_tr_c = w_tr_c * t2_tr_t1.inverse(); 

        let arrow = rerun::Arrows3D::from_vectors(
            [w_tr_c.rotation.euler_angles()]
            ).
            with_origins([(
                w_tr_c.translation.x,
                w_tr_c.translation.y,
                w_tr_c.translation.z,
            )]);
        rec.log("world/camera/#0", &arrow)?;

        let rgb_image = visualize_tracked_points(&prev_img, &pixels, &projected_points).unwrap(); 
        rec.log("tracks", &rerun::Image::try_from(rgb_image)?)?;

        (prev_depth_map, prev_img) = (cur_depth_map, cur_img); 
    }

    Ok(())
}

/// Open an association file and parse it into a vector of Association.
fn parse_associations<P: AsRef<Path>>(
    file_path: P,
) -> Result<Vec<dataset::Association>, Box<dyn Error>> {
    let file = fs::File::open(&file_path)?;
    let mut file_reader = BufReader::new(file);
    let mut content = String::new();
    file_reader.read_to_string(&mut content)?;
    dataset::parse::associations(&content)
        .map(|v| v.iter().map(|a| abs_path(&file_path, a)).collect())
        .map_err(|s| s.into())
}

/// Transform relative images file paths into absolute ones.
fn abs_path<P: AsRef<Path>>(file_path: P, assoc: &dataset::Association) -> dataset::Association {
    let parent = file_path
        .as_ref()
        .parent()
        .expect("How can this have no parent");
    dataset::Association {
        depth_timestamp: assoc.depth_timestamp,
        depth_file_path: parent.join(&assoc.depth_file_path),
        color_timestamp: assoc.color_timestamp,
        color_file_path: parent.join(&assoc.color_file_path),
    }
}


