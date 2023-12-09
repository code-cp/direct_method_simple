use image; 
use nalgebra as na; 
use direct_method_simple as dms;

use na::DMatrix;
use std::{env, error::Error, fs, io::BufReader, io::Read, path::Path, path::PathBuf};

use dms::dataset;
use dms::track::*; 
use dms::type_aliases::*; 

fn main() -> Result<(), Box<dyn Error>> {
    let associations_file_path = "/home/sean/workspace/rgbd_dataset_freiburg1_xyz/associations.txt"; 
    let associations = parse_associations(associations_file_path)?;

    let intrinsics = dataset::INTRINSICS_FR1; 
    let mut w_tr_c = SE3::identity(); 

    let (mut prev_depth_map, mut prev_img) = read_images(&associations[0])?;
    let depth_time = associations[0].depth_timestamp;
    let img_time = associations[0].color_timestamp;

    let pixels = get_pixel_values(prev_depth_map.ncols(), prev_depth_map.nrows());

    for assoc in associations.iter().skip(1) {
        let (cur_depth_map, cur_img) = read_images(assoc)?;
        
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

/// Read a depth and color image given by an association.
fn read_images(assoc: &dataset::Association) -> Result<(DMatrix<u16>, DMatrix<u8>), Box<dyn Error>> {
    let (w, h, depth_map_vec_u16) = dataset::read_png_16bits(&assoc.depth_file_path)?;
    let depth_map = DMatrix::from_row_slice(h, w, depth_map_vec_u16.as_slice());
    let img = dataset::matrix_from_image(image::open(&assoc.color_file_path)?.to_luma());
    Ok((depth_map, img))
}

