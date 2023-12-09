use nalgebra as na; 
use std::path::PathBuf; 

use crate::camera::Intrinsics; 
use crate::type_aliases::{Float, SE3}; 

pub const DEPTH_SCALE: Float = 5000.0;

#[allow(clippy::excessive_precision)]
pub const INTRINSICS_FR1: Intrinsics = Intrinsics {
    principal_point: (318.643_040, 255.313_989),
    focal: (517.306_408, 516.469_215),
};

#[derive(Debug)]
pub struct Frame {
    pub timestamp: f64, 
    pub w_tr_c: SE3, 
}

#[derive(Debug)]
pub struct Association {
    /// Timestamp of the depth image.
    pub depth_timestamp: f64,
    /// File path of the depth image.
    pub depth_file_path: PathBuf,
    /// Timestamp of the color image.
    pub color_timestamp: f64,
    /// File path of the color image.
    pub color_file_path: PathBuf,
}

impl std::string::ToString for Frame {
    fn to_string(&self) -> String {
        let t = self.w_tr_c.translation.vector; 
        let q = self.w_tr_c.rotation.into_inner().coords; 
        format!(
            "{} {} {} {} {} {} {} {}",
            self.timestamp, t.x, t.y, t.z, q.x, q.y, q.z, q.w
        )
    }
}

/// ref https://github.com/mpizenberg/visual-odometry-rs/tree/master
/// Parse useful files (trajectories, associations, ...) in a dataset using the TUM RGB-D format.
pub mod parse {
    use super::*;
    use nom::{
        alt, anychar, do_parse, double, float, is_not, many0, map, named, space, tag,
        types::CompleteStr,
    };

    /// Parse an association file into a vector of `Association`.
    pub fn associations(file_content: &str) -> Result<Vec<Association>, String> {
        multi_line(association_line, file_content)
    }

    /// Parse a trajectory file into a vector of `Frame`.
    pub fn trajectory(file_content: &str) -> Result<Vec<Frame>, String> {
        multi_line(trajectory_line, file_content)
    }

    fn multi_line<F, T>(line_parser: F, file_content: &str) -> Result<Vec<T>, String>
    where
        F: Fn(CompleteStr) -> nom::IResult<CompleteStr, Option<T>>,
    {
        let mut vec_data = Vec::new();
        for line in file_content.lines() {
            match line_parser(CompleteStr(line)) {
                Ok((_, Some(data))) => vec_data.push(data),
                Ok(_) => (),
                Err(_) => return Err("Parsing error".to_string()),
            }
        }
        Ok(vec_data)
    }

    // nom parsers #############################################################

    // Associations --------------------

    // Association line is either a comment or two timestamps and file paths.
    named!(association_line<CompleteStr, Option<Association> >,
        alt!( map!(comment, |_| None) | map!(association, Some) )
    );

    // Parse an association of depth and color timestamps and file paths.
    named!(association<CompleteStr, Association>,
        do_parse!(
            depth_timestamp: double >> space >>
            depth_file_path: path >> space >>
            color_timestamp: double >> space >>
            color_file_path: path >>
            (Association { depth_timestamp, depth_file_path, color_timestamp, color_file_path })
        )
    );

    named!(path<CompleteStr, PathBuf>,
        map!(is_not!(" \t\r\n"), |s| PathBuf::from(*s))
    );

    // Trajectory ----------------------

    // Trajectory line is either a comment or a frame timestamp and pose.
    named!(trajectory_line<CompleteStr, Option<Frame> >,
        alt!( map!(comment, |_| None) | map!(frame, Some) )
    );

    // Parse a comment.
    named!(comment<CompleteStr,()>,
        do_parse!( tag!("#") >> many0!(anychar) >> ())
    );

    // Parse a frame.
    named!(frame<CompleteStr, Frame>,
        do_parse!(
            t: double >> space >>
            p: pose >>
            (Frame { timestamp: t, w_tr_c: p })
        )
    );

    // Parse extrinsics camera parameters.
    named!(pose<CompleteStr, SE3 >,
        do_parse!(
            t: translation >> space >>
            r: rotation >>
            (SE3::from_parts(t, r))
        )
    );

    // Parse components of a translation.
    named!(translation<CompleteStr, na::Translation3<Float> >,
        do_parse!(
            x: float >> space >>
            y: float >> space >>
            z: float >>
            (na::Translation3::new(x, y, z))
        )
    );

    // Parse components of a unit quaternion describing the rotation.
    named!(rotation<CompleteStr, na::UnitQuaternion<Float> >,
        do_parse!(
            qx: float >> space >>
            qy: float >> space >>
            qz: float >> space >>
            qw: float >>
            (na::UnitQuaternion::from_quaternion(na::Quaternion::new(qw, qx, qy, qz)))
        )
    );

} // pub mod parse

use std::{self, fs::File, io::Cursor, path::Path};
use byteorder::{BigEndian, ReadBytesExt};
use png::{self, HasParameters};

/// Read a 16 bit gray png image from a file.
pub fn read_png_16bits<P: AsRef<Path>>(
    file_path: P,
) -> Result<(usize, usize, Vec<u16>), png::DecodingError> {
    // Load 16 bits PNG depth image.
    let img_file = File::open(file_path)?;
    let mut decoder = png::Decoder::new(img_file);
    // Use the IDENTITY transformation because by default
    // it will use STRIP_16 which only keep 8 bits.
    // See also SWAP_ENDIAN that might be useful
    //   (but seems not possible to use according to documentation).
    decoder.set(png::Transformations::IDENTITY);
    let (info, mut reader) = decoder.read_info()?;
    let mut buffer = vec![0; info.buffer_size()];
    reader.next_frame(&mut buffer)?;

    // Transform buffer into 16 bits slice.
    // if cfg!(target_endian = "big") ...
    let mut buffer_u16 = vec![0; (info.width * info.height) as usize];
    let mut buffer_cursor = Cursor::new(buffer);
    buffer_cursor.read_u16_into::<BigEndian>(&mut buffer_u16)?;

    // Return u16 buffer.
    Ok((info.width as usize, info.height as usize, buffer_u16))
}

use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use nalgebra::DMatrix;

/// Convert an `u8` matrix into a `GrayImage`.
/// Inverse operation of `matrix_from_image`.
///
/// Performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn image_from_matrix(mat: &DMatrix<u8>) -> GrayImage {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = GrayImage::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

/// Convert a `GrayImage` into an `u8` matrix.
/// Inverse operation of `image_from_matrix`.
pub fn matrix_from_image(img: GrayImage) -> DMatrix<u8> {
    let (width, height) = img.dimensions();
    DMatrix::from_row_slice(height as usize, width as usize, &img.into_raw())
}