use nalgebra as na;

/// At the moment, the library is focused on f32 computation.
pub type Float = f32;

/// A point with two Float coordinates.
pub type Point2 = na::Point2<Float>;
/// A point with three Float coordinates.
pub type Point3 = na::Point3<Float>;

/// A vector with two Float coordinates.
pub type Vec2 = na::Vector2<Float>;
/// A vector with three Float coordinates.
pub type Vec3 = na::Vector3<Float>;
/// A vector with six Float coordinates.
pub type Vec6 = na::Vector6<Float>;

/// A 2x6 matrix of Floats.
pub type Mat26 = na::MatrixMN::<Float, na::U2, na::U6>;
/// A 3x3 matrix of Floats.
pub type Mat3 = na::Matrix3<Float>;
/// A 4x4 matrix of Floats.
pub type Mat4 = na::Matrix4<Float>;
/// A 6x6 matrix of Floats.
pub type Mat6 = na::Matrix6<Float>;

/// A direct 3D isometry, also known as rigid body motion.
pub type SE3 = na::Isometry3<Float>;