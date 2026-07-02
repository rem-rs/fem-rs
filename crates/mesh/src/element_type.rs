/// Finite element cell type.
///
/// Variants name the geometric shape followed by the node count.
/// Only first-order (linear) and second-order (quadratic) serendipity
/// elements are listed; higher orders require a separate `order` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// A single node (used for point physical groups in GMSH).
    Point1,
    /// 2-node line segment.
    Line2,
    /// 3-node line segment (quadratic).
    Line3,
    /// 3-node linear triangle.
    Tri3,
    /// 6-node quadratic triangle.
    Tri6,
    /// 4-node bilinear quadrilateral.
    Quad4,
    /// 8-node serendipity quadrilateral.
    Quad8,
    /// 9-node biquadratic tensor-product quadrilateral (Q2, full tensor).
    Quad9,
    /// 4-node linear tetrahedron.
    Tet4,
    /// 10-node quadratic tetrahedron.
    Tet10,
    /// 8-node trilinear hexahedron.
    Hex8,
    /// 20-node serendipity hexahedron.
    Hex20,
    /// 27-node full quadratic hexahedron (Q2, with face and body centers).
    Hex27,
    /// 6-node linear triangular prism.
    Prism6,
    /// 15-node quadratic (complete) triangular prism.
    Prism15,
    /// 18-node quadratic (serendipity) triangular prism.
    Prism18,
    /// 5-node linear pyramid.
    Pyramid5,
    /// 13-node quadratic pyramid.
    Pyramid13,
    /// Variable-node polygon (used by VEM, node count per element varies).
    Polygon,
}

impl ElementType {
    /// Number of nodes per element.
    pub const fn nodes_per_element(self) -> usize {
        match self {
            Self::Point1    =>  1,
            Self::Line2     =>  2,
            Self::Line3     =>  3,
            Self::Tri3      =>  3,
            Self::Tri6      =>  6,
            Self::Quad4     =>  4,
            Self::Quad8     =>  8,
            Self::Quad9     =>  9,
            Self::Tet4      =>  4,
            Self::Tet10     => 10,
            Self::Hex8      =>  8,
            Self::Hex20     => 20,
            Self::Hex27     => 27,
            Self::Prism6    =>  6,
            Self::Prism15   => 15,
            Self::Prism18   => 18,
            Self::Pyramid5  =>  5,
            Self::Pyramid13 => 13,
            Self::Polygon   => 0, // variable per element
        }
    }

    /// Topological dimension of the element (0 = point, 1 = edge, 2 = face, 3 = cell).
    pub const fn dim(self) -> u8 {
        match self {
            Self::Point1                        => 0,
            Self::Line2 | Self::Line3           => 1,
            Self::Tri3  | Self::Tri6
          | Self::Quad4 | Self::Quad8 | Self::Quad9
          | Self::Polygon                             => 2,
            Self::Tet4  | Self::Tet10
          | Self::Hex8  | Self::Hex20 | Self::Hex27
          | Self::Prism6 | Self::Prism15 | Self::Prism18
          | Self::Pyramid5 | Self::Pyramid13        => 3,
        }
    }

    /// Map GMSH element type integer to `ElementType`.
    ///
    /// Returns `None` for unsupported or unknown type codes.
    pub fn from_gmsh_type(code: i32) -> Option<Self> {
        match code {
             1 => Some(Self::Line2),
             2 => Some(Self::Tri3),
             3 => Some(Self::Quad4),
             4 => Some(Self::Tet4),
             5 => Some(Self::Hex8),
             6 => Some(Self::Prism6),
             7 => Some(Self::Pyramid5),
             8 => Some(Self::Line3),
             9 => Some(Self::Tri6),
            11 => Some(Self::Tet10),
            15 => Some(Self::Point1),
            16 => Some(Self::Prism15),
             17 => Some(Self::Hex20),
             12 => Some(Self::Hex27),
             19 => Some(Self::Pyramid13),
            _  => None,
        }
    }

    /// Boundary element type for this element (one dimension lower).
    ///
    /// Returns `None` for 0-D elements.
    pub const fn boundary_type(self) -> Option<Self> {
        match self {
            Self::Tri3  | Self::Tri6  => Some(Self::Line2),
            Self::Quad4 | Self::Quad8 | Self::Quad9 => Some(Self::Line3),
            Self::Tet4  | Self::Tet10 => Some(Self::Tri3),
            Self::Hex8  | Self::Hex20 | Self::Hex27 => Some(Self::Quad4),
            Self::Prism6 | Self::Prism15 | Self::Prism18 => Some(Self::Tri3),
            Self::Pyramid5 | Self::Pyramid13 => Some(Self::Tri3),
            Self::Line2 | Self::Line3 => Some(Self::Point1),
            _                         => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_per_element_tri3() {
        assert_eq!(ElementType::Tri3.nodes_per_element(), 3);
    }

    #[test]
    fn dim_tet4() {
        assert_eq!(ElementType::Tet4.dim(), 3);
    }

    #[test]
    fn gmsh_roundtrip() {
        assert_eq!(ElementType::from_gmsh_type(2), Some(ElementType::Tri3));
        assert_eq!(ElementType::from_gmsh_type(4), Some(ElementType::Tet4));
        assert_eq!(ElementType::from_gmsh_type(99), None);
    }
}
