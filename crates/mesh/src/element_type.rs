/// Finite element cell type.
///
/// Variants name the geometric shape followed by the node count.
/// Only first-order (linear) and second-order (quadratic) serendipity
/// elements are listed; higher orders require a separate `order` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
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

    /// Convert to `fem_element::ElemType` for the reference-element factory.
    pub fn to_elem_type(self) -> fem_element::lagrange::factory::ElemType {
        use fem_element::lagrange::factory::ElemType;
        match self {
            Self::Line2 | Self::Line3 => ElemType::Seg,
            Self::Tri3 | Self::Tri6 => ElemType::Tri,
            Self::Tet4 | Self::Tet10 => ElemType::Tet,
            Self::Quad4 | Self::Quad8 | Self::Quad9 => ElemType::Quad,
            Self::Hex8 | Self::Hex20 | Self::Hex27 => ElemType::Hex,
            Self::Prism6 | Self::Prism15 | Self::Prism18 => ElemType::Prism,
            Self::Pyramid5 | Self::Pyramid13 => ElemType::Pyramid,
            _ => panic!("to_elem_type: unsupported {self:?}"),
        }
    }

    /// Return the reference element for this mesh type and polynomial order.
    ///
    /// Wraps `fem_element::ref_elem()` with the mesh-side `ElementType`.
    pub fn ref_elem(self, order: u8) -> Box<dyn fem_element::ReferenceElement> {
        fem_element::ref_elem(self.to_elem_type(), order)
    }

    /// Map GMSH element type integer to `ElementType`.
    ///
    /// Codes follow the Gmsh .msh file format specification (node counts in
    /// parentheses): 1 line (2), 2 tri (3), 3 quad (4), 4 tet (4), 5 hex (8),
    /// 6 prism (6), 7 pyramid (5), 8 line2 (3), 9 tri2 (6), 10 quad2 (9),
    /// 11 tet2 (10), 12 hex2 (27), 13 prism2 (18), 15 point (1), 16 quad
    /// serendipity (8), 17 hex serendipity (20), 18 prism (15), 19 pyramid
    /// (13).  MFEM 4.10 (`mesh/gmsh.cpp` `GmshReader::types`) accepts the
    /// complete order-2 codes 10/11/12/13 but rejects the serendipity codes
    /// 16/17/18/19; fem-rs has native Quad8/Prism15 shapes, so 16 and 18 are
    /// mapped to them (D297: code 16 was previously mislabeled Prism15 and
    /// code 18 was missing).
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
            10 => Some(Self::Quad9),
            11 => Some(Self::Tet10),
            12 => Some(Self::Hex27),
            13 => Some(Self::Prism18),
            15 => Some(Self::Point1),
            16 => Some(Self::Quad8),
            17 => Some(Self::Hex20),
            18 => Some(Self::Prism15),
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

    #[test]
    fn gmsh_second_order_codes() {
        // D297: Gmsh type codes for second-order elements.  Code 16 is the
        // 8-node serendipity quadrangle (was mislabeled Prism15), code 18 is
        // the 15-node prism (was missing); 10/13 are the complete order-2
        // quad/prism (also were missing).
        assert_eq!(ElementType::from_gmsh_type(16), Some(ElementType::Quad8));
        assert_eq!(ElementType::from_gmsh_type(18), Some(ElementType::Prism15));
        assert_eq!(ElementType::from_gmsh_type(10), Some(ElementType::Quad9));
        assert_eq!(ElementType::from_gmsh_type(13), Some(ElementType::Prism18));
        // node-count consistency for every mapped code
        for (code, expect_npe) in [
            (1, 2), (2, 3), (3, 4), (4, 4), (5, 8), (6, 6), (7, 5), (8, 3),
            (9, 6), (10, 9), (11, 10), (12, 27), (13, 18), (15, 1), (16, 8),
            (17, 20), (18, 15), (19, 13),
        ] {
            let t = ElementType::from_gmsh_type(code).expect("mapped");
            assert_eq!(
                t.nodes_per_element(),
                expect_npe,
                "gmsh code {code} -> {t:?}"
            );
        }
    }
}
