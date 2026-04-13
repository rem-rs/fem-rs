/// Physical group tag assigned to a boundary face (edge in 2-D, face in 3-D).
///
/// Tags match the GMSH physical group numbers defined in the `.geo` file.
/// Tag 0 means "untagged" (interior face or boundary without a label).
/// Negative tags are reserved by fem-rs for internal use.
pub type BoundaryTag = i32;

/// Assign a human-readable name to a physical group.
#[derive(Debug, Clone)]
pub struct PhysicalGroup {
    /// Topological dimension of the group (0–3).
    pub dim: u8,
    /// GMSH tag number.
    pub tag: BoundaryTag,
    /// Name as defined in the `.geo` file.
    pub name: String,
}

/// A named collection of element/boundary tags.
///
/// This is a lightweight baseline for MFEM-like named attribute sets, used
/// to bind human-readable region names to numeric tag groups.
#[derive(Debug, Clone, Default)]
pub struct NamedAttributeSet {
    pub name: String,
    pub element_tags: Vec<i32>,
    pub boundary_tags: Vec<BoundaryTag>,
}

impl NamedAttributeSet {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            element_tags: Vec::new(),
            boundary_tags: Vec::new(),
        }
    }

    pub fn with_element_tags(mut self, tags: impl IntoIterator<Item = i32>) -> Self {
        self.element_tags = tags.into_iter().collect();
        self.element_tags.sort_unstable();
        self.element_tags.dedup();
        self
    }

    pub fn with_boundary_tags(mut self, tags: impl IntoIterator<Item = BoundaryTag>) -> Self {
        self.boundary_tags = tags.into_iter().collect();
        self.boundary_tags.sort_unstable();
        self.boundary_tags.dedup();
        self
    }

    pub fn has_element_tag(&self, tag: i32) -> bool {
        self.element_tags.binary_search(&tag).is_ok()
    }

    pub fn has_boundary_tag(&self, tag: BoundaryTag) -> bool {
        self.boundary_tags.binary_search(&tag).is_ok()
    }
}

/// Registry for named attribute sets.
#[derive(Debug, Clone, Default)]
pub struct NamedAttributeRegistry {
    sets: std::collections::HashMap<String, NamedAttributeSet>,
}

impl NamedAttributeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, set: NamedAttributeSet) {
        self.sets.insert(set.name.clone(), set);
    }

    pub fn get(&self, name: &str) -> Option<&NamedAttributeSet> {
        self.sets.get(name)
    }

    pub fn names(&self) -> Vec<&str> {
        let mut out: Vec<&str> = self.sets.keys().map(|s| s.as_str()).collect();
        out.sort_unstable();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_attribute_set_dedups_and_queries() {
        let s = NamedAttributeSet::new("conductors")
            .with_element_tags([2, 2, 7, 5])
            .with_boundary_tags([11, 3, 11]);
        assert!(s.has_element_tag(7));
        assert!(!s.has_element_tag(9));
        assert!(s.has_boundary_tag(3));
        assert!(!s.has_boundary_tag(99));
    }

    #[test]
    fn named_attribute_registry_insert_and_lookup() {
        let mut r = NamedAttributeRegistry::new();
        r.insert(
            NamedAttributeSet::new("pec")
                .with_boundary_tags([1, 4]),
        );
        let names = r.names();
        assert_eq!(names, vec!["pec"]);
        let pec = r.get("pec").expect("missing pec set");
        assert!(pec.has_boundary_tag(4));
    }
}
