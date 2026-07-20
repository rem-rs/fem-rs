//! Physical units system for fem-rs.
//!
//! Provides:
//! - [`SiUnit`] — 7-base-dimension unit representation with i8 powers
//! - [`PhysicalQuantity`] — a value tagged with its unit
//! - Arithmetic with dimensional consistency checking
//! - [`UnitSystem`] — user-level configuration for display/conversion
//!
//! # Example
//!
//! ```rust
//! use fem_core::units::*;
//!
//! let length = PhysicalQuantity::new(1.0, SiUnit::METER);
//! let time   = PhysicalQuantity::new(2.0, SiUnit::SECOND);
//! let velocity = length / time;
//! assert_eq!(velocity.unit, SiUnit::METER_PER_SECOND);
//! assert!((velocity.value - 0.5).abs() < 1e-15);
//! ```

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

// ─── SiUnit — 7-base-dimension unit ──────────────────────────────────────────

/// SI unit represented by integer powers of the 7 base dimensions.
///
/// Powers can be negative. Two units are compatible when they have the same
/// power tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SiUnit {
    /// Length (meter) exponent
    pub m: i8,
    /// Mass (kilogram) exponent
    pub kg: i8,
    /// Time (second) exponent
    pub s: i8,
    /// Electric current (ampere) exponent
    pub a: i8,
    /// Thermodynamic temperature (kelvin) exponent
    pub k: i8,
    /// Amount of substance (mole) exponent
    pub mol: i8,
    /// Luminous intensity (candela) exponent
    pub cd: i8,
}

impl SiUnit {
    /// The dimensionless unit (all exponents zero).
    pub const ZERO: Self = Self { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 };

    // ── Base SI units ──────────────────────────────────────────────────────

    pub const METER:      Self = Self { m: 1, ..Self::ZERO };
    pub const KILOGRAM:   Self = Self { kg: 1, ..Self::ZERO };
    pub const SECOND:     Self = Self { s: 1, ..Self::ZERO };
    pub const AMPERE:     Self = Self { a: 1, ..Self::ZERO };
    pub const KELVIN:     Self = Self { k: 1, ..Self::ZERO };
    pub const MOLE:       Self = Self { mol: 1, ..Self::ZERO };
    pub const CANDELA:    Self = Self { cd: 1, ..Self::ZERO };

    // ── Derived SI units ────────────────────────────────────────────────────

    pub const RADIAN:     Self = Self::ZERO; // dimensionless
    pub const STERADIAN:  Self = Self::ZERO; // dimensionless
    pub const HERTZ:      Self = Self { s: -1, ..Self::ZERO };
    pub const NEWTON:     Self = Self { m: 1, kg: 1, s: -2, ..Self::ZERO };
    pub const PASCAL:     Self = Self { m: -1, kg: 1, s: -2, ..Self::ZERO };
    pub const JOULE:      Self = Self { m: 2, kg: 1, s: -2, ..Self::ZERO };
    pub const WATT:       Self = Self { m: 2, kg: 1, s: -3, ..Self::ZERO };
    pub const COULOMB:    Self = Self { s: 1, a: 1, ..Self::ZERO };
    pub const VOLT:       Self = Self { m: 2, kg: 1, s: -3, a: -1, ..Self::ZERO };
    pub const FARAD:      Self = Self { m: -2, kg: -1, s: 4, a: 2, ..Self::ZERO };
    pub const OHM:        Self = Self { m: 2, kg: 1, s: -3, a: -2, ..Self::ZERO };
    pub const SIEMENS:    Self = Self { m: -2, kg: -1, s: 3, a: 2, ..Self::ZERO };
    pub const WEBER:      Self = Self { m: 2, kg: 1, s: -2, a: -1, ..Self::ZERO };
    pub const TESLA:      Self = Self { kg: 1, s: -2, a: -1, ..Self::ZERO };
    pub const HENRY:      Self = Self { m: 2, kg: 1, s: -2, a: -2, ..Self::ZERO };

    // ── Common compound units ───────────────────────────────────────────────

    pub const METER_PER_SECOND:       Self = Self { m: 1, s: -1, ..Self::ZERO };
    pub const METER_PER_SECOND2:      Self = Self { m: 1, s: -2, ..Self::ZERO };
    pub const RAD_PER_SECOND:         Self = Self { s: -1, ..Self::ZERO };
    pub const RAD_PER_SECOND2:        Self = Self { s: -2, ..Self::ZERO };
    pub const NEWTON_METER:           Self = Self { m: 2, kg: 1, s: -2, ..Self::ZERO };
    pub const WATT_PER_METER_KELVIN:  Self = Self { m: 1, kg: 1, s: -3, k: -1, ..Self::ZERO };
    pub const JOULE_PER_KILOGRAM_KELVIN: Self = Self { m: 2, s: -2, k: -1, ..Self::ZERO };
    pub const KILOGRAM_PER_METER3:    Self = Self { m: -3, kg: 1, ..Self::ZERO };
    pub const PASCAL_SECOND:          Self = Self { m: -1, kg: 1, s: -1, ..Self::ZERO };

    /// Check if this unit is dimensionless (all exponents zero).
    pub fn is_dimensionless(&self) -> bool {
        *self == Self::ZERO
    }

    /// Multiply two units (add exponents).
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            m: self.m + rhs.m,
            kg: self.kg + rhs.kg,
            s: self.s + rhs.s,
            a: self.a + rhs.a,
            k: self.k + rhs.k,
            mol: self.mol + rhs.mol,
            cd: self.cd + rhs.cd,
        }
    }

    /// Divide two units (subtract exponents).
    pub fn div(self, rhs: Self) -> Self {
        Self {
            m: self.m - rhs.m,
            kg: self.kg - rhs.kg,
            s: self.s - rhs.s,
            a: self.a - rhs.a,
            k: self.k - rhs.k,
            mol: self.mol - rhs.mol,
            cd: self.cd - rhs.cd,
        }
    }

    /// Raise unit to an integer power.
    pub fn powi(self, n: i8) -> Self {
        Self {
            m: self.m * n,
            kg: self.kg * n,
            s: self.s * n,
            a: self.a * n,
            k: self.k * n,
            mol: self.mol * n,
            cd: self.cd * n,
        }
    }

    /// Square root (only valid when all exponents are even).
    pub fn sqrt(self) -> Option<Self> {
        if self.m % 2 != 0 || self.kg % 2 != 0 || self.s % 2 != 0
            || self.a % 2 != 0 || self.k % 2 != 0 || self.mol % 2 != 0 || self.cd % 2 != 0
        {
            return None;
        }
        Some(Self {
            m: self.m / 2,
            kg: self.kg / 2,
            s: self.s / 2,
            a: self.a / 2,
            k: self.k / 2,
            mol: self.mol / 2,
            cd: self.cd / 2,
        })
    }
}

impl fmt::Display for SiUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "—");
        }
        let parts = [
            ("kg", self.kg), ("m", self.m), ("s", self.s),
            ("A", self.a), ("K", self.k), ("mol", self.mol), ("cd", self.cd),
        ];
        let mut first = true;
        for (name, exp) in parts {
            if exp != 0 {
                if !first { write!(f, "·")?; }
                write!(f, "{}", format_exp(name, exp))?;
                first = false;
            }
        }
        Ok(())
    }
}

// ─── PhysicalQuantity — value + unit ──────────────────────────────────────────

/// A floating-point value tagged with its physical unit.
#[derive(Debug, Clone, Copy)]
pub struct PhysicalQuantity {
    /// Numerical value (in SI base units).
    pub value: f64,
    /// The unit of this quantity.
    pub unit: SiUnit,
}

impl PhysicalQuantity {
    /// Create a new quantity with the given value and unit.
    pub const fn new(value: f64, unit: SiUnit) -> Self {
        Self { value, unit }
    }

    /// Create a dimensionless quantity.
    pub fn dimensionless(value: f64) -> Self {
        Self { value, unit: SiUnit::ZERO }
    }

    /// Convert to a different compatible unit using the given scale factor.
    ///
    /// `factor` is the multiplier to convert *from* the target unit *to* SI:
    /// `value_in_si = value_in_target * factor`.
    ///
    /// For example, to convert from mm to m: `factor = 1e-3`.
    pub fn convert_to(&self, factor: f64) -> f64 {
        self.value / factor
    }

    /// Apply a scalar function to the value (for dimensionless scaling).
    pub fn map<F: FnOnce(f64) -> f64>(self, f: F) -> Self {
        Self { value: f(self.value), unit: self.unit }
    }
}

// ── Arithmetic ───────────────────────────────────────────────────────────────

/// Error returned when arithmetic would produce an invalid unit operation.
#[derive(Debug, Clone)]
pub enum UnitError {
    /// Addition/subtraction requires identical units.
    IncompatibleUnits { left: SiUnit, right: SiUnit, op: &'static str },
    /// Square root of a unit with odd exponents.
    NonEvenExponents(SiUnit),
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleUnits { left, right, op } => {
                write!(f, "cannot {op} quantities with units {left} and {right}")
            }
            Self::NonEvenExponents(u) => {
                write!(f, "cannot sqrt unit {u}: not all exponents are even")
            }
        }
    }
}

impl std::error::Error for UnitError {}

// Debug builds check unit compatibility; release builds skip for performance.
macro_rules! check_units {
    ($cond:expr, $err:expr) => {
        debug_assert!($cond, "{}", $err);
    };
}

impl Add for PhysicalQuantity {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        check_units!(
            self.unit == rhs.unit,
            UnitError::IncompatibleUnits { left: self.unit, right: rhs.unit, op: "add" }
        );
        Self { value: self.value + rhs.value, unit: self.unit }
    }
}

impl Sub for PhysicalQuantity {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        check_units!(
            self.unit == rhs.unit,
            UnitError::IncompatibleUnits { left: self.unit, right: rhs.unit, op: "subtract" }
        );
        Self { value: self.value - rhs.value, unit: self.unit }
    }
}

impl Mul for PhysicalQuantity {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            unit: self.unit.mul(rhs.unit),
        }
    }
}

impl Div for PhysicalQuantity {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            value: self.value / rhs.value,
            unit: self.unit.div(rhs.unit),
        }
    }
}

// Scalar * PhysicalQuantity
impl Mul<PhysicalQuantity> for f64 {
    type Output = PhysicalQuantity;
    fn mul(self, rhs: PhysicalQuantity) -> PhysicalQuantity {
        PhysicalQuantity { value: self * rhs.value, unit: rhs.unit }
    }
}

impl Mul<f64> for PhysicalQuantity {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self { value: self.value * rhs, unit: self.unit }
    }
}

impl Div<f64> for PhysicalQuantity {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self { value: self.value / rhs, unit: self.unit }
    }
}

// ─── UnitSystem — user-level configuration ────────────────────────────────────

/// Engineering unit system for user input/output.
///
/// Defines the base units used in a particular analysis. All internal
/// computation uses SI, but inputs and outputs can use a different system.
#[derive(Debug, Clone)]
pub struct UnitSystem {
    /// Length unit name and scale (1 unit = ? meters)
    pub length: (&'static str, f64),
    /// Mass unit name and scale (1 unit = ? kg)
    pub mass: (&'static str, f64),
    /// Time unit name and scale (1 unit = ? seconds)
    pub time: (&'static str, f64),
    /// Temperature unit name and offset/scale
    pub temperature: TemperatureUnit,
}

/// Temperature measurement convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureUnit {
    Kelvin,
    Celsius,
}

impl UnitSystem {
    /// SI: m, kg, s, K
    pub const SI: Self = Self {
        length: ("m", 1.0),
        mass: ("kg", 1.0),
        time: ("s", 1.0),
        temperature: TemperatureUnit::Kelvin,
    };

    /// mm, mg, s (commonly used in small-scale MEMS):
    pub const MM_MG_S: Self = Self {
        length: ("mm", 1e-3),
        mass: ("mg", 1e-6),
        time: ("s", 1.0),
        temperature: TemperatureUnit::Kelvin,
    };

    /// mm, ton, s (commonly used in Abaqus structural analysis):
    /// 1 ton = 1000 kg, 1 mm = 0.001 m → derived unit: N = 10⁻³·10³/1² = 1 N
    /// This is convenient: stress in N/mm² = MPa.
    pub const MM_TON_S: Self = Self {
        length: ("mm", 1e-3),
        mass: ("ton", 1e3),
        time: ("s", 1.0),
        temperature: TemperatureUnit::Kelvin,
    };

    /// mm, kg, ms (millimeter-kilogram-millisecond — convenient for dynamics)
    pub const MM_KG_MS: Self = Self {
        length: ("mm", 1e-3),
        mass: ("kg", 1.0),
        time: ("ms", 1e-3),
        temperature: TemperatureUnit::Kelvin,
    };

    /// Describe the derived unit for a given SI unit in this system.
    ///
    /// Returns the scale factor to convert *from* this system's representation
    /// *to* SI, plus a human-readable label.
    ///
    /// `factor` satisfies: `value_si = value_in_system * factor`
    pub fn derived_unit(&self, unit: SiUnit) -> (f64, String) {
        let l = self.length.1;  // length scale
        let m = self.mass.1;    // mass scale
        let t = self.time.1;    // time scale

        // SI base: [L^m_l * M^m_m * T^m_t * K^m_k * A^m_a * ...]
        let scale = l.powi(unit.m as i32) * m.powi(unit.kg as i32) * t.powi(unit.s as i32);
        let label = self.format_unit(unit);
        (scale, label)
    }

    /// Format a unit symbol using this system's base names.
    fn format_unit(&self, unit: SiUnit) -> String {
        if unit.is_dimensionless() {
            return "—".to_string();
        }
        let bases = [
            (self.mass.0, unit.kg),
            (self.length.0, unit.m),
            (self.time.0, unit.s),
        ];
        let mut parts = Vec::new();
        for (name, exp) in bases {
            if exp != 0 {
                parts.push(format_exp(name, exp));
            }
        }
        parts.join("·")
    }
}

// ─── Conversion factors ──────────────────────────────────────────────────────

/// Common conversion factors from non-SI units to SI.
pub mod convert {
    // Length
    pub const INCH_TO_M: f64 = 0.0254;
    pub const FT_TO_M: f64 = 0.3048;
    pub const MM_TO_M: f64 = 1e-3;
    pub const CM_TO_M: f64 = 1e-2;
    pub const MICRON_TO_M: f64 = 1e-6;

    // Mass
    pub const LB_TO_KG: f64 = 0.45359237;
    pub const G_TO_KG: f64 = 1e-3;
    pub const TON_TO_KG: f64 = 1e3;

    // Force
    pub const LBF_TO_N: f64 = 4.4482216152605;
    pub const KGF_TO_N: f64 = 9.80665;

    // Pressure
    pub const PSI_TO_PA: f64 = 6894.7572931783;
    pub const BAR_TO_PA: f64 = 1e5;
    pub const ATM_TO_PA: f64 = 101325.0;
    pub const MPA_TO_PA: f64 = 1e6;
    pub const GPA_TO_PA: f64 = 1e9;

    // Energy
    pub const CAL_TO_J: f64 = 4.184;
    pub const EV_TO_J: f64 = 1.602176634e-19;

    // Temperature
    pub fn celsius_to_kelvin(c: f64) -> f64 { c + 273.15 }
    pub fn kelvin_to_celsius(k: f64) -> f64 { k - 273.15 }

    // Time
    pub const MIN_TO_S: f64 = 60.0;
    pub const HR_TO_S: f64 = 3600.0;
    pub const MS_TO_S: f64 = 1e-3;
}

// ─── Physical constants with units ───────────────────────────────────────────

/// Commonly used physical constants, each tagged with its SI unit.
pub mod constants {
    use super::*;

    /// Stefan–Boltzmann constant: 5.670374419 × 10⁻⁸ W·m⁻²·K⁻⁴
    pub const STEFAN_BOLTZMANN: PhysicalQuantity = PhysicalQuantity::new(
        5.670_374_419e-8,
        SiUnit { m: 0, kg: 1, s: -3, a: 0, k: -4, mol: 0, cd: 0 }, // W/m²K⁴
    );

    /// Standard gravity: 9.80665 m/s²
    pub const GRAVITY: PhysicalQuantity = PhysicalQuantity::new(
        9.806_65,
        SiUnit::METER_PER_SECOND2,
    );

    /// Vacuum permittivity: 8.854187817 × 10⁻¹² F/m
    pub const EPSILON_0: PhysicalQuantity = PhysicalQuantity::new(
        8.854_187_817e-12,
        SiUnit { m: -3, kg: -1, s: 4, a: 2, k: 0, mol: 0, cd: 0 }, // F/m = kg⁻¹·m⁻³·s⁴·A²
    );

    /// Vacuum permeability: 4π × 10⁻⁷ H/m = 1.256637061 × 10⁻⁶ H/m
    pub const MU_0: PhysicalQuantity = PhysicalQuantity::new(
        1.256_637_061_435_917_3e-6,
        SiUnit { m: 1, kg: 1, s: -2, a: -2, k: 0, mol: 0, cd: 0 }, // H/m = kg·m·s⁻²·A⁻²
    );

    /// Speed of light in vacuum: 299792458 m/s
    pub const SPEED_OF_LIGHT: PhysicalQuantity = PhysicalQuantity::new(
        299_792_458.0,
        SiUnit::METER_PER_SECOND,
    );

    /// Boltzmann constant: 1.380649 × 10⁻²³ J/K
    pub const BOLTZMANN: PhysicalQuantity = PhysicalQuantity::new(
        1.380_649e-23,
        SiUnit { m: 2, kg: 1, s: -2, k: -1, mol: 0, a: 0, cd: 0 }, // J/K
    );

    /// Universal gas constant: 8.314462618 J/(mol·K)
    pub const GAS_CONSTANT: PhysicalQuantity = PhysicalQuantity::new(
        8.314_462_618,
        SiUnit { m: 2, kg: 1, s: -2, k: -1, mol: -1, a: 0, cd: 0 },
    );
}

// ─── Helper: format exponent with superscript characters ─────────────────────

fn format_exp(name: &str, exp: i8) -> String {
    if exp == 1 {
        name.to_string()
    } else if exp == 2 {
        format!("{name}²")
    } else if exp == 3 {
        format!("{name}³")
    } else if exp == -1 {
        format!("{name}⁻¹")
    } else if exp == -2 {
        format!("{name}⁻²")
    } else if exp == -3 {
        format!("{name}⁻³")
    } else if exp < 0 {
        format!("{name}{}", exp.to_string().replace('-', "⁻"))
    } else {
        format!("{name}{}", exp.to_string())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_equality() {
        assert_eq!(SiUnit::METER, SiUnit::METER);
        assert_ne!(SiUnit::METER, SiUnit::SECOND);
    }

    #[test]
    fn unit_multiplication() {
        let force = SiUnit::METER.mul(SiUnit::KILOGRAM).div(SiUnit::SECOND.powi(2));
        assert_eq!(force, SiUnit::NEWTON);
    }

    #[test]
    fn unit_division() {
        let velocity = SiUnit::METER.div(SiUnit::SECOND);
        assert_eq!(velocity, SiUnit::METER_PER_SECOND);
    }

    #[test]
    fn unit_power() {
        let area = SiUnit::METER.powi(2);
        assert_eq!(area, SiUnit { m: 2, ..SiUnit::ZERO });
    }

    #[test]
    fn unit_sqrt() {
        let area = SiUnit::METER.powi(2);
        assert_eq!(area.sqrt(), Some(SiUnit::METER));
        assert_eq!(SiUnit::METER.sqrt(), None); // odd exponent
    }

    #[test]
    fn dimensionless() {
        assert!(SiUnit::ZERO.is_dimensionless());
        assert!(SiUnit::RADIAN.is_dimensionless());
    }

    #[test]
    fn physical_quantity_add() {
        let a = PhysicalQuantity::new(3.0, SiUnit::METER);
        let b = PhysicalQuantity::new(2.0, SiUnit::METER);
        assert!((a + b).value - 5.0 < 1e-15);
    }

    #[test]
    #[should_panic(expected = "cannot add")]
    fn physical_quantity_add_incompatible() {
        let a = PhysicalQuantity::new(3.0, SiUnit::METER);
        let b = PhysicalQuantity::new(2.0, SiUnit::SECOND);
        let _ = a + b; // panics in debug mode
    }

    #[test]
    fn physical_quantity_mul() {
        let force = PhysicalQuantity::new(10.0, SiUnit::NEWTON);
        let dist = PhysicalQuantity::new(2.0, SiUnit::METER);
        let work = force * dist;
        assert!((work.value - 20.0).abs() < 1e-15);
        assert_eq!(work.unit, SiUnit::JOULE);
    }

    #[test]
    fn physical_quantity_div() {
        let dist = PhysicalQuantity::new(100.0, SiUnit::METER);
        let time = PhysicalQuantity::new(10.0, SiUnit::SECOND);
        let vel = dist / time;
        assert!((vel.value - 10.0).abs() < 1e-15);
        assert_eq!(vel.unit, SiUnit::METER_PER_SECOND);
    }

    #[test]
    fn scalar_mul_quantity() {
        let q = 2.0 * PhysicalQuantity::new(5.0, SiUnit::METER);
        assert!((q.value - 10.0).abs() < 1e-15);
        assert_eq!(q.unit, SiUnit::METER);
    }

    #[test]
    fn quantity_mul_scalar() {
        let q = PhysicalQuantity::new(5.0, SiUnit::METER) * 2.0;
        assert!((q.value - 10.0).abs() < 1e-15);
    }

    #[test]
    fn convert_to() {
        let len = PhysicalQuantity::new(1.0, SiUnit::METER);
        let in_mm = len.convert_to(convert::MM_TO_M);
        assert!((in_mm - 1000.0).abs() < 1e-12);
    }

    #[test]
    fn unit_system_si() {
        let sys = UnitSystem::SI;
        assert_eq!(sys.length.0, "m");
        assert!((sys.length.1 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn unit_system_mm_ton_s() {
        let sys = UnitSystem::MM_TON_S;
        let (scale, label) = sys.derived_unit(SiUnit::PASCAL);
        // 1 N/mm² = 1 MPa = 1e6 Pa
        // In mm-ton-s: force = (ton*mm/s²) = 1e3 kg * 1e-3 m / s² = 1 kg·m/s² = 1 N
        // stress = N/mm² = 1 N / (1e-3 m)² = 1e6 Pa
        // So 1 (in system) = 1e6 Pa
        assert!((scale - 1e6).abs() < 1e-9, "scale = {scale}");
        assert_eq!(label, "ton·mm⁻¹·s⁻²");
    }

    #[test]
    fn stefan_boltzmann_unit() {
        // σ has units W·m⁻²·K⁻⁴
        let sb = constants::STEFAN_BOLTZMANN;
        // W = kg·m²·s⁻³, so W·m⁻²·K⁻⁴ = kg·s⁻³·K⁻⁴
        let expected = SiUnit { kg: 1, s: -3, k: -4, ..SiUnit::ZERO };
        assert_eq!(sb.unit, expected);
    }

    #[test]
    fn gravity_unit() {
        let g = constants::GRAVITY;
        assert_eq!(g.unit, SiUnit::METER_PER_SECOND2);
    }

    #[test]
    fn dimensionless_map() {
        let ratio = PhysicalQuantity::dimensionless(0.5);
        let doubled = ratio.map(|x| x * 2.0);
        assert!((doubled.value - 1.0).abs() < 1e-15);
        assert!(doubled.unit.is_dimensionless());
    }

    #[test]
    fn unit_display() {
        assert_eq!(format!("{}", SiUnit::NEWTON), "kg·m·s⁻²");
        assert_eq!(format!("{}", SiUnit::ZERO), "—");
        assert_eq!(format!("{}", SiUnit::HERTZ), "s⁻¹");
    }

    #[test]
    fn derived_unit_display() {
        let sys = UnitSystem::MM_TON_S;
        let (_scale, label) = sys.derived_unit(SiUnit::NEWTON);
        // N = kg·m·s⁻² → in mm-ton-s: ton·mm·s⁻²
        assert_eq!(label, "ton·mm·s⁻²");
    }
}
