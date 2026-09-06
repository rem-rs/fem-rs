// 2D and 3D matrix invariants evaluators for TMOP quality metrics.
//
// Ported from MFEM's `linalg/invariants.hpp`.
// Jacobian storage is column-major (matching MFEM DenseMatrix::GetData()).

/// Evaluates 2D matrix invariants and their 1st/2nd derivatives.
#[derive(Debug, Clone)]
pub struct InvariantsEvaluator2D {
    j: [f64; 4],
    i1: f64,
    i1b: f64,
    i2b: f64,
    di1: [f64; 4],
    di1b: [f64; 4],
    di2: [f64; 4],
    di2b: [f64; 4],
    d_height: usize,
    d: Option<Vec<f64>>,
    eval_state: u32,
}

const HAVE_I1: u32 = 1;
const HAVE_I1B: u32 = 2;
const HAVE_I2B: u32 = 4;
const HAVE_DI1: u32 = 8;
const HAVE_DI1B: u32 = 16;
const HAVE_DI2: u32 = 32;
const HAVE_DI2B: u32 = 64;
const HAVE_DAJ: u32 = 128;
const HAVE_DJT: u32 = 256;

impl InvariantsEvaluator2D {
    pub fn new(jac: Option<&[f64; 4]>) -> Self {
        Self {
            j: jac.copied().unwrap_or([0.0; 4]),
            i1: 0.0,
            i1b: 0.0,
            i2b: 0.0,
            di1: [0.0; 4],
            di1b: [0.0; 4],
            di2: [0.0; 4],
            di2b: [0.0; 4],
            d_height: 0,
            d: None,
            eval_state: 0,
        }
    }

    pub fn set_jacobian(&mut self, jac: &[f64; 4]) {
        self.j = *jac;
        self.eval_state = 0;
    }

    pub fn set_derivative_matrix(&mut self, height: usize, deriv: &[f64]) {
        self.eval_state &= !(HAVE_DAJ | HAVE_DJT);
        self.d_height = height;
        let size = height * 2;
        let mut d = self.d.take().unwrap_or_default();
        d.resize(size, 0.0);
        d.copy_from_slice(&deriv[..size]);
        self.d = Some(d);
    }

    #[inline]
    fn has(&self, mask: u32) -> bool {
        self.eval_state & mask != 0
    }

    fn eval_i1(&mut self) {
        self.eval_state |= HAVE_I1;
        let j = self.j;
        self.i1 = j[0] * j[0] + j[1] * j[1] + j[2] * j[2] + j[3] * j[3];
    }

    fn eval_i1b(&mut self) {
        self.eval_state |= HAVE_I1B;
        if !self.has(HAVE_I1) { self.eval_i1(); }
        if !self.has(HAVE_I2B) { self.eval_i2b(); }
        self.i1b = self.i1 / self.i2b;
    }

    fn eval_i2b(&mut self) {
        self.eval_state |= HAVE_I2B;
        let j = self.j;
        self.i2b = j[0] * j[3] - j[1] * j[2];
    }

    fn eval_di1(&mut self) {
        self.eval_state |= HAVE_DI1;
        let j = self.j;
        self.di1[0] = 2.0 * j[0];
        self.di1[2] = 2.0 * j[2];
        self.di1[1] = 2.0 * j[1];
        self.di1[3] = 2.0 * j[3];
    }

    fn eval_di1b(&mut self) {
        self.eval_state |= HAVE_DI1B;
        if !self.has(HAVE_I2B) { self.eval_i2b(); }
        let c1 = 2.0 / self.i2b;
        let c2 = self.i1b / 2.0;
        if !self.has(HAVE_DI2B) { self.eval_di2b(); }
        let j = self.j;
        let di2b = self.di2b;
        self.di1b[0] = c1 * (j[0] - c2 * di2b[0]);
        self.di1b[1] = c1 * (j[1] - c2 * di2b[1]);
        self.di1b[2] = c1 * (j[2] - c2 * di2b[2]);
        self.di1b[3] = c1 * (j[3] - c2 * di2b[3]);
    }

    fn eval_di2(&mut self) {
        self.eval_state |= HAVE_DI2;
        if !self.has(HAVE_I2B) { self.eval_i2b(); }
        let c1 = 2.0 * self.i2b;
        if !self.has(HAVE_DI2B) { self.eval_di2b(); }
        let di2b = self.di2b;
        self.di2[0] = c1 * di2b[0];
        self.di2[1] = c1 * di2b[1];
        self.di2[2] = c1 * di2b[2];
        self.di2[3] = c1 * di2b[3];
    }

    fn eval_di2b(&mut self) {
        self.eval_state |= HAVE_DI2B;
        if !self.has(HAVE_I2B) { self.eval_i2b(); }
        let j = self.j;
        self.di2b[0] = j[3];
        self.di2b[1] = -j[2];
        self.di2b[2] = -j[1];
        self.di2b[3] = j[0];
    }

    fn eval_daj(&mut self) {
        self.eval_state |= HAVE_DAJ;
        if !self.has(HAVE_DI2B) { self.eval_di2b(); }
    }

    fn eval_djt(&mut self) {
        self.eval_state |= HAVE_DJT;
    }

    fn compute_dzt(z: &[f64], nd: usize, d: &[f64]) -> Vec<f64> {
        let mut dzt = vec![0.0; 2 * nd];
        for i in 0..nd {
            let i0 = i + nd * 0;
            let i1 = i + nd * 1;
            dzt[i0] = d[i0] * z[0] + d[i1] * z[2];
            dzt[i1] = d[i0] * z[1] + d[i1] * z[3];
        }
        dzt
    }

    pub fn get_i1(&mut self) -> f64 {
        if !self.has(HAVE_I1) { self.eval_i1(); }
        self.i1
    }

    pub fn get_i1b(&mut self) -> f64 {
        if !self.has(HAVE_I1B) { self.eval_i1b(); }
        self.i1b
    }

    pub fn get_i2(&mut self) -> f64 {
        if !self.has(HAVE_I2B) { self.eval_i2b(); }
        self.i2b * self.i2b
    }

    pub fn get_i2b(&mut self) -> f64 {
        if !self.has(HAVE_I2B) { self.eval_i2b(); }
        self.i2b
    }

    pub fn get_di1(&mut self) -> &[f64; 4] {
        if !self.has(HAVE_DI1) { self.eval_di1(); }
        &self.di1
    }

    pub fn get_di1b(&mut self) -> &[f64; 4] {
        if !self.has(HAVE_DI1B) { self.eval_di1b(); }
        &self.di1b
    }

    pub fn get_di2(&mut self) -> &[f64; 4] {
        if !self.has(HAVE_DI2) { self.eval_di2(); }
        &self.di2
    }

    pub fn get_di2b(&mut self) -> &[f64; 4] {
        if !self.has(HAVE_DI2B) { self.eval_di2b(); }
        &self.di2b
    }

    pub fn get_daj(&mut self) -> Vec<f64> {
        if !self.has(HAVE_DAJ) { self.eval_daj(); }
        let di2b = self.di2b;
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        Self::compute_dzt(&di2b, nd, &d)
    }

    pub fn get_djt(&mut self) -> Vec<f64> {
        if !self.has(HAVE_DJT) { self.eval_djt(); }
        let j = self.j;
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        Self::compute_dzt(&j, nd, &d)
    }

    pub fn get_d(&self) -> &[f64] {
        self.d.as_ref().map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn d_height(&self) -> usize {
        self.d_height
    }

    pub fn assemble_dd_i1(&mut self, w: f64, a: &mut [f64]) {
        let nd = self.d_height;
        let ah = 2 * nd;
        let a2 = 2.0 * w;
        if let Some(d) = &self.d {
            for i in 0..nd {
                let i0 = i + nd * 0;
                let i1 = i + nd * 1;
                let a_di = [a2 * d[i0], a2 * d[i1]];
                let a_ddt_ii = a_di[0] * d[i0] + a_di[1] * d[i1];
                a[i0 + ah * i0] += a_ddt_ii;
                a[i1 + ah * i1] += a_ddt_ii;
                for k in 0..i {
                    let k0 = k + nd * 0;
                    let k1 = k + nd * 1;
                    let a_ddt_ik = a_di[0] * d[k0] + a_di[1] * d[k1];
                    a[i0 + ah * k0] += a_ddt_ik;
                    a[k0 + ah * i0] += a_ddt_ik;
                    a[i1 + ah * k1] += a_ddt_ik;
                    a[k1 + ah * i1] += a_ddt_ik;
                }
            }
        }
    }

    pub fn assemble_dd_i1b(&mut self, w: f64, a: &mut [f64]) {
        if !self.has(HAVE_DAJ) { self.eval_daj(); }
        if !self.has(HAVE_DJT) { self.eval_djt(); }
        let daj = self.get_daj();
        let djt = self.get_djt();
        let nd = self.d_height;
        let ah = 2 * nd;
        if !self.has(HAVE_I1B) { self.eval_i1b(); }
        if !self.has(HAVE_I2B) { self.eval_i2b(); }
        let i2 = self.i2b * self.i2b;
        let coeff_a = w * self.i1b / i2;
        let coeff_b = 2.0 * w / self.i2b;
        let coeff_c = -2.0 * w / i2;
        if let Some(d) = &self.d {
            for i in 0..nd {
                let i0 = i + nd * 0;
                let i1 = i + nd * 1;
                let a_daj_i = [coeff_a * daj[i0], coeff_a * daj[i1]];
                let b_d_i = [coeff_b * d[i0], coeff_b * d[i1]];
                let c_djt_i = [coeff_c * djt[i0], coeff_c * djt[i1]];
                let c_daj_i = [coeff_c * daj[i0], coeff_c * daj[i1]];
                {
                    let a2_ii = b_d_i[0] * d[i0] + b_d_i[1] * d[i1];
                    a[i0 + ah * i0] += 2.0 * (a_daj_i[0] + c_djt_i[0]) * daj[i0] + a2_ii;
                    let a_ii_01 = (2.0 * a_daj_i[0] + c_djt_i[0]) * daj[i1] + c_daj_i[0] * djt[i1];
                    a[i0 + ah * i1] += a_ii_01;
                    a[i1 + ah * i0] += a_ii_01;
                    a[i1 + ah * i1] += 2.0 * (a_daj_i[1] + c_djt_i[1]) * daj[i1] + a2_ii;
                }
                for k in 0..i {
                    let k0 = k + nd * 0;
                    let k1 = k + nd * 1;
                    let a1_ik_01 = a_daj_i[0] * daj[k1] + a_daj_i[1] * daj[k0];
                    let a2_ik = b_d_i[0] * d[k0] + b_d_i[1] * d[k1];
                    let a_ik_00 =
                        (2.0 * a_daj_i[0] + c_djt_i[0]) * daj[k0] + a2_ik + c_daj_i[0] * djt[k0];
                    a[i0 + ah * k0] += a_ik_00;
                    a[k0 + ah * i0] += a_ik_00;
                    let a_ik_01 =
                        a1_ik_01 + c_djt_i[0] * daj[k1] + c_daj_i[0] * djt[k1];
                    a[i0 + ah * k1] += a_ik_01;
                    a[k1 + ah * i0] += a_ik_01;
                    let a_ik_10 =
                        a1_ik_01 + c_djt_i[1] * daj[k0] + c_daj_i[1] * djt[k0];
                    a[i1 + ah * k0] += a_ik_10;
                    a[k0 + ah * i1] += a_ik_10;
                    let a_ik_11 =
                        (2.0 * a_daj_i[1] + c_djt_i[1]) * daj[k1] + a2_ik + c_daj_i[1] * djt[k1];
                    a[i1 + ah * k1] += a_ik_11;
                    a[k1 + ah * i1] += a_ik_11;
                }
            }
        }
    }

    pub fn assemble_dd_i2(&mut self, w: f64, a: &mut [f64]) {
        if !self.has(HAVE_DAJ) { self.eval_daj(); }
        let daj = self.get_daj();
        let nd = self.d_height;
        let ah = 2 * nd;
        let a2 = 2.0 * w;
        for i in 0..ah {
            let avi = a2 * daj[i];
            a[i + ah * i] += avi * daj[i];
            for j in 0..i {
                let avv = avi * daj[j];
                a[i + ah * j] += avv;
                a[j + ah * i] += avv;
            }
        }
        let j = 1usize;
        let l = 0usize;
        for i in 0..nd {
            let ij = i + nd * j;
            let il = i + nd * l;
            let a_daj_ij = a2 * daj[ij];
            let a_daj_il = a2 * daj[il];
            for k in 0..i {
                let kj = k + nd * j;
                let kl = k + nd * l;
                let a_ijkl = a_daj_ij * daj[kl] - a_daj_il * daj[kj];
                a[ij + ah * kl] += a_ijkl;
                a[kl + ah * ij] += a_ijkl;
                a[kj + ah * il] -= a_ijkl;
                a[il + ah * kj] -= a_ijkl;
            }
        }
    }

    pub fn assemble_dd_i2b(&mut self, w: f64, a: &mut [f64]) {
        if !self.has(HAVE_DAJ) { self.eval_daj(); }
        let daj = self.get_daj();
        let nd = self.d_height;
        let ah = 2 * nd;
        if !self.has(HAVE_I2B) { self.eval_i2b(); }
        let a_w = w / self.i2b;
        let j = 1usize;
        let l = 0usize;
        for i in 0..nd {
            let ij = i + nd * j;
            let il = i + nd * l;
            let a_daj_ij = a_w * daj[ij];
            let a_daj_il = a_w * daj[il];
            for k in 0..i {
                let kj = k + nd * j;
                let kl = k + nd * l;
                let a_ijkl = a_daj_ij * daj[kl] - a_daj_il * daj[kj];
                a[ij + ah * kl] += a_ijkl;
                a[kl + ah * ij] += a_ijkl;
                a[kj + ah * il] -= a_ijkl;
                a[il + ah * kj] -= a_ijkl;
            }
        }
    }

    pub fn assemble_tprod_xy(&mut self, w: f64, x: &[f64; 4], y: &[f64; 4], a: &mut [f64]) {
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        let dxt = Self::compute_dzt(x, nd, &d);
        let dyt = Self::compute_dzt(y, nd, &d);
        let ah = 2 * nd;
        for i in 0..ah {
            let axi = w * dxt[i];
            let ayi = w * dyt[i];
            a[i + ah * i] += 2.0 * axi * dyt[i];
            for j in 0..i {
                let a_ij = axi * dyt[j] + ayi * dxt[j];
                a[i + ah * j] += a_ij;
                a[j + ah * i] += a_ij;
            }
        }
    }

    pub fn assemble_tprod_xx(&mut self, w: f64, x: &[f64; 4], a: &mut [f64]) {
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        let dxt = Self::compute_dzt(x, nd, &d);
        let ah = 2 * nd;
        for i in 0..ah {
            let axi = w * dxt[i];
            a[i + ah * i] += axi * dxt[i];
            for j in 0..i {
                let a_ij = axi * dxt[j];
                a[i + ah * j] += a_ij;
                a[j + ah * i] += a_ij;
            }
        }
    }
}

/// Evaluates 3D matrix invariants and their 1st/2nd derivatives.
#[derive(Debug, Clone)]
pub struct InvariantsEvaluator3D {
    j: [f64; 9],
    i1: f64,
    i1b: f64,
    i2: f64,
    i2b: f64,
    i3b: f64,
    i3b_p: f64,
    di1: [f64; 9],
    di1b: [f64; 9],
    di2: [f64; 9],
    di2b: [f64; 9],
    di3: [f64; 9],
    di3b: [f64; 9],
    b: [f64; 6],
    d_height: usize,
    d: Option<Vec<f64>>,
    eval_state: u32,
}

const HAVE_I1_3D: u32 = 1;
const HAVE_I1B_3D: u32 = 2;
const HAVE_B_OFFD_3D: u32 = 4;
const HAVE_I2_3D: u32 = 8;
const HAVE_I2B_3D: u32 = 16;
const HAVE_I3B_3D:  u32 = 1 << 5;
const HAVE_I3B_P_3D: u32 = 1 << 6;
const HAVE_DI1_3D:  u32 = 1 << 7;
const HAVE_DI1B_3D: u32 = 1 << 8;
const HAVE_DI2_3D:  u32 = 1 << 9;
const HAVE_DI2B_3D: u32 = 1 << 10;
const HAVE_DI3_3D:  u32 = 1 << 11;
const HAVE_DI3B_3D: u32 = 1 << 12;
const HAVE_DAJ_3D:  u32 = 1 << 13;
const HAVE_DJT_3D:  u32 = 1 << 14;
const HAVE_DDI2T_3D: u32 = 1 << 15;

impl InvariantsEvaluator3D {
    pub fn new(jac: Option<&[f64; 9]>) -> Self {
        Self {
            j: jac.copied().unwrap_or([0.0; 9]),
            i1: 0.0,
            i1b: 0.0,
            i2: 0.0,
            i2b: 0.0,
            i3b: 0.0,
            i3b_p: 0.0,
            di1: [0.0; 9],
            di1b: [0.0; 9],
            di2: [0.0; 9],
            di2b: [0.0; 9],
            di3: [0.0; 9],
            di3b: [0.0; 9],
            b: [0.0; 6],
            d_height: 0,
            d: None,
            eval_state: 0,
        }
    }

    pub fn set_jacobian(&mut self, jac: &[f64; 9]) {
        self.j = *jac;
        self.eval_state = 0;
    }

    pub fn set_derivative_matrix(&mut self, height: usize, deriv: &[f64]) {
        self.eval_state &= !(HAVE_DAJ_3D | HAVE_DJT_3D | HAVE_DDI2T_3D);
        self.d_height = height;
        let size = height * 3;
        let mut d = self.d.take().unwrap_or_default();
        d.resize(size, 0.0);
        d.copy_from_slice(&deriv[..size]);
        self.d = Some(d);
    }

    #[inline]
    fn has(&self, mask: u32) -> bool {
        self.eval_state & mask != 0
    }

    fn eval_i1(&mut self) {
        self.eval_state |= HAVE_I1_3D;
        let j = self.j;
        self.b[0] = j[0] * j[0] + j[3] * j[3] + j[6] * j[6];
        self.b[1] = j[1] * j[1] + j[4] * j[4] + j[7] * j[7];
        self.b[2] = j[2] * j[2] + j[5] * j[5] + j[8] * j[8];
        self.i1 = self.b[0] + self.b[1] + self.b[2];
    }

    fn eval_i1b(&mut self) {
        self.eval_state |= HAVE_I1B_3D;
        if !self.has(HAVE_I1_3D) { self.eval_i1(); }
        let i3b_p = self.get_i3b_p();
        self.i1b = self.i1 * i3b_p;
    }

    fn eval_b_offd(&mut self) {
        self.eval_state |= HAVE_B_OFFD_3D;
        let j = self.j;
        self.b[3] = j[0] * j[1] + j[3] * j[4] + j[6] * j[7];
        self.b[4] = j[0] * j[2] + j[3] * j[5] + j[6] * j[8];
        self.b[5] = j[1] * j[2] + j[4] * j[5] + j[7] * j[8];
    }

    fn eval_i2(&mut self) {
        self.eval_state |= HAVE_I2_3D;
        if !self.has(HAVE_I1_3D) { self.eval_i1(); }
        if !self.has(HAVE_B_OFFD_3D) { self.eval_b_offd(); }
        let bf2 = self.b[0] * self.b[0]
            + self.b[1] * self.b[1]
            + self.b[2] * self.b[2]
            + 2.0 * (self.b[3] * self.b[3] + self.b[4] * self.b[4] + self.b[5] * self.b[5]);
        self.i2 = (self.i1 * self.i1 - bf2) / 2.0;
    }

    fn eval_i2b(&mut self) {
        self.eval_state |= HAVE_I2B_3D;
        let i3b_p = self.get_i3b_p();
        if !self.has(HAVE_I2_3D) { self.eval_i2(); }
        self.i2b = self.i2 * i3b_p * i3b_p;
    }

    fn eval_i3b(&mut self) {
        self.eval_state |= HAVE_I3B_3D;
        let j = self.j;
        self.i3b = j[0] * (j[4] * j[8] - j[7] * j[5])
            - j[1] * (j[3] * j[8] - j[5] * j[6])
            + j[2] * (j[3] * j[7] - j[4] * j[6]);
    }

    fn get_i3b_p(&mut self) -> f64 {
        if !self.has(HAVE_I3B_P_3D) {
            self.eval_state |= HAVE_I3B_P_3D;
            if !self.has(HAVE_I3B_3D) { self.eval_i3b(); }
            self.i3b_p = self.i3b.powf(-2.0 / 3.0);
        }
        self.i3b_p
    }

    fn eval_di1(&mut self) {
        self.eval_state |= HAVE_DI1_3D;
        let j = self.j;
        for i in 0..9 {
            self.di1[i] = 2.0 * j[i];
        }
    }

    fn eval_di1b(&mut self) {
        self.eval_state |= HAVE_DI1B_3D;
        let i3b_p = self.get_i3b_p();
        let c1 = 2.0 * i3b_p;
        if !self.has(HAVE_I1_3D) { self.eval_i1(); }
        let c2 = self.i1 / (3.0 * self.i3b);
        if !self.has(HAVE_DI3B_3D) { self.eval_di3b(); }
        let j = self.j;
        let di3b = self.di3b;
        for i in 0..9 {
            self.di1b[i] = c1 * (j[i] - c2 * di3b[i]);
        }
    }

    fn eval_di2(&mut self) {
        self.eval_state |= HAVE_DI2_3D;
        if !self.has(HAVE_I1_3D) { self.eval_i1(); }
        if !self.has(HAVE_B_OFFD_3D) { self.eval_b_offd(); }
        let i1 = self.i1;
        let b = self.b;
        let c = [
            2.0 * (i1 - b[0]),
            2.0 * (i1 - b[1]),
            2.0 * (i1 - b[2]),
            -2.0 * b[3],
            -2.0 * b[4],
            -2.0 * b[5],
        ];
        let j = self.j;
        self.di2[0] = c[0] * j[0] + c[3] * j[1] + c[4] * j[2];
        self.di2[1] = c[3] * j[0] + c[1] * j[1] + c[5] * j[2];
        self.di2[2] = c[4] * j[0] + c[5] * j[1] + c[2] * j[2];
        self.di2[3] = c[0] * j[3] + c[3] * j[4] + c[4] * j[5];
        self.di2[4] = c[3] * j[3] + c[1] * j[4] + c[5] * j[5];
        self.di2[5] = c[4] * j[3] + c[5] * j[4] + c[2] * j[5];
        self.di2[6] = c[0] * j[6] + c[3] * j[7] + c[4] * j[8];
        self.di2[7] = c[3] * j[6] + c[1] * j[7] + c[5] * j[8];
        self.di2[8] = c[4] * j[6] + c[5] * j[7] + c[2] * j[8];
    }

    fn eval_di2b(&mut self) {
        self.eval_state |= HAVE_DI2B_3D;
        let i3b_p = self.get_i3b_p();
        let c1 = i3b_p * i3b_p;
        if !self.has(HAVE_I2_3D) { self.eval_i2(); }
        let c2 = (4.0 * self.i2 / self.i3b) / 3.0;
        if !self.has(HAVE_DI2_3D) { self.eval_di2(); }
        if !self.has(HAVE_DI3B_3D) { self.eval_di3b(); }
        let di2 = self.di2;
        let di3b = self.di3b;
        for i in 0..9 {
            self.di2b[i] = c1 * (di2[i] - c2 * di3b[i]);
        }
    }

    fn eval_di3(&mut self) {
        self.eval_state |= HAVE_DI3_3D;
        let c1 = 2.0 * self.i3b;
        if !self.has(HAVE_DI3B_3D) { self.eval_di3b(); }
        let di3b = self.di3b;
        for i in 0..9 {
            self.di3[i] = c1 * di3b[i];
        }
    }

    fn eval_di3b(&mut self) {
        self.eval_state |= HAVE_DI3B_3D;
        let j = self.j;
        self.di3b[0] = j[4] * j[8] - j[5] * j[7];
        self.di3b[1] = j[5] * j[6] - j[3] * j[8];
        self.di3b[2] = j[3] * j[7] - j[4] * j[6];
        self.di3b[3] = j[2] * j[7] - j[1] * j[8];
        self.di3b[4] = j[0] * j[8] - j[2] * j[6];
        self.di3b[5] = j[1] * j[6] - j[0] * j[7];
        self.di3b[6] = j[1] * j[5] - j[2] * j[4];
        self.di3b[7] = j[2] * j[3] - j[0] * j[5];
        self.di3b[8] = j[0] * j[4] - j[1] * j[3];
    }

    fn eval_daj(&mut self) {
        self.eval_state |= HAVE_DAJ_3D;
        if !self.has(HAVE_DI3B_3D) { self.eval_di3b(); }
    }

    fn eval_djt(&mut self) {
        self.eval_state |= HAVE_DJT_3D;
    }

    fn eval_ddi2t(&mut self) {
        self.eval_state |= HAVE_DDI2T_3D;
        if !self.has(HAVE_DI2_3D) { self.eval_di2(); }
    }

    fn compute_dzt(z: &[f64], nd: usize, d: &[f64]) -> Vec<f64> {
        let mut dzt = vec![0.0; 3 * nd];
        for i in 0..nd {
            let i0 = i + nd * 0;
            let i1 = i + nd * 1;
            let i2 = i + nd * 2;
            dzt[i0] = d[i0] * z[0] + d[i1] * z[3] + d[i2] * z[6];
            dzt[i1] = d[i0] * z[1] + d[i1] * z[4] + d[i2] * z[7];
            dzt[i2] = d[i0] * z[2] + d[i1] * z[5] + d[i2] * z[8];
        }
        dzt
    }

    pub fn get_i1(&mut self) -> f64 {
        if !self.has(HAVE_I1_3D) { self.eval_i1(); }
        self.i1
    }

    pub fn get_i1b(&mut self) -> f64 {
        if !self.has(HAVE_I1B_3D) { self.eval_i1b(); }
        self.i1b
    }

    pub fn get_i2(&mut self) -> f64 {
        if !self.has(HAVE_I2_3D) { self.eval_i2(); }
        self.i2
    }

    pub fn get_i2b(&mut self) -> f64 {
        if !self.has(HAVE_I2B_3D) { self.eval_i2b(); }
        self.i2b
    }

    pub fn get_i3(&mut self) -> f64 {
        if !self.has(HAVE_I3B_3D) { self.eval_i3b(); }
        self.i3b * self.i3b
    }

    pub fn get_i3b(&mut self) -> f64 {
        if !self.has(HAVE_I3B_3D) { self.eval_i3b(); }
        self.i3b
    }

    pub fn get_di1(&mut self) -> &[f64; 9] {
        if !self.has(HAVE_DI1_3D) { self.eval_di1(); }
        &self.di1
    }

    pub fn get_di1b(&mut self) -> &[f64; 9] {
        if !self.has(HAVE_DI1B_3D) { self.eval_di1b(); }
        &self.di1b
    }

    pub fn get_di2(&mut self) -> &[f64; 9] {
        if !self.has(HAVE_DI2_3D) { self.eval_di2(); }
        &self.di2
    }

    pub fn get_di2b(&mut self) -> &[f64; 9] {
        if !self.has(HAVE_DI2B_3D) { self.eval_di2b(); }
        &self.di2b
    }

    pub fn get_di3(&mut self) -> &[f64; 9] {
        if !self.has(HAVE_DI3_3D) { self.eval_di3(); }
        &self.di3
    }

    pub fn get_di3b(&mut self) -> &[f64; 9] {
        if !self.has(HAVE_DI3B_3D) { self.eval_di3b(); }
        &self.di3b
    }

    pub fn get_daj(&mut self) -> Vec<f64> {
        if !self.has(HAVE_DAJ_3D) { self.eval_daj(); }
        let di3b = self.di3b;
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        Self::compute_dzt(&di3b, nd, &d)
    }

    pub fn get_djt(&mut self) -> Vec<f64> {
        if !self.has(HAVE_DJT_3D) { self.eval_djt(); }
        let j = self.j;
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        Self::compute_dzt(&j, nd, &d)
    }

    pub fn get_ddi2t(&mut self) -> Vec<f64> {
        if !self.has(HAVE_DDI2T_3D) { self.eval_ddi2t(); }
        let di2 = self.di2;
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        Self::compute_dzt(&di2, nd, &d)
    }

    pub fn get_d(&self) -> &[f64] {
        self.d.as_ref().map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn d_height(&self) -> usize {
        self.d_height
    }

    pub fn assemble_dd_i1(&mut self, w: f64, a: &mut [f64]) {
        let nd = self.d_height;
        let ah = 3 * nd;
        let a2 = 2.0 * w;
        if let Some(d) = &self.d {
            for i in 0..nd {
                let i0 = i;
                let i1 = i + nd;
                let i2 = i + 2 * nd;
                let a_di = [a2 * d[i0], a2 * d[i1], a2 * d[i2]];
                let a_ddt_ii = a_di[0] * d[i0] + a_di[1] * d[i1] + a_di[2] * d[i2];
                a[i0 + ah * i0] += a_ddt_ii;
                a[i1 + ah * i1] += a_ddt_ii;
                a[i2 + ah * i2] += a_ddt_ii;
                for k in 0..i {
                    let k0 = k;
                    let k1 = k + nd;
                    let k2 = k + 2 * nd;
                    let a_ddt_ik = a_di[0] * d[k0] + a_di[1] * d[k1] + a_di[2] * d[k2];
                    a[i0 + ah * k0] += a_ddt_ik;
                    a[k0 + ah * i0] += a_ddt_ik;
                    a[i1 + ah * k1] += a_ddt_ik;
                    a[k1 + ah * i1] += a_ddt_ik;
                    a[i2 + ah * k2] += a_ddt_ik;
                    a[k2 + ah * i2] += a_ddt_ik;
                }
            }
        }
    }

    pub fn assemble_dd_i1b(&mut self, w: f64, a: &mut [f64]) {
        if !self.has(HAVE_DAJ_3D) { self.eval_daj(); }
        if !self.has(HAVE_DJT_3D) { self.eval_djt(); }
        let daj = self.get_daj();
        let djt = self.get_djt();
        let nd = self.d_height;
        let ah = 3 * nd;
        if !self.has(HAVE_I1B_3D) { self.eval_i1b(); }
        if !self.has(HAVE_I3B_3D) { self.eval_i3b(); }
        let r23 = 2.0 / 3.0;
        let r53 = 5.0 / 3.0;
        let i3 = self.i3b * self.i3b;
        let coeff_a = r23 * w * self.i1b / i3;
        let coeff_b = 2.0 * w * self.get_i3b_p();
        let coeff_c = -r23 * coeff_b / self.i3b;
        if let Some(d) = &self.d {
            for i in 0..nd {
                let i0 = i;
                let i1 = i + nd;
                let i2 = i + 2 * nd;
                let a_daj_i = [coeff_a * daj[i0], coeff_a * daj[i1], coeff_a * daj[i2]];
                let b_d_i = [coeff_b * d[i0], coeff_b * d[i1], coeff_b * d[i2]];
                let c_djt_i = [coeff_c * djt[i0], coeff_c * djt[i1], coeff_c * djt[i2]];
                let c_daj_i = [coeff_c * daj[i0], coeff_c * daj[i1], coeff_c * daj[i2]];
                {
                    let a2_ii = b_d_i[0] * d[i0] + b_d_i[1] * d[i1] + b_d_i[2] * d[i2];
                    a[i0 + ah * i0] += (r53 * a_daj_i[0] + 2.0 * c_djt_i[0]) * daj[i0] + a2_ii;
                    a[i1 + ah * i1] += (r53 * a_daj_i[1] + 2.0 * c_djt_i[1]) * daj[i1] + a2_ii;
                    a[i2 + ah * i2] += (r53 * a_daj_i[2] + 2.0 * c_djt_i[2]) * daj[i2] + a2_ii;
                    for j in 1..3 {
                        let ij = i + nd * j;
                        for l in 0..j {
                            let il = i + nd * l;
                            let a_ii_jl =
                                (r53 * a_daj_i[j] + c_djt_i[j]) * daj[il] + c_daj_i[j] * djt[il];
                            a[ij + ah * il] += a_ii_jl;
                            a[il + ah * ij] += a_ii_jl;
                        }
                    }
                }
                for k in 0..i {
                    let k0 = k;
                    let k1 = k + nd;
                    let k2 = k + 2 * nd;
                    let a2_ik = b_d_i[0] * d[k0] + b_d_i[1] * d[k1] + b_d_i[2] * d[k2];
                    for j in 0..3 {
                        let ij = i + nd * j;
                        let kj = k + nd * j;
                        let a_ik_jj =
                            (r53 * a_daj_i[j] + c_djt_i[j]) * daj[kj] + c_daj_i[j] * djt[kj] + a2_ik;
                        a[ij + ah * kj] += a_ik_jj;
                        a[kj + ah * ij] += a_ik_jj;
                    }
                    for j in 1..3 {
                        let ij = i + nd * j;
                        let kj = k + nd * j;
                        for l in 0..j {
                            let il = i + nd * l;
                            let kl = k + nd * l;
                            let a1b_ik_jl = a_daj_i[l] * daj[kj];
                            let a1b_ik_lj = a_daj_i[j] * daj[kl];
                            let a_ik_jl = a1b_ik_jl
                                + r23 * a1b_ik_lj
                                + c_djt_i[j] * daj[kl]
                                + c_daj_i[j] * djt[kl];
                            a[ij + ah * kl] += a_ik_jl;
                            a[kl + ah * ij] += a_ik_jl;
                            let a_ik_lj = r23 * a1b_ik_jl
                                + a1b_ik_lj
                                + c_djt_i[l] * daj[kj]
                                + c_daj_i[l] * djt[kj];
                            a[il + ah * kj] += a_ik_lj;
                            a[kj + ah * il] += a_ik_lj;
                        }
                    }
                }
            }
        }
    }

    pub fn assemble_dd_i2(&mut self, w: f64, a: &mut [f64]) {
        if !self.has(HAVE_DJT_3D) { self.eval_djt(); }
        let djt = self.get_djt();
        let nd = self.d_height;
        let ah = 3 * nd;
        if !self.has(HAVE_I1_3D) { self.eval_i1(); }
        let i1_val = self.i1;
        if !self.has(HAVE_B_OFFD_3D) { self.eval_b_offd(); }
        let b = self.b;
        let a2 = 2.0 * w;
        // First loop
        for i in 0..ah {
            let avi = a2 * djt[i];
            a[i + ah * i] += avi * djt[i];
            for j in 0..i {
                let avv = avi * djt[j];
                a[i + ah * j] += avv;
                a[j + ah * i] += avv;
            }
        }
        // Second + third loops
        if let Some(d) = &self.d {
            for i in 0..nd {
                let i0 = i;
                let i1_idx = i + nd;
                let i2_idx = i + 2 * nd;
                let a_d_i = [a2 * d[i0], a2 * d[i1_idx], a2 * d[i2_idx]];
                let a_djt_i = [a2 * djt[i0], a2 * djt[i1_idx], a2 * djt[i2_idx]];
                // k == i
                {
                    let a_ddt_ii =
                        a_d_i[0] * d[i0] + a_d_i[1] * d[i1_idx] + a_d_i[2] * d[i2_idx];
                    let z1_ii = i1_val * a_ddt_ii
                        - (a_djt_i[0] * djt[i0] + a_djt_i[1] * djt[i1_idx] + a_djt_i[2] * djt[i2_idx]);
                    for j in 0..3 {
                        let ij = i + nd * j;
                        a[ij + ah * ij] += z1_ii - a_ddt_ii * b[j];
                    }
                    let z2_ii_01 = a_ddt_ii * b[3];
                    let z2_ii_02 = a_ddt_ii * b[4];
                    let z2_ii_12 = a_ddt_ii * b[5];
                    a[i0 + ah * i1_idx] -= z2_ii_01;
                    a[i1_idx + ah * i0] -= z2_ii_01;
                    a[i0 + ah * i2_idx] -= z2_ii_02;
                    a[i2_idx + ah * i0] -= z2_ii_02;
                    a[i1_idx + ah * i2_idx] -= z2_ii_12;
                    a[i2_idx + ah * i1_idx] -= z2_ii_12;
                }
                // 0 <= k < i
                for k in 0..i {
                    let k0 = k;
                    let k1_idx = k + nd;
                    let k2_idx = k + 2 * nd;
                    let a_ddt_ik =
                        a_d_i[0] * d[k0] + a_d_i[1] * d[k1_idx] + a_d_i[2] * d[k2_idx];
                    let z1_ik = i1_val * a_ddt_ik
                        - (a_djt_i[0] * djt[k0] + a_djt_i[1] * djt[k1_idx] + a_djt_i[2] * djt[k2_idx]);
                    for j in 0..3 {
                        let ij = i + nd * j;
                        let kj = k + nd * j;
                        let z2_ik_jj = z1_ik - a_ddt_ik * b[j];
                        a[ij + ah * kj] += z2_ik_jj;
                        a[kj + ah * ij] += z2_ik_jj;
                    }
                    {
                        let z2_ik_01 = a_ddt_ik * b[3];
                        a[i0 + ah * k1_idx] -= z2_ik_01;
                        a[i1_idx + ah * i0] -= z2_ik_01;
                        a[k0 + ah * i1_idx] -= z2_ik_01;
                        a[k1_idx + ah * i0] -= z2_ik_01;
                        let z2_ik_02 = a_ddt_ik * b[4];
                        a[i0 + ah * k2_idx] -= z2_ik_02;
                        a[i2_idx + ah * i0] -= z2_ik_02;
                        a[k0 + ah * i2_idx] -= z2_ik_02;
                        a[k2_idx + ah * i0] -= z2_ik_02;
                        let z2_ik_12 = a_ddt_ik * b[5];
                        a[i1_idx + ah * k2_idx] -= z2_ik_12;
                        a[i2_idx + ah * k1_idx] -= z2_ik_12;
                        a[k1_idx + ah * i2_idx] -= z2_ik_12;
                        a[k2_idx + ah * i1_idx] -= z2_ik_12;
                    }
                    for j in 1..3 {
                        let ij = i + nd * j;
                        let kj = k + nd * j;
                        for l in 0..j {
                            let il = i + nd * l;
                            let kl = k + nd * l;
                            let z3_ik_jl = a_djt_i[j] * djt[kl] - a_djt_i[l] * djt[kj];
                            a[ij + ah * kl] += z3_ik_jl;
                            a[kl + ah * ij] += z3_ik_jl;
                            a[kj + ah * il] -= z3_ik_jl;
                            a[il + ah * kj] -= z3_ik_jl;
                        }
                    }
                }
            }
        }
    }

    pub fn assemble_dd_i2b(&mut self, w: f64, a: &mut [f64]) {
        if !self.has(HAVE_DAJ_3D) { self.eval_daj(); }
        let daj = self.get_daj();
        let nd = self.d_height;
        let ah = 3 * nd;
        if !self.has(HAVE_I3B_3D) { self.eval_i3b(); }
        let a_w = w / self.i3b;
        for i in 0..ah {
            let avi = a_w * daj[i];
            a[i + ah * i] += avi * daj[i];
            for j in 0..i {
                let avv = avi * daj[j];
                a[i + ah * j] += avv;
                a[j + ah * i] += avv;
            }
        }
    }

    pub fn assemble_dd_i3(&mut self, w: f64, a: &mut [f64]) {
        if !self.has(HAVE_DI3B_3D) { self.eval_di3b(); }
        let i3b = self.i3b;
        let di3b = self.get_di3b().clone();
        self.assemble_tprod_xx(2.0 * w, &di3b, a);
        self.assemble_dd_i3b(2.0 * w * i3b, a);
    }

    pub fn assemble_dd_i3b(&mut self, w: f64, a: &mut [f64]) {
        if !self.has(HAVE_DAJ_3D) { self.eval_daj(); }
        let daj = self.get_daj();
        let nd = self.d_height;
        let ah = 3 * nd;
        let a2 = 2.0 * w;
        for i in 0..ah {
            let avi = a2 * daj[i];
            a[i + ah * i] += avi * daj[i];
            for j in 0..i {
                let avv = avi * daj[j];
                a[i + ah * j] += avv;
                a[j + ah * i] += avv;
            }
        }
    }

    pub fn assemble_tprod_xy(&mut self, w: f64, x: &[f64; 9], y: &[f64; 9], a: &mut [f64]) {
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        let dxt = Self::compute_dzt(x, nd, &d);
        let dyt = Self::compute_dzt(y, nd, &d);
        let ah = 3 * nd;
        for i in 0..ah {
            let axi = w * dxt[i];
            let ayi = w * dyt[i];
            a[i + ah * i] += 2.0 * axi * dyt[i];
            for j in 0..i {
                let a_ij = axi * dyt[j] + ayi * dxt[j];
                a[i + ah * j] += a_ij;
                a[j + ah * i] += a_ij;
            }
        }
    }

    pub fn assemble_tprod_xx(&mut self, w: f64, x: &[f64; 9], a: &mut [f64]) {
        let nd = self.d_height;
        let d = self.d.clone().unwrap_or_default();
        let dxt = Self::compute_dzt(x, nd, &d);
        let ah = 3 * nd;
        for i in 0..ah {
            let axi = w * dxt[i];
            a[i + ah * i] += axi * dxt[i];
            for j in 0..i {
                let a_ij = axi * dxt[j];
                a[i + ah * j] += a_ij;
                a[j + ah * i] += a_ij;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ie2d_basic() {
        let jac = [1.0, 0.0, 0.0, 1.0];
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        assert!((ie.get_i1() - 2.0).abs() < 1e-14);
        assert!((ie.get_i2b() - 1.0).abs() < 1e-14);
        assert!((ie.get_i1b() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_ie2d_scaled() {
        let jac = [2.0, 0.0, 0.0, 2.0];
        let mut ie = InvariantsEvaluator2D::new(Some(&jac));
        assert!((ie.get_i1() - 8.0).abs() < 1e-14);
        assert!((ie.get_i2b() - 4.0).abs() < 1e-14);
        assert!((ie.get_i1b() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_ie3d_basic() {
        let jac = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        assert!((ie.get_i1() - 3.0).abs() < 1e-14);
        assert!((ie.get_i3b() - 1.0).abs() < 1e-14);
        assert!((ie.get_i1b() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_ie3d_scaled() {
        let jac = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let mut ie = InvariantsEvaluator3D::new(Some(&jac));
        assert!((ie.get_i1() - 12.0).abs() < 1e-14);
        assert!((ie.get_i3b() - 8.0).abs() < 1e-14);
        assert!((ie.get_i1b() - 3.0).abs() < 1e-14);
    }
}
