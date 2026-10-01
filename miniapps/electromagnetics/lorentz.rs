//! # Miniapp Lorentz — Simple Lorentz Force Particle Mover (1:1 port of MFEM
//! miniapps/electromagnetics/lorentz.cpp, `mpirun -np 1` semantics)
//!
//! Computes the trajectories of charged particles under Lorentz forces
//! `dp/dt = q (E + v×B)` with the explicit Boris algorithm (Qin et al.,
//! Phys. Plasmas 20, 082111 (2013)) — the same four-term momentum update as
//! `Boris::ParticleStep` in the C++ miniapp, transcribed term for term.
//!
//! The E/B fields are read from VisIt data collections (`-er/-ef/-ec`,
//! `-br/-bf/-bc`, PARALLEL_FORMAT) such as those written by the Volta/Tesla
//! miniapps; the two fields do not need to share a mesh.  Particles that leave
//! either mesh are removed, exactly like the C++ `RemoveLostParticles` after a
//! `FindPointsGSLIB` miss.
//!
//! ## np=1 semantic port (multidomain precedent)
//!
//! The C++ miniapp is MPI-only (ParGridFunction/FindPointsGSLIB/Redistribute).
//! This port implements the `mpirun -np 1` semantics:
//! * one rank owns all `-npt` particles (the C++ per-rank count
//!   `npt/num_ranks + (rank < npt%num_ranks)` reduces to `npt`);
//! * particle initialization draws from `std::mt19937(seed = rank)`; the
//!   libstdc++ engine and distributions are replicated exactly (see
//!   `Mt19937`/`uniform_real_01`/`NormalLibstdcxx` below) so the random
//!   initial conditions are bit-identical at np=1;
//! * `ParticleSet::Redistribute` does not exist without GSLIB and is a no-op
//!   at np=1 (every entry of the rank list is the local rank), mirrored by
//!   `Boris::redistribute` — the surrounding call structure (including the
//!   `FindParticles` refresh) is kept;
//! * FindPointsGSLIB is replaced by a brute-force point search (`SerialFinder`)
//!   on the same straight meshes — the instrumented C++ acceptance copy
//!   (`$HOME/work/d103lor/d103_lorentz_instr.cpp`) uses the identical
//!   stand-in, so the trajectory truth run and this port apply the same
//!   finder semantics.
//!
//! ## Acceptance / instrumentation
//!
//! The console protocol is the C++ one: the banner followed by one
//! `Step: N | Time: T` line per step (T printed at C++ default 6-significant
//! digits).  Setting `D103_DUMP=<file>` additionally writes a full-precision
//! per-step trace (INIT/EVAL/STEP/REMOVED records, one row per particle:
//! `i x y z px py pz ex ey ez bx by bz`) in the same format as the
//! instrumented C++ copy, for step-by-step trajectory comparison.
//!
//! ## Declared gaps (exit status 3)
//!
//! * `-d/--device` other than `cpu` (no device push path);
//! * `-epdc/-epdr/-bpdc/-bpdr` other than the default 6 (the fem-rs VisIt
//!   loader pins 6-digit cycle/rank padding);
//! * field families other than vector H1 in the collections (Volta/Tesla E/B
//!   are vector H1; other families are not evaluatable here);
//! * `-vis`/default visualization: GLVis streaming is not implemented — a
//!   warning is printed and the run continues, exactly like the C++ when the
//!   GLVis socket cannot connect; `-vt`/`-vf` only affect that stream.
//! * no E and no B collection (`-er`/`-br` both empty): the C++
//!   `MFEM_VERIFY(E_gf || B_gf)` abort is an exit 3 here.
//!
//! Non-3-D collections and unreadable collections exit 1, like the C++
//! `Error loading ... field` path.
//!
//! C++ reference: MFEM 4.10 miniapps/electromagnetics/lorentz.cpp.

use std::io::Write;
use std::path::Path;

use fem_assembly::postproc::grid_function::GridFunction;
use fem_io::data_collection::{read_gf_slice, read_mesh_slice, read_visit_root};
use fem_io::mfem::read_mfem;
use fem_mesh::particle::{Particle, ParticleSet};
use fem_mesh::transformation::find_points;
use fem_mesh::Mesh;
use fem_solver::fmt_g;
use fem_space::{FESpace, H1Space};

/// `Boris::Fields` — layout of the per-particle data block
/// `[mass, charge, mom(3), E(3), B(3)]`.
const MASS: usize = 0;
const CHARGE: usize = 1;
const MOM: usize = 2;
const EFIELD: usize = 5;
const BFIELD: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// lorentz.cpp `struct LorentzContext`
// ─────────────────────────────────────────────────────────────────────────────

struct DColl
{
   coll_name: String,
   field_name: String,
   cycle: usize,
   pad_digits_cycle: usize,
   pad_digits_rank: usize,
}

struct LorentzContext
{
   e: DColl,
   b: DColl,
   ordering: i64,        // 0 - byNODES, 1 - byVDIM
   npt: i64,             // total number of particles
   q: f64,               // particle charge
   m: f64,               // particle mass
   x_min: [f64; 3],      // initial position min
   x_max: [f64; 3],      // initial position max
   p_min: [f64; 3],      // initial momentum min
   p_max: [f64; 3],      // initial momentum max
   dt: f64,              // time step
   nt: i64,              // number of timesteps
   redist_interval: i64, // redistribution interval
   redist_mesh: i64,     // redistribution mesh: 0: E mesh, 1: B mesh
   device_config: String,
}

impl Default for LorentzContext
{
   fn default() -> Self
   {
      LorentzContext {
         e: DColl { coll_name: String::new(), field_name: "E".to_string(),
                    cycle: 10, pad_digits_cycle: 6, pad_digits_rank: 6 },
         b: DColl { coll_name: String::new(), field_name: "B".to_string(),
                    cycle: 10, pad_digits_cycle: 6, pad_digits_rank: 6 },
         ordering: 1,
         npt: 1,
         q: 1.0,
         m: 1.0,
         x_min: [-1.0, -1.0, -1.0],
         x_max: [1.0, 1.0, 1.0],
         p_min: [-1.0, -1.0, -1.0],
         p_max: [1.0, 1.0, 1.0],
         dt: 1e-2,
         nt: 1000,
         redist_interval: 5,
         redist_mesh: 0,
         device_config: "cpu".to_string(),
      }
   }
}

// ─────────────────────────────────────────────────────────────────────────────
// libstdc++ PRNG replication (bit-exact at np=1)
//
// The C++ `InitializeChargedParticles` seeds `std::mt19937 gen(rank)` and draws
// `std::uniform_real_distribution<real_t>(0,1)` for positions and
// `std::normal_distribution<real_t>` (one object per momentum component, with
// the Marsaglia polar pair cache) for momenta.  The engine is fully specified;
// the two distributions are libstdc++-specific, so their exact algorithms
// (GCC 13 `bits/random.tcc`, the reference build) are replicated here.
// ─────────────────────────────────────────────────────────────────────────────

/// `std::mt19937` (libstdc++ `mersenne_twister_engine`, w=32 n=624 m=397
/// r=31 a=0x9908b0df u=11 d=0xffffffff s=7 b=0x9d2c5680 t=15 c=0xefc60000
/// l=18 f=1812433253; twist-on-demand via `_M_p = state_size` after seed).
struct Mt19937
{
   x: [u32; 624],
   p: usize,
}

impl Mt19937
{
   fn new(seed: u32) -> Self
   {
      let mut x = [0u32; 624];
      x[0] = seed;
      for i in 1..624
      {
         let mut v = x[i - 1];
         v ^= v >> 30; // w - 2
         v = v.wrapping_mul(1812433253); // f
         v = v.wrapping_add(i as u32);
         x[i] = v;
      }
      Mt19937 { x, p: 624 }
   }

   fn gen_rand(&mut self)
   {
      const N: usize = 624;
      const M: usize = 397;
      const A: u32 = 0x9908_b0df;
      let upper: u32 = 0x8000_0000; // (~0) << r, r = 31
      let lower: u32 = 0x7fff_ffff;
      for k in 0..(N - M)
      {
         let y = (self.x[k] & upper) | (self.x[k + 1] & lower);
         self.x[k] = self.x[k + M] ^ (y >> 1) ^ if y & 1 != 0 { A } else { 0 };
      }
      for k in (N - M)..(N - 1)
      {
         let y = (self.x[k] & upper) | (self.x[k + 1] & lower);
         self.x[k] = self.x[k + M - N] ^ (y >> 1) ^ if y & 1 != 0 { A } else { 0 };
      }
      let y = (self.x[N - 1] & upper) | (self.x[0] & lower);
      self.x[N - 1] = self.x[M - 1] ^ (y >> 1) ^ if y & 1 != 0 { A } else { 0 };
      self.p = 0;
   }

   fn next_u32(&mut self) -> u32
   {
      if self.p >= 624
      {
         self.gen_rand();
      }
      // Tempering: z ^= (z >> u) & d  (d = 0xffffffff masks nothing)
      let mut z = self.x[self.p];
      self.p += 1;
      z ^= z >> 11;
      z ^= (z << 7) & 0x9d2c_5680;
      z ^= (z << 15) & 0xefc6_0000;
      z ^= z >> 18;
      z
   }
}

/// libstdc++ `generate_canonical<double, 53>` for a 32-bit engine
/// (`bits/random.tcc`): two draws assembled in `f64` in the same order,
/// `sum = x0*1 + x1*2^32`, divided by the post-loop factor `2^64`, with the
/// `ret >= 1` clamp to `nextafter(1, 0)`.
fn generate_canonical_64(urng: &mut Mt19937) -> f64
{
   let r: f64 = 4294967296.0; // 2^32, exact in f64 (long double in libstdc++)
   const M: usize = 2; // ceil(53 / 32)
   let mut sum = 0.0f64;
   let mut tmp = 1.0f64;
   for _ in 0..M
   {
      sum += urng.next_u32() as f64 * tmp;
      tmp *= r;
   }
   let ret = sum / tmp;
   if ret >= 1.0
   {
      return f64::from_bits(1.0f64.to_bits() - 1); // nextafter(1, 0)
   }
   ret
}

/// `std::uniform_real_distribution<double>(0, 1)`:
/// `generate_canonical * (b - a) + a` with (a, b) = (0, 1).
fn uniform_real_01(urng: &mut Mt19937) -> f64
{
   generate_canonical_64(urng) * (1.0 - 0.0) + 0.0
}

/// libstdc++ `std::normal_distribution<double>` (Marsaglia polar method, with
/// the saved second-of-pair state).  One instance per component, like the
/// C++ `std::vector<std::normal_distribution<real_t>> norm_dist_p`.
struct NormalLibstdcxx
{
   mean: f64,
   stddev: f64,
   saved_available: bool,
   saved: f64,
}

impl NormalLibstdcxx
{
   fn new(mean: f64, stddev: f64) -> Self
   {
      NormalLibstdcxx { mean, stddev, saved_available: false, saved: 0.0 }
   }

   fn sample(&mut self, urng: &mut Mt19937) -> f64
   {
      let ret = if self.saved_available
      {
         self.saved_available = false;
         self.saved
      }
      else
      {
         let (mut x, mut y, mut r2);
         loop
         {
            x = 2.0 * uniform_real_01(urng) - 1.0;
            y = 2.0 * uniform_real_01(urng) - 1.0;
            r2 = x * x + y * y;
            if !(r2 > 1.0 || r2 == 0.0)
            {
               break;
            }
         }
         let mult = (-2.0 * r2.ln() / r2).sqrt();
         self.saved = x * mult;
         self.saved_available = true;
         y * mult
      };
      ret * self.stddev + self.mean
   }
}

// ─────────────────────────────────────────────────────────────────────────────
// np=1 stand-in for FindPointsGSLIB (mirror of SerialFinder in the instrumented
// C++ acceptance copy): brute-force point location + grid-function evaluation
// at the found (element, reference coords).  Not-found points keep their stale
// field values, exactly like `FindPointsGSLIB::Interpolate`.
// ─────────────────────────────────────────────────────────────────────────────

type H1Gf<'a> = GridFunction<'a, H1Space<Mesh<3>>>;

struct SerialFinder
{
   /// Per-particle `(element, reference coords in [0,1]^3)` when located.
   found: Vec<Option<(u32, Vec<f64>)>>,
   /// Indices of points no element claimed (`GetPointsNotFoundIndices`).
   not_found: Vec<usize>,
}

impl SerialFinder
{
   fn new() -> Self
   {
      SerialFinder { found: Vec::new(), not_found: Vec::new() }
   }

   /// `FindPointsGSLIB::FindPoints(X)` on a straight 3-D mesh.
   fn find_points(&mut self, mesh: &Mesh<3>, ps: &ParticleSet)
   {
      let n = ps.n_particles();
      let mut pts = Vec::with_capacity(n * 3);
      for p in ps.iter()
      {
         pts.extend_from_slice(&p.x);
      }
      let (elem_ids, ips) = find_points(mesh, &pts, n);
      self.found = elem_ids
         .iter()
         .zip(ips)
         .map(|(&e, xi)| if e < 0 { None } else { Some((e as u32, xi)) })
         .collect();
      self.not_found = self
         .found
         .iter()
         .enumerate()
         .filter(|(_, f)| f.is_none())
         .map(|(i, _)| i)
         .collect();
   }

   /// `FindPointsGSLIB::Interpolate(gf, field, ordering)` for a vector H1
   /// field stored as one scalar grid function per component.
   fn interpolate(&self, ps: &mut ParticleSet, comps: &[H1Gf<'_>], base: usize)
   {
      for (i, p) in ps.iter_mut().enumerate()
      {
         if let Some((e, xi)) = &self.found[i]
         {
            for d in 0..3
            {
               p.data[base + d] = comps[d].evaluate_at_element(*e, xi);
            }
         }
      }
   }
}

// ─────────────────────────────────────────────────────────────────────────────
// Boris (the C++ class, transcribed)
// ─────────────────────────────────────────────────────────────────────────────

/// A loaded collection field as the step loop sees it: its mesh (for the
/// finder) and one scalar grid function per vector component (each borrowing
/// the field's `H1Space`, which lives in the owning `FieldBundle`).
struct FieldData<'a>
{
   mesh: &'a Mesh<3>,
   comps: Vec<H1Gf<'a>>,
}

/// Owned collection field: mesh + the H1 space + the raw per-component DOF
/// vectors.  `view()` lends them to the step loop as a [`FieldData`].
struct FieldBundle
{
   mesh: Mesh<3>,
   space: H1Space<Mesh<3>>,
   comp_vals: [Vec<f64>; 3],
}

impl FieldBundle
{
   fn view(&self) -> FieldData<'_>
   {
      FieldData {
         mesh: &self.mesh,
         comps: self
            .comp_vals
            .iter()
            .map(|v| GridFunction::new(&self.space, v.clone()))
            .collect(),
      }
   }
}

struct Boris<'a>
{
   e: Option<FieldData<'a>>,
   b: Option<FieldData<'a>>,
   e_finder: SerialFinder,
   b_finder: SerialFinder,
   particles: ParticleSet,
   /// [D103] per-step trajectory dump (None = disabled).
   dump: Option<std::fs::File>,
}

/// Single particle Boris step — `Boris::ParticleStep`, term for term:
/// pm = p + 0.5·dt·q·e; pp = (a1·(pm×B) + a2·pm + a3·B)/a4;
/// p = pp + 0.5·dt·q·e; x += (dt/m)·p.
fn particle_step(part: &mut Particle, dt: f64)
{
   let m = part.data[MASS];
   let q = part.data[CHARGE];
   let p = [part.data[MOM], part.data[MOM + 1], part.data[MOM + 2]];
   let e = [part.data[EFIELD], part.data[EFIELD + 1], part.data[EFIELD + 2]];
   let b = [part.data[BFIELD], part.data[BFIELD + 1], part.data[BFIELD + 2]];

   // Compute half of the contribution from q E: add(p, 0.5*dt*q, e, pm_)
   let half_dt_q = 0.5 * dt * q;
   let pm = [p[0] + half_dt_q * e[0],
             p[1] + half_dt_q * e[1],
             p[2] + half_dt_q * e[2]];

   // Compute the contribution from q p x B
   let b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];

   // ... along pm x B: pm_.cross3D(b, pxB_); pp_.Set(a1, pxB_)
   let a1 = 4.0 * dt * q * m;
   let pmxb = [pm[1] * b[2] - pm[2] * b[1],
               pm[2] * b[0] - pm[0] * b[2],
               pm[0] * b[1] - pm[1] * b[0]];
   let mut pp = [a1 * pmxb[0], a1 * pmxb[1], a1 * pmxb[2]];

   // ... along pm: pp_.Add(a2, pm_)
   let a2 = 4.0 * m * m - dt * dt * q * q * b2;
   pp[0] += a2 * pm[0];
   pp[1] += a2 * pm[1];
   pp[2] += a2 * pm[2];

   // ... along B: pp_.Add(a3, b)
   let b_dot_pm = b[0] * pm[0] + b[1] * pm[1] + b[2] * pm[2];
   let a3 = 2.0 * dt * dt * q * q * b_dot_pm;
   pp[0] += a3 * b[0];
   pp[1] += a3 * b[1];
   pp[2] += a3 * b[2];

   // scale by common denominator: pp_ /= a4
   let a4 = 4.0 * m * m + dt * dt * q * q * b2;
   pp[0] /= a4;
   pp[1] /= a4;
   pp[2] /= a4;

   // Update the momentum: add(pp_, 0.5*dt*q, e, p)
   part.data[MOM] = pp[0] + half_dt_q * e[0];
   part.data[MOM + 1] = pp[1] + half_dt_q * e[1];
   part.data[MOM + 2] = pp[2] + half_dt_q * e[2];

   // Update the position: x.Add(dt / m, p)
   let scale = dt / m;
   part.x[0] += scale * part.data[MOM];
   part.x[1] += scale * part.data[MOM + 1];
   part.x[2] += scale * part.data[MOM + 2];
}

impl<'a> Boris<'a>
{
   /// `Boris::Boris(comm, E_gf, B_gf, nparticles, ordering, use_device)`.
   /// The ordering argument only selects the C++ particle-data memory layout
   /// (byNODES/byVDIM); the fem-rs particle set stores per-particle blocks, so
   /// both orderings describe the same data here.
   fn new(e: Option<FieldData<'a>>, b: Option<FieldData<'a>>, nparticles: usize,
          ordering: i64) -> Self
   {
      let _ = ordering; // layout-only in the C++ too (see doc comment)
      // ParticleSet(comm, nparticles, dim, field_vdims{1,1,dim,dim,dim}, ...):
      // per-particle block [mass, charge, mom(3), E(3), B(3)] = 11 slots.
      let mut particles = ParticleSet::new(3);
      for _ in 0..nparticles
      {
         particles.add_particle(vec![0.0; 3], 11);
      }
      Boris {
         e,
         b,
         e_finder: SerialFinder::new(),
         b_finder: SerialFinder::new(),
         particles,
         dump: None,
      }
   }

   /// `Boris::FindParticles` — locate all particles in the E and B meshes.
   fn find_particles(&mut self)
   {
      if let Some(f) = &self.e
      {
         let mesh = f.mesh;
         self.e_finder.find_points(mesh, &self.particles);
      }
      if let Some(f) = &self.b
      {
         let mesh = f.mesh;
         self.b_finder.find_points(mesh, &self.particles);
      }
   }

   /// `Boris::EvaluateFieldsAtParticles` — interpolate E/B at the particle
   /// locations (must follow `find_particles`, as in the C++).  A missing
   /// collection zeroes its field (the C++ `E = 0.0` / `B = 0.0` arms).
   fn evaluate_fields_at_particles(&mut self)
   {
      if let Some(f) = &self.e
      {
         let comps = &f.comps;
         self.e_finder.interpolate(&mut self.particles, comps, EFIELD);
      }
      else
      {
         for p in self.particles.iter_mut()
         {
            p.data[EFIELD] = 0.0;
            p.data[EFIELD + 1] = 0.0;
            p.data[EFIELD + 2] = 0.0;
         }
      }
      if let Some(f) = &self.b
      {
         let comps = &f.comps;
         self.b_finder.interpolate(&mut self.particles, comps, BFIELD);
      }
      else
      {
         for p in self.particles.iter_mut()
         {
            p.data[BFIELD] = 0.0;
            p.data[BFIELD + 1] = 0.0;
            p.data[BFIELD + 2] = 0.0;
         }
      }
   }

   /// `Boris::Step` — evaluate fields, push every particle, re-locate.
   fn step(&mut self, t: &mut f64, dt: f64)
   {
      self.evaluate_fields_at_particles();
      // [D103] field values at the (pre-push) particle positions
      self.dump_state("EVAL", -1, *t);

      let n = self.particles.n_particles();
      for i in 0..n
      {
         particle_step(self.particles.get_mut(i), dt);
      }

      self.find_particles();
      *t += dt;
   }

   /// `Boris::RemoveLostParticles` — union of the E and B not-found lists,
   /// then removal from the set.  Returns the removed indices (ascending,
   /// like the C++ `Array::Union` chain).
   fn remove_lost_particles(&mut self) -> Vec<usize>
   {
      let mut lost: Vec<usize> = Vec::new();
      for &i in self.e_finder.not_found.iter().chain(self.b_finder.not_found.iter())
      {
         if let Err(pos) = lost.binary_search(&i)
         {
            lost.insert(pos, i);
         }
      }
      let lost_set = &lost;
      let mut i = 0usize;
      self.particles.retain(|_| {
         let keep = lost_set.binary_search(&i).is_err();
         i += 1;
         keep
      });
      lost
   }

   /// `Boris::Redistribute` — np=1 no-op.  MFEM's
   /// `ParticleSet::Redistribute(rank_list)` moves nothing when every entry of
   /// the rank list equals the local rank ("Index = this rank means no data is
   /// moved", fem/particleset.hpp), which is always the case at one rank.
   /// The C++ body's only state effect beyond the move
   /// (`proc_list.DeleteAt(removed_idxs)`) touches a local copy.
   fn redistribute(&mut self, redist_mesh: usize, removed_idxs: &[usize])
   {
      let _ = (redist_mesh, removed_idxs);
   }

   /// [D103] Full-precision state dump: `tag step t N` followed by one row per
   /// particle `i x y z px py pz ex ey ez bx by bz` (Rust `{}` on f64 is the
   /// shortest round-trip form; the comparison parses the numbers, so the
   /// C++ `%.17g` and this format carry identical values).
   fn dump_state(&mut self, tag: &str, step: i64, t: f64)
   {
      let Some(file) = self.dump.as_mut()
      else
      {
         return;
      };
      let n = self.particles.n_particles();
      let mut buf = format!("{tag} {step} {t} {n}\n");
      for i in 0..n
      {
         let p = self.particles.get(i);
         buf.push_str(&format!(
            "{i} {} {} {} {} {} {} {} {} {} {} {} {}\n",
            p.x[0], p.x[1], p.x[2],
            p.data[MOM], p.data[MOM + 1], p.data[MOM + 2],
            p.data[EFIELD], p.data[EFIELD + 1], p.data[EFIELD + 2],
            p.data[BFIELD], p.data[BFIELD + 1], p.data[BFIELD + 2]));
      }
      let _ = file.write_all(buf.as_bytes());
   }
}

/// `InitializeChargedParticles` — random initial conditions from
/// `std::mt19937(seed = rank)`: uniform positions in `[x_min, x_max]`
/// (degenerate boxes pin the coordinate), Gaussian momenta centered between
/// `p_min`/`p_max` with 3-sigma spanning the box, redrawn until inside
/// (degenerate boxes pin the momentum).  At np=1 the seed is 0.
fn initialize_charged_particles(ps: &mut ParticleSet, x_min: &[f64; 3],
                                x_max: &[f64; 3], p_min: &[f64; 3],
                                p_max: &[f64; 3], m: f64, q: f64, rank: u32)
{
   let mut gen = Mt19937::new(rank);

   // Gaussian per component: centered between p_min and p_max with the
   // 3-sigma range covering the box — `add(0.5, p_min, p_max, p_center)`
   // (MFEM `z = a·(x+y)`) and `dp = (p_max - p_min) * (1_r/6_r)`; both
   // spellings are kept so the rounding matches the C++ bit for bit.
   let mut norm_dist_p = Vec::with_capacity(3);
   for d in 0..3
   {
      let center = 0.5 * (p_min[d] + p_max[d]);
      let dp = (p_max[d] - p_min[d]) * (1.0 / 6.0);
      norm_dist_p.push(NormalLibstdcxx::new(center, if dp > 0.0 { dp } else { 1.0 }));
   }

   for i in 0..ps.n_particles()
   {
      for d in 0..3
      {
         if x_min[d] >= x_max[d]
         {
            ps.get_mut(i).x[d] = x_min[d];
         }
         else
         {
            ps.get_mut(i).x[d] =
               x_min[d] + uniform_real_01(&mut gen) * (x_max[d] - x_min[d]);
         }

         if p_min[d] >= p_max[d]
         {
            ps.get_mut(i).data[MOM + d] = p_min[d];
         }
         else
         {
            loop
            {
               let p_val = norm_dist_p[d].sample(&mut gen);
               if !(p_val < p_min[d] || p_val > p_max[d])
               {
                  ps.get_mut(i).data[MOM + d] = p_val;
                  break;
               }
            }
         }
      }
      ps.get_mut(i).data[MASS] = m;
      ps.get_mut(i).data[CHARGE] = q;
   }
}

// ─────────────────────────────────────────────────────────────────────────────
// VisIt data collection input (lorentz.cpp `ReadGridFunction`)
// ─────────────────────────────────────────────────────────────────────────────

/// `get-values.rs` convention: `"H1_3D_P2"` → `("H1", 2)`.
fn parse_family_order(basis: &str) -> (String, u8)
{
   let order = basis
      .rsplit(['P', '_'])
      .next()
      .and_then(|s| s.parse::<u8>().ok())
      .unwrap_or(1);
   let family = basis.split('_').next().unwrap_or("H1").to_string();
   (family, order)
}

/// Ordering header of a field slice: 0 = byNODES blocked layout,
/// 1 = byVDIM interleaved.  `read_gf_slice` does not surface the header's
/// `Ordering:` line, so the miniapp re-reads the same slice's header;
/// MFEM's default is 1 (FiniteElementSpace byVDIM).
fn slice_ordering(slice: &Path) -> u32
{
   let Ok(content) = std::fs::read_to_string(slice)
   else
   {
      return 1;
   };
   for line in content.lines()
   {
      if let Some(rest) = line.strip_prefix("Ordering:")
      {
         return rest.trim().parse().unwrap_or(1);
      }
      if line.is_empty()
      {
         break; // header block ended
      }
   }
   1
}

/// Match the `{...}` block starting at `s[start..]`'s first `{` and return its
/// inner text (span includes both braces).
fn json_block(s: &str, start: usize) -> Option<&str>
{
   let bytes = s.as_bytes();
   let open = bytes[start..].iter().position(|&b| b == b'{')? + start;
   let mut depth = 0usize;
   let mut in_str = false;
   let mut escape = false;
   for (i, &b) in bytes.iter().enumerate().skip(open)
   {
      if in_str
      {
         if escape
         {
            escape = false;
         }
         else if b == b'\\'
         {
            escape = true;
         }
         else if b == b'"'
         {
            in_str = false;
         }
         continue;
      }
      match b
      {
         b'"' => in_str = true,
         b'{' => depth += 1,
         b'}' =>
         {
            depth -= 1;
            if depth == 0
            {
               return Some(&s[open..=i]);
            }
         }
         _ => {}
      }
   }
   None
}

/// Quoted string value following `"key":` inside `block`.
fn json_string_value<'a>(block: &'a str, key: &str) -> Option<&'a str>
{
   let pos = block.find(key)? + key.len();
   let rest = block[pos..].trim_start();
   let rest = rest.strip_prefix(':').unwrap_or(rest).trim_start();
   let rest = rest.strip_prefix('"')?;
   let end = rest.find('"')?;
   Some(&rest[..end])
}

/// `ReadGridFunction(coll_name, field_name, pad_digits_cycle, pad_digits_rank,
/// cycle, dc, gf)` — open the collection, load `cycle`, return the mesh and
/// the named vector H1 field as per-component DOF vectors.
///
/// `prefix` is the collection directory (`VisItDataCollection::SetPrefixPath`;
/// the C++ default is the current directory).  The mesh slice is resolved like
/// `VisItDataCollection::Load` (fem/datacollection.cpp:740-760): from the root
/// file's `"mesh"` object, whose `path` carries the `%06d` rank placeholder —
/// PARALLEL_FORMAT names the slice `pmesh.%06d` (fem/datacollection.cpp:270),
/// older roots `mesh.%06d`; both are probed when the root has no mesh path.
/// Field slices follow the writer's `<coll>_<cycle>/<field>.%06d` convention.
fn load_field(prefix: &Path, coll_name: &str, field_name: &str,
              pad_digits_cycle: usize, pad_digits_rank: usize,
              cycle: usize) -> Result<FieldBundle, String>
{
   if pad_digits_cycle != 6
   {
      // fem_io's VisIt reader pins 6-digit cycle padding.
      eprintln!(
         "miniapp_lorentz: pad-digits-cycle {pad_digits_cycle} is not supported (6-digit cycle \
          padding pinned) — exiting with status 3"
      );
      std::process::exit(3);
   }
   let _ = pad_digits_rank; // np=1: the single rank slice is always .000000
   let root_path = prefix.join(format!("{coll_name}_{cycle:06}.mfem_root"));

   // `dc->Load(cycle)` + `dc->HasField(field_name)` error paths.
   let (root_content, fields_meta) = {
      let (cycle_read, _domains, fields) =
         read_visit_root(&root_path).map_err(|e| format!("{e}"))?;
      let _ = cycle_read;
      let content =
         std::fs::read_to_string(&root_path).map_err(|e| format!("{e}"))?;
      (content, fields)
   };
   if !fields_meta.iter().any(|f| f.name == field_name)
   {
      return Err(format!("field '{field_name}' not in collection"));
   }

   let dir = root_path.parent().unwrap_or_else(|| Path::new("."));
   let cycle_dir = dir.join(format!("{coll_name}_{cycle:06}"));

   // Mesh slice: root "mesh" object first (the C++ path), historical names as
   // the fallback.
   let mesh_path = json_block(&root_content, root_content.find("\"mesh\"").unwrap_or(0))
      .and_then(|mesh_obj| json_string_value(mesh_obj, "\"path\""))
      .map(|p| dir.join(p.replace("%06d", "000000")))
      .unwrap_or_else(|| cycle_dir.join("pmesh.000000"));
   let mesh_path = if mesh_path.exists()
   {
      mesh_path
   }
   else
   {
      cycle_dir.join("mesh.000000")
   };
   let mesh_txt = read_mesh_slice(&mesh_path).map_err(|e| format!("{e}"))?;
   let mfem =
      read_mfem(mesh_txt.as_bytes()).map_err(|e| format!("mesh parse error: {e}"))?;
   let mesh =
      mfem.mesh3d.ok_or_else(|| "only 3D meshes are currently supported".to_string())?;

   // Field slice.
   let slice = cycle_dir.join(format!("{field_name}.000000"));
   let (basis, vdim, values) = read_gf_slice(&slice).map_err(|e| format!("{e}"))?;

   let (family, order) = parse_family_order(&basis);
   if family != "H1"
   {
      eprintln!(
         "miniapp_lorentz: field '{field_name}' has basis '{basis}' — only vector H1 collections \
          are supported here — exiting with status 3"
      );
      std::process::exit(3);
   }
   if vdim != 3
   {
      eprintln!(
         "miniapp_lorentz: field '{field_name}' has VDim {vdim} — only vector (VDim=3) H1 fields \
          are supported here — exiting with status 3"
      );
      std::process::exit(3);
   }
   if values.len() % 3 != 0
   {
      return Err(format!(
         "field '{field_name}' slice size {} is not a multiple of VDim=3",
         values.len()
      ));
   }

   // Component layout from the slice's `Ordering:` header.
   let ordering = slice_ordering(&slice);
   let n = values.len() / 3;
   let mut comp_vals: [Vec<f64>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
   match ordering
   {
      1 =>
      {
         for (i, chunk) in values.chunks_exact(3).enumerate()
         {
            comp_vals[0][i] = chunk[0];
            comp_vals[1][i] = chunk[1];
            comp_vals[2][i] = chunk[2];
         }
      }
      _ =>
      {
         for (c, vals) in comp_vals.iter_mut().enumerate()
         {
            vals.copy_from_slice(&values[c * n..(c + 1) * n]);
         }
      }
   }

   let space = H1Space::new(mesh.clone(), order);
   if space.n_dofs() != n
   {
      return Err(format!(
         "field '{field_name}' has {n} dofs/component but the H1 space builds {} (order {order})",
         space.n_dofs()
      ));
   }
   Ok(FieldBundle { mesh, space, comp_vals })
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI (the C++ OptionsParser surface)
// ─────────────────────────────────────────────────────────────────────────────

fn usage()
{
   println!("Usage: miniapp_lorentz [options]");
   println!("   -er <str>      (--e-root-file) VisIt data collection E field root file prefix");
   println!("   -ef <str>      (--e-field-name) E field name (default E)");
   println!("   -ec <int>      (--e-cycle) E field cycle index to read (default 10)");
   println!("   -epdc/-epdr <int>  E field cycle/rank pad digits (default 6)");
   println!("   -br <str>      (--b-root-file) VisIt data collection B field root file prefix");
   println!("   -bf <str>      (--b-field-name) B field name (default B)");
   println!("   -bc <int>      (--b-cycle) B field cycle index to read (default 10)");
   println!("   -bpdc/-bpdr <int>  B field cycle/rank pad digits (default 6)");
   println!("   -rdf <int>     (--redist-interval) redistribution interval, 0 = none (default 5)");
   println!("   -rdm <int>     (--redistribution-mesh) 0: E mesh, 1: B mesh (default 0)");
   println!("   -o <int>       (--ordering) particle data ordering 0/1 (default 1)");
   println!("   -npt <int>     (--num-particles) total number of particles (default 1)");
   println!("   -m <float>     (--mass) particles' mass (default 1)");
   println!("   -q <float>     (--charge) particles' charge (default 1)");
   println!("   -xmin/-xmax <'x y z'>   initial particle location bounds (default -1..1 cube)");
   println!("   -pmin/-pmax <'px py pz'> initial momentum bounds (default -1..1 cube)");
   println!("   -dt <float>    (--time-step) time step (default 1e-2)");
   println!("   -nt <int>      (--num-timesteps) number of timesteps (default 1000)");
   println!("   -vis/-no-vis   (default on) GLVis visualization");
   println!("   -vt <int>      (--vis-tail-size) GLVis trajectory tail size (default 5)");
   println!("   -vf <int>      (--vis-interval) GLVis update interval (default 4)");
   println!("   -d <str>       (--device) device configuration (default cpu)");
   println!("   -h/--help      print this usage and exit");
}

fn parse_vec3(args: &[String], i: usize, flag: &str) -> [f64; 3]
{
   let toks: Vec<&str> = args[i + 1].split_whitespace().collect();
   if toks.len() != 3
   {
      eprintln!("miniapp_lorentz: {flag} needs 3 values, got {}", toks.len());
      usage();
      std::process::exit(1);
   }
   let mut out = [0.0f64; 3];
   for (d, tok) in toks.iter().enumerate()
   {
      out[d] = tok.parse().unwrap_or_else(|_| {
         eprintln!("miniapp_lorentz: bad float '{tok}' for {flag} — exiting with status 1");
         std::process::exit(1);
      });
   }
   out
}

/// The C++ `OptionsParser` surface; returns the context plus the three
/// visualization settings.  Unknown options and missing values print the usage
/// and exit 1, like `args.Good() == false` in the C++.
fn parse_args() -> (LorentzContext, bool, i64, i64)
{
   let mut ctx = LorentzContext::default();
   let mut visualization = true;
   let mut vis_tail_size = 5i64;
   let mut vis_interval = 4i64;
   let args: Vec<String> = std::env::args().collect();
   let mut i = 1usize;
   while i < args.len()
   {
      let flag = args[i].as_str();
      // One argv token of value for most options (MFEM vector options also
      // arrive as one quoted token); boolean flags take none.
      macro_rules! value
      {
         () =>
         {
            args.get(i + 1).cloned().unwrap_or_else(|| {
               eprintln!("miniapp_lorentz: option {flag} requires an argument");
               usage();
               std::process::exit(1);
            })
         };
      }
      macro_rules! parsed
      {
         ($ty:ty, $default:expr) =>
         {
            value!().parse::<$ty>().unwrap_or_else(|_| {
               eprintln!("miniapp_lorentz: bad value for {flag}");
               usage();
               std::process::exit(1);
            })
         };
      }
      let mut consumed_value = true;
      match flag
      {
         "-er" | "--e-root-file" => ctx.e.coll_name = value!(),
         "-ef" | "--e-field-name" => ctx.e.field_name = value!(),
         "-ec" | "--e-cycle" => ctx.e.cycle = parsed!(usize, 10),
         "-epdc" | "--e-pad-digits-cycle" => ctx.e.pad_digits_cycle = parsed!(usize, 6),
         "-epdr" | "--e-pad-digits-rank" => ctx.e.pad_digits_rank = parsed!(usize, 6),
         "-br" | "--b-root-file" => ctx.b.coll_name = value!(),
         "-bf" | "--b-field-name" => ctx.b.field_name = value!(),
         "-bc" | "--b-cycle" => ctx.b.cycle = parsed!(usize, 10),
         "-bpdc" | "--b-pad-digits-cycle" => ctx.b.pad_digits_cycle = parsed!(usize, 6),
         "-bpdr" | "--b-pad-digits-rank" => ctx.b.pad_digits_rank = parsed!(usize, 6),
         "-rdf" | "--redist-interval" => ctx.redist_interval = parsed!(i64, 5),
         "-rdm" | "--redistribution-mesh" => ctx.redist_mesh = parsed!(i64, 0),
         "-o" | "--ordering" => ctx.ordering = parsed!(i64, 1),
         "-npt" | "--num-particles" => ctx.npt = parsed!(i64, 1),
         "-m" | "--mass" => ctx.m = parsed!(f64, 1.0),
         "-q" | "--charge" => ctx.q = parsed!(f64, 1.0),
         "-xmin" | "--x-min" => ctx.x_min = parse_vec3(&args, i, flag),
         "-xmax" | "--x-max" => ctx.x_max = parse_vec3(&args, i, flag),
         "-pmin" | "--p-min" => ctx.p_min = parse_vec3(&args, i, flag),
         "-pmax" | "--p-max" => ctx.p_max = parse_vec3(&args, i, flag),
         "-dt" | "--time-step" => ctx.dt = parsed!(f64, 1e-2),
         "-nt" | "--num-timesteps" => ctx.nt = parsed!(i64, 1000),
         "-vis" | "--visualization" =>
         {
            visualization = true;
            consumed_value = false;
         }
         "-no-vis" | "--no-visualization" =>
         {
            visualization = false;
            consumed_value = false;
         }
         "-vt" | "--vis-tail-size" => vis_tail_size = parsed!(i64, 5),
         "-vf" | "--vis-interval" => vis_interval = parsed!(i64, 4),
         "-d" | "--device" => ctx.device_config = value!(),
         "-h" | "--help" =>
         {
            usage();
            std::process::exit(0);
         }
         _ =>
         {
            eprintln!("miniapp_lorentz: unknown option {flag}");
            usage();
            std::process::exit(1);
         }
      }
      i += if consumed_value { 2 } else { 1 };
   }
   (ctx, visualization, vis_tail_size, vis_interval)
}

/// The lorentz.cpp banner.
fn display_banner()
{
   println!("   ____                                __          ");
   println!("  |    |    ___________   ____   _____/  |_________");
   println!("  |    |   /  _ \\_  __ \\_/ __ \\ /    \\   __\\___   /");
   println!("  |    |__(  <_> )  | \\/\\  ___/|   |  \\  |  /    / ");
   println!("  |_______ \\____/|__|    \\___  >___|  /__| /_____ \\");
   println!("          \\/                 \\/     \\/           \\/");
}

fn main()
{
   let (ctx, visualization, vis_tail_size, vis_interval) = parse_args();
   display_banner();

   // MFEM_VERIFY(E_gf || B_gf): at least one of E/B must be provided.
   if ctx.e.coll_name.is_empty() && ctx.b.coll_name.is_empty()
   {
      eprintln!(
         "miniapp_lorentz: Must pass an E field or B field to Boris. — the C++ miniapp reads its \
          fields from VisIt data collections written by the Volta/Tesla miniapps \
          (-er/-br root prefixes). Exiting with status 3."
      );
      std::process::exit(3);
   }

   // Device configuration: only the host path exists here (the C++ device
   // kernel StepDevice is the same math, but there is no device backend).
   if ctx.device_config != "cpu"
   {
      eprintln!(
         "miniapp_lorentz: device configuration '{}' is not supported (host only) — exiting with \
          status 3",
         ctx.device_config
      );
      std::process::exit(3);
   }

   // GLVis streaming is not implemented; the C++ continues without it when the
   // socket cannot connect (a warning, not an error), so the run continues.
   if visualization && vis_interval > 0
   {
      println!(
         "miniapp_lorentz: GLVis visualization not implemented (vis_tail_size {vis_tail_size}, \
          vis_interval {vis_interval}) — continuing without it, like the C++ when the GLVis \
          socket cannot connect."
      );
   }

   if ctx.npt < 0
   {
      eprintln!("miniapp_lorentz: invalid -npt {} — exiting with status 1", ctx.npt);
      std::process::exit(1);
   }

   // Read E field if provided (C++ `Error loading E field` → exit 1).
   let cwd = Path::new(".");
   let e_bundle = if !ctx.e.coll_name.is_empty()
   {
      match load_field(cwd, &ctx.e.coll_name, &ctx.e.field_name,
                       ctx.e.pad_digits_cycle, ctx.e.pad_digits_rank, ctx.e.cycle)
      {
         Ok(f) => Some(f),
         Err(e) =>
         {
            eprintln!("Error loading E field: {e}");
            std::process::exit(1);
         }
      }
   }
   else
   {
      None
   };

   // Read B field if provided.
   let b_bundle = if !ctx.b.coll_name.is_empty()
   {
      match load_field(cwd, &ctx.b.coll_name, &ctx.b.field_name,
                       ctx.b.pad_digits_cycle, ctx.b.pad_digits_rank, ctx.b.cycle)
      {
         Ok(f) => Some(f),
         Err(e) =>
         {
            eprintln!("Error loading B field: {e}");
            std::process::exit(1);
         }
      }
   }
   else
   {
      None
   };
   // [D103] the original computes Mesh::GetBoundingBox here; its only consumer
   // is the GLVis trajectory bounding box.

   // np=1: every particle is on rank 0 (the C++ per-rank count
   // `npt/num_ranks + (rank < npt%num_ranks)` reduces to `npt`).
   let num_particles = ctx.npt as usize;

   let boris_e = e_bundle.as_ref().map(|f| f.view());
   let boris_b = b_bundle.as_ref().map(|f| f.view());
   let mut boris = Boris::new(boris_e, boris_b, num_particles, ctx.ordering);

   initialize_charged_particles(&mut boris.particles, &ctx.x_min, &ctx.x_max,
                                &ctx.p_min, &ctx.p_max, ctx.m, ctx.q,
                                0 /* rank */);

   // [D103] trajectory dump
   if let Ok(dump_path) = std::env::var("D103_DUMP")
   {
      match std::fs::File::create(&dump_path)
      {
         Ok(mut f) =>
         {
            let _ = writeln!(
               f,
               "# ctx npt={} m={} q={} dt={} nt={} er={} ef={} ec={} br={} bf={} bc={} o={} \
                rdf={} rdm={}",
               ctx.npt, ctx.m, ctx.q, ctx.dt, ctx.nt,
               ctx.e.coll_name, ctx.e.field_name, ctx.e.cycle,
               ctx.b.coll_name, ctx.b.field_name, ctx.b.cycle,
               ctx.ordering, ctx.redist_interval, ctx.redist_mesh
            );
            boris.dump = Some(f);
         }
         Err(e) =>
         {
            eprintln!("D103: cannot open dump file {dump_path}: {e}");
            std::process::exit(1);
         }
      }
   }

   boris.find_particles();
   boris.redistribute(ctx.redist_mesh as usize, &[]);
   boris.find_particles();
   boris.evaluate_fields_at_particles();
   // [D103]
   boris.dump_state("INIT", 0, 0.0);

   let mut t: f64 = 0.0;
   let dt = ctx.dt;

   // [D103] the original sets up the GLVis ParticleTrajectories here; removed.

   let nt = ctx.nt;
   for step in 1..=nt
   {
      // Step the Boris algorithm (the device path is refused above).
      boris.step(&mut t, dt);

      println!("Step: {step} | Time: {}", fmt_g(t));
      // [D103]
      boris.dump_state("STEP", step, t);

      // Remove lost particles from particle set and output
      let removed_idxs = boris.remove_lost_particles();
      // [D103] removal record
      if boris.dump.is_some()
      {
         let mut line = format!("REMOVED {step} {}", removed_idxs.len());
         for idx in &removed_idxs
         {
            line.push_str(&format!(" {idx}"));
         }
         line.push('\n');
         if let Some(f) = boris.dump.as_mut()
         {
            let _ = f.write_all(line.as_bytes());
         }
      }

      let particles_removed = !removed_idxs.is_empty();

      // Redistribute (np=1: no-op, but the call structure and the
      // FindParticles refresh below mirror the C++).
      let mut redistributed = false;
      if ctx.redist_interval > 0
         && step % ctx.redist_interval == 0
         && boris.particles.n_particles() > 0
      {
         boris.redistribute(ctx.redist_mesh as usize, &removed_idxs);
         redistributed = true;
      }

      // Keep the finder data synchronized with the ParticleSet after
      // particles have been removed or redistributed.
      if particles_removed || redistributed
      {
         boris.find_particles();
      }
   }
}

#[cfg(test)]
mod tests
{
   use super::*;

   /// mt19937(seed 0) outputs, confirmed against GCC 13 libstdc++ on this
   /// machine (`std::mt19937 gen(0)` — the engine twists before the first
   /// use, so the sequence differs from the textbook untwisted vector).
   #[test]
   fn mt19937_seed0_reference_values()
   {
      let mut g = Mt19937::new(0);
      assert_eq!(g.next_u32(), 2357136044);
      assert_eq!(g.next_u32(), 2546248239);
      assert_eq!(g.next_u32(), 3071714933);
   }

   /// libstdc++ `generate_canonical<double, 53>` with a 32-bit engine:
   /// (x0 + x1·2^32)/2^64 in the exact libstdc++ arithmetic order.  The
   /// expected value equals the GCC 13 `uniform_real_distribution(0,1)`
   /// first draw (verified against a sequential C++ probe).
   #[test]
   fn uniform01_first_values()
   {
      let mut g = Mt19937::new(0);
      let x0 = 2357136044u32 as f64;
      let x1 = 2546248239u32 as f64;
      let expected = (x0 + x1 * 4294967296.0_f64) / (4294967296.0_f64 * 4294967296.0_f64);
      let got = uniform_real_01(&mut g);
      assert_eq!(got, expected);
      assert_eq!(got, 0.59284461651668263);
      assert!((0.0..1.0).contains(&got));
   }

   /// One full Boris step in a constant field, against the C++ ParticleStep
   /// formulas evaluated symbolically for B = (0, 0, 1).
   #[test]
   fn particle_step_constant_fields()
   {
      let mut ps = ParticleSet::new(3);
      ps.add_particle(vec![0.5, 0.5, 0.5], 11);
      let p = ps.get_mut(0);
      p.data[MASS] = 1.0;
      p.data[CHARGE] = 1.0;
      p.data[MOM] = 0.2;
      p.data[MOM + 1] = 0.1;
      p.data[MOM + 2] = 0.0;
      // E = (0.1, 0, 0), B = (0, 0, 1)
      p.data[EFIELD] = 0.1;
      p.data[BFIELD + 2] = 1.0;

      let dt = 0.01;
      let half = 0.5 * dt;
      let pm = [0.2 + half * 0.1, 0.1, 0.0];
      let pxb = [pm[1] * 1.0, -pm[0] * 1.0, 0.0];
      let a1 = 4.0 * dt;
      let mut pp = [a1 * pxb[0], a1 * pxb[1], a1 * pxb[2]];
      let a2 = 4.0 - dt * dt;
      pp[0] += a2 * pm[0];
      pp[1] += a2 * pm[1];
      pp[2] += a2 * pm[2];
      let a3 = 2.0 * dt * dt * pm[2];
      pp[2] += a3;
      let a4 = 4.0 + dt * dt;
      for v in &mut pp
      {
         *v /= a4;
      }
      let expect_p = [pp[0] + half * 0.1, pp[1], pp[2]];

      particle_step(ps.get_mut(0), dt);
      let p = ps.get(0);
      for d in 0..3
      {
         assert_eq!(p.data[MOM + d], expect_p[d], "momentum comp {d}");
      }
      // x_new = x_old + dt * p_new
      assert_eq!(p.x[0], 0.5 + dt * expect_p[0]);
   }

   /// Removal bookkeeping: a particle no element claims is dropped from the
   /// set; surviving particles keep their order and their data.
   #[test]
   fn remove_lost_particles_union()
   {
      let bundle = test_field_bundle();
      // Boris::new pre-creates 3 particles (the C++ ParticleSet ctor count).
      let mut boris = Boris::new(Some(bundle.view()), None, 3, 1);
      boris.particles.get_mut(0).x = vec![0.5, 0.5, 0.5]; // inside
      boris.particles.get_mut(1).x = vec![5.0, 0.5, 0.5]; // outside
      boris.particles.get_mut(2).x = vec![0.25, 0.5, 0.5]; // inside
      // A nonzero datum on a survivor: removal must keep order and data.
      boris.particles.get_mut(0).data[MOM] = 0.7;

      boris.find_particles();
      let lost = boris.remove_lost_particles();
      assert_eq!(lost, vec![1]);
      assert_eq!(boris.particles.n_particles(), 2);
      assert_eq!(boris.particles.get(0).x[0], 0.5);
      assert_eq!(boris.particles.get(0).data[MOM], 0.7);
      assert_eq!(boris.particles.get(1).x[0], 0.25);
   }

   /// A constant-zero vector H1 field on a one-element unit-cube hex mesh
   /// (the smallest stand-in for a collection field).
   fn test_field_bundle() -> FieldBundle
   {
      let mesh = Mesh::<3>::unit_cube_hex(1);
      let space = H1Space::new(mesh.clone(), 1);
      let n = space.n_dofs();
      FieldBundle { mesh, space, comp_vals: [vec![0.0; n], vec![0.0; n], vec![0.0; n]] }
   }

   // ── D103 regression pins — C++ truth constants (round 103 acceptance) ──
   //
   // The constants below are taken verbatim from the full-precision dump of
   // the instrumented C++ copy (mpirun -np 1, GCC 13 libstdc++, MFEM 4.10
   // libmfem.a) for the S1 acceptance scenario:
   //   -er D103E-Parallel -br D103B-Parallel -npt 20
   //   -xmin '0.25 0.3 0.25' -xmax '0.75 0.7 0.75'
   //   -pmin '0.15 0.05 0.1' -pmax '0.35 0.2 0.3' -dt 0.01 -nt 200 -no-vis
   // (see fem-rs/tmp/d103lor/ for the probe, the instrumented copy and the
   // dumps; the acceptance run matched the port bit for bit).

   /// Truth constant: S1 scenario particle limits.
   const D103_X_MIN: [f64; 3] = [0.25, 0.3, 0.25];
   const D103_X_MAX: [f64; 3] = [0.75, 0.7, 0.75];
   const D103_P_MIN: [f64; 3] = [0.15, 0.05, 0.1];
   const D103_P_MAX: [f64; 3] = [0.35, 0.2, 0.3];

   /// The RNG chain (mt19937 seed 0 + libstdc++ distributions) reproduces the
   /// C++ initial particle states exactly: pins particles 0 and 1 of the S1
   /// truth dump (`INIT 0 0 20`, rows 0-1).
   #[test]
   fn d103_cxx_truth_initial_particles()
   {
      let mut boris = Boris::new(None, None, 20, 1);
      initialize_charged_particles(&mut boris.particles, &D103_X_MIN,
                                   &D103_X_MAX, &D103_P_MIN, &D103_P_MAX,
                                   1.0, 1.0, 0);
      assert_eq!(boris.particles.n_particles(), 20);

      let p0 = boris.particles.get(0);
      assert_eq!(p0.x[0], 0.54642230825834126);
      assert_eq!(p0.x[1], 0.63890069495373247);
      assert_eq!(p0.x[2], 0.3987673026786171);
      assert_eq!(p0.data[MOM], 0.25394988282107211);
      assert_eq!(p0.data[MOM + 1], 0.089441853953914949);
      assert_eq!(p0.data[MOM + 2], 0.19816549648997306);
      assert_eq!(p0.data[MASS], 1.0);
      assert_eq!(p0.data[CHARGE], 1.0);

      let p1 = boris.particles.get(1);
      assert_eq!(p1.x[0], 0.48883255587232316);
      assert_eq!(p1.x[1], 0.62486749065962854);
      assert_eq!(p1.x[2], 0.48998858576278376);
      assert_eq!(p1.data[MOM], 0.25379892719224045);
      assert_eq!(p1.data[MOM + 1], 0.16300173747771804);
      assert_eq!(p1.data[MOM + 2], 0.19642298606563274);
   }

   /// One Boris step from the C++ truth state reproduces the C++ result
   /// exactly: particle 0 of the S1 truth dump, `EVAL -1 0 20` row 0
   /// (pre-push state, E/B interpolated by MFEM) pushed with dt = 0.01 must
   /// equal `STEP 1 0.01 20` row 0.
   #[test]
   fn d103_cxx_truth_first_boris_step()
   {
      let mut ps = ParticleSet::new(3);
      ps.add_particle(vec![0.0; 3], 11);
      {
         let p = ps.get_mut(0);
         // INIT row 0: position and momentum.
         p.x[0] = 0.54642230825834126;
         p.x[1] = 0.63890069495373247;
         p.x[2] = 0.3987673026786171;
         p.data[MOM] = 0.25394988282107211;
         p.data[MOM + 1] = 0.089441853953914949;
         p.data[MOM + 2] = 0.19816549648997306;
         p.data[MASS] = 1.0;
         p.data[CHARGE] = 1.0;
         // EVAL row 0: MFEM-interpolated E and B at the particle.
         p.data[EFIELD] = 0.28940396315812172;
         p.data[EFIELD + 1] = 0.014543101513842411;
         p.data[EFIELD + 2] = 0.014972594114516829;
         p.data[BFIELD] = 0.039876730267861711;
         p.data[BFIELD + 1] = -0.05464223082583413;
         p.data[BFIELD + 2] = 1.0;
      }

      particle_step(ps.get_mut(0), 0.01);
      let p = ps.get(0);
      // STEP 1 row 0.
      assert_eq!(p.x[0], 0.54900065769580764);
      assert_eq!(p.x[1], 0.63977176873085495);
      assert_eq!(p.x[2], 0.40074870463939244);
      assert_eq!(p.data[MOM], 0.25783494374663846);
      assert_eq!(p.data[MOM + 1], 0.087107377712245485);
      assert_eq!(p.data[MOM + 2], 0.19814019607753511);
   }

   /// End-to-end collection round trip: a C++-shaped PARALLEL_FORMAT-style
   /// fixture (mfem_root + `pmesh.000000` — the MFEM 4.10 parallel mesh-slice
   /// naming — plus an interleaved byVDIM field slice with the MFEM
   /// `FiniteElementSpace` header) loads through the miniapp's collection
   /// path, and the vector H1 field evaluates exactly at known points.
   #[test]
   fn d103_dc_roundtrip_evaluation()
   {
      let root_dir = std::env::temp_dir().join("d103_lorentz_dc");
      let cycle_dir = root_dir.join("D103T_000010");
      std::fs::create_dir_all(&cycle_dir).expect("mkdir");

      // 1-hex unit-cube mesh (MFEM v1.0 text, with boundary quads).
      let mesh_txt = concat!(
         "MFEM mesh v1.0\n\ndimension\n3\n\nelements\n1\n1 5 0 1 2 3 4 5 6 7\n\n",
         "boundary\n6\n1 3 0 1 2 3\n1 3 1 5 6 2\n1 3 0 4 5 1\n1 3 3 7 6 2\n",
         "1 3 0 4 7 3\n1 3 4 5 6 7\n\nvertices\n8\n3\n",
         "0 0 0\n1 0 0\n1 1 0\n0 1 0\n0 0 1\n1 0 1\n1 1 1\n0 1 1\n"
      );
      // Root file with the mesh path carrying the %06d rank placeholder.
      let root_json = concat!(
         "{\n  \"dsets\": {\n    \"main\": {\n      \"cycle\": 10,\n      \"domains\": 1,\n",
         "      \"fields\": {\n        \"E\": {\n          \"path\": \"D103T_000010/E.%06d\",\n",
         "          \"tags\": { \"assoc\": \"nodes\", \"basis\": \"H1_3D_P1\", \"comps\": \"3\",\n",
         "                     \"order\": \"1\" } } },\n",
         "      \"mesh\": { \"format\": \"1\", \"path\": \"D103T_000010/pmesh.%06d\",\n",
         "                 \"tags\": { \"spatial_dim\": \"3\", \"topo_dim\": \"3\" } } } } }\n"
      );
      // E = (x, y, z) at the 8 vertices, interleaved byVDIM (Ordering: 1).
      let corners: [[f64; 3]; 8] =
         [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
          [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]];
      let mut slice = String::from(
         "FiniteElementSpace\nFiniteElementCollection: H1_3D_P1\nVDim: 3\nOrdering: 1\n\n",
      );
      for c in &corners
      {
         slice.push_str(&format!("{} {} {}\n", c[0], c[1], c[2]));
      }
      std::fs::write(root_dir.join("D103T_000010.mfem_root"), root_json).unwrap();
      std::fs::write(cycle_dir.join("pmesh.000000"), mesh_txt).unwrap();
      std::fs::write(cycle_dir.join("E.000000"), &slice).unwrap();

      let bundle = load_field(&root_dir, "D103T", "E", 6, 6, 10).expect("load_field");
      let mut boris = Boris::new(Some(bundle.view()), None, 0, 1);
      boris.particles.add_particle(vec![0.5, 0.5, 0.5], 11);
      boris.particles.add_particle(vec![0.25, 0.5, 1.0], 11);

      boris.find_particles();
      assert!(boris.e_finder.not_found.is_empty(), "both particles must locate");
      boris.evaluate_fields_at_particles();
      // The trilinear interpolation of the corner data is exact at these
      // points (bit-exact for the cube corners' 0/1 data).
      let p0 = boris.particles.get(0);
      assert_eq!(p0.data[EFIELD], 0.5);
      assert_eq!(p0.data[EFIELD + 1], 0.5);
      assert_eq!(p0.data[EFIELD + 2], 0.5);
      let p1 = boris.particles.get(1);
      assert_eq!(p1.data[EFIELD], 0.25);
      assert_eq!(p1.data[EFIELD + 1], 0.5);
      assert_eq!(p1.data[EFIELD + 2], 1.0);

      let _ = std::fs::remove_dir_all(&root_dir);
   }
}
