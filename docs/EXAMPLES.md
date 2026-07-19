# fem-rs Example Index

120 registered examples (119 `mfem_ex`/`mfem_pex` aligned + native examples).

## Poisson / Scalar Diffusion (ex1)

| Example | Description |
|---------|-------------|
| [`mfem_ex1_poisson`](../examples/mfem_ex1_poisson.rs) | Poisson 2D/3D, CG + ILU0 |
| [`mfem_ex1_gpu`](../examples/mfem_ex1_gpu.rs) | Poisson GPU (wgpu backend) |
| [`mfem_ex1_poisson_gpu`](../examples/mfem_ex1_poisson_gpu.rs) | Poisson GPU PA operator |

## Elasticity (ex2)

| Example | Description |
|---------|-------------|
| [`mfem_ex2_elasticity`](../examples/mfem_ex2_elasticity.rs) | Linear elasticity 2D/3D |
| [`mfem_pex2_parallel_elasticity`](../examples/mfem_pex2_parallel_elasticity.rs) | Parallel elasticity |
| [`mfem_pex2_mixed_darcy`](../examples/mfem_pex2_mixed_darcy.rs) | Mixed Darcy (Hdiv) |

## Maxwell / EM (ex3)

| Example | Description |
|---------|-------------|
| [`mfem_ex3_maxwell_cavity`](../examples/mfem_ex3_maxwell_cavity.rs) | HCurl cavity modes |
| [`mfem_pex3_maxwell_cavity`](../examples/mfem_pex3_maxwell_cavity.rs) | Parallel HCurl |
| [`mfem_ex22_complex_helmholtz`](../examples/mfem_ex22_complex_helmholtz.rs) | Complex Helmholtz |
| [`mfem_pex22_parallel_complex_helmholtz`](../examples/mfem_pex22_parallel_complex_helmholtz.rs) | Parallel complex |
| [`mfem_ex25_pml_helmholtz`](../examples/mfem_ex25_pml_helmholtz.rs) | PML Helmholtz |
| [`mfem_ex31_anisotropic_maxwell`](../examples/mfem_ex31_anisotropic_maxwell.rs) | Anisotropic Maxwell |
| [`mfem_ex32_impedance_maxwell`](../examples/mfem_ex32_impedance_maxwell.rs) | Impedance BC |
| [`mfem_ex33_fractional_laplacian`](../examples/mfem_ex33_fractional_laplacian.rs) | Fractional Laplacian |
| [`mfem_ex33_tangential_drive_maxwell`](../examples/mfem_ex33_tangential_drive_maxwell.rs) | Tangential drive |
| [`mfem_ex34_absorbing_maxwell`](../examples/mfem_ex34_absorbing_maxwell.rs) | Absorbing BC |
| [`mfem_maxwell`](../examples/mfem_maxwell.rs) | Electromagnetics miniapp |
| [`mfem_joule`](../examples/mfem_joule.rs) | Joule heating miniapp |
| [`mfem_tesla`](../examples/mfem_tesla.rs) | Tesla coil miniapp |
| [`mfem_volta`](../examples/mfem_volta.rs) | Volta miniapp |

## Darcy / Mixed (ex4–5)

| Example | Description |
|---------|-------------|
| [`mfem_ex4_darcy`](../examples/mfem_ex4_darcy.rs) | Darcy flow |
| [`mfem_ex5_mixed_darcy`](../examples/mfem_ex5_mixed_darcy.rs) | Mixed Darcy (RT0) |
| [`mfem_pex5_hdiv_darcy`](../examples/mfem_pex5_hdiv_darcy.rs) | Parallel Hdiv Darcy |

## Mixed BC (ex7)

| Example | Description |
|---------|-------------|
| [`mfem_ex7_neumann_mixed_bc`](../examples/mfem_ex7_neumann_mixed_bc.rs) | Neumann/mixed BC |
| [`mfem_pex7_parallel_surface`](../examples/mfem_pex7_parallel_surface.rs) | Parallel surface BC |

## Hybridization (ex8)

| Example | Description |
|---------|-------------|
| [`mfem_ex8_hybridization`](../examples/mfem_ex8_hybridization.rs) | HDG hybridization |

## DG (ex9, ex14–15, ex17–18)

| Example | Description |
|---------|-------------|
| [`mfem_ex9_dg_advection`](../examples/mfem_ex9_dg_advection.rs) | DG advection |
| [`mfem_pex9_parallel_dg_advection`](../examples/mfem_pex9_parallel_dg_advection.rs) | Parallel DG |
| [`mfem_ex14_dg_poisson`](../examples/mfem_ex14_dg_poisson.rs) | DG Poisson |
| [`mfem_ex15_dg_amr`](../examples/mfem_ex15_dg_amr.rs) | DG AMR |
| [`mfem_ex15_tet_nc_amr`](../examples/mfem_ex15_tet_nc_amr.rs) | Tet NC AMR |
| [`mfem_ex15_dynamic_amr`](../examples/mfem_ex15_dynamic_amr.rs) | Dynamic AMR |
| [`mfem_pex15_parallel_dynamic_amr`](../examples/mfem_pex15_parallel_dynamic_amr.rs) | Parallel dynamic AMR |
| [`mfem_ex17_dg_elasticity`](../examples/mfem_ex17_dg_elasticity.rs) | DG elasticity |
| [`mfem_pex17_parallel_dg_elasticity`](../examples/mfem_pex17_parallel_dg_elasticity.rs) | Parallel DG elasticity |
| [`mfem_ex18_euler`](../examples/mfem_ex18_euler.rs) | Euler DG CFD |
| [`mfem_pex18_parallel_euler`](../examples/mfem_pex18_parallel_euler.rs) | Parallel Euler |

## Nonlinear / Hyperelastic (ex10, ex16, ex19)

| Example | Description |
|---------|-------------|
| [`mfem_ex10_hyperelastic_dyn`](../examples/mfem_ex10_hyperelastic_dyn.rs) | Hyperelastic dynamics |
| [`mfem_pex10_parallel_hyperelastic`](../examples/mfem_pex10_parallel_hyperelastic.rs) | Parallel hyperelastic |
| [`mfem_ex16_nonlinear_heat`](../examples/mfem_ex16_nonlinear_heat.rs) | Nonlinear heat |
| [`mfem_pex16_parallel_nonlinear_heat`](../examples/mfem_pex16_parallel_nonlinear_heat.rs) | Parallel nonlinear heat |
| [`mfem_ex19_hyperelastic_dyn_incomp`](../examples/mfem_ex19_hyperelastic_dyn_incomp.rs) | Incompressible hyperelastic |

## Multigrid (ex11, ex26)

| Example | Description |
|---------|-------------|
| [`mfem_ex11_p_multigrid`](../examples/mfem_ex11_p_multigrid.rs) | p-multigrid |
| [`mfem_pex8_parallel_p_multigrid`](../examples/mfem_pex8_parallel_p_multigrid.rs) | Parallel p-multigrid |
| [`mfem_ex26_geom_mg`](../examples/mfem_ex26_geom_mg.rs) | Geometric multigrid |

## Eigenvalue (ex12–13)

| Example | Description |
|---------|-------------|
| [`mfem_ex12_elastic_eigen`](../examples/mfem_ex12_elastic_eigen.rs) | Elastic eigenmodes |
| [`mfem_ex13_eigenvalue`](../examples/mfem_ex13_eigenvalue.rs) | Eigenvalue solver |
| [`mfem_ex13_laplacian_eigen`](../examples/mfem_ex13_laplacian_eigen.rs) | Laplacian eigen |
| [`mfem_pex11_parallel_eigenvalue`](../examples/mfem_pex11_parallel_eigenvalue.rs) | Parallel eigenvalue |

## ODE / Time stepping (ex20–21, ex23)

| Example | Description |
|---------|-------------|
| [`mfem_ex20_symplectic`](../examples/mfem_ex20_symplectic.rs) | Symplectic integrator |
| [`mfem_pex20_parallel_symplectic`](../examples/mfem_pex20_parallel_symplectic.rs) | Parallel symplectic |
| [`mfem_ex21_amr_elasticity`](../examples/mfem_ex21_amr_elasticity.rs) | AMR elasticity |
| [`mfem_ex23_wave_equation`](../examples/mfem_ex23_wave_equation.rs) | Wave equation |
| [`mfem_pex23_parallel_wave`](../examples/mfem_pex23_parallel_wave.rs) | Parallel wave |

## Discrete / Geometry (ex24, ex29)

| Example | Description |
|---------|-------------|
| [`mfem_ex24_discrete_ops`](../examples/mfem_ex24_discrete_ops.rs) | Discrete operators |
| [`mfem_ex29_curved_poisson`](../examples/mfem_ex29_curved_poisson.rs) | Curved mesh Poisson |

## Robin / Contact / Sliding (ex27–28)

| Example | Description |
|---------|-------------|
| [`mfem_ex27_robin_bc`](../examples/mfem_ex27_robin_bc.rs) | Robin BC |
| [`mfem_pex27_parallel_robin_bc`](../examples/mfem_pex27_parallel_robin_bc.rs) | Parallel Robin |
| [`mfem_ex28_sliding_elasticity`](../examples/mfem_ex28_sliding_elasticity.rs) | Sliding contact |

## Multidomain / Optimization / Immersed (ex35–38)

| Example | Description |
|---------|-------------|
| [`mfem_ex35_multidomain`](../examples/mfem_ex35_multidomain.rs) | Multidomain coupling |
| [`mfem_ex36_obstacle`](../examples/mfem_ex36_obstacle.rs) | Obstacle problem |
| [`mfem_ex37_topology_optimization`](../examples/mfem_ex37_topology_optimization.rs) | Topology optimization |
| [`mfem_ex38_immersed_boundary`](../examples/mfem_ex38_immersed_boundary.rs) | Immersed boundary |

## Named attributes / I/O (ex39, ex43)

| Example | Description |
|---------|-------------|
| [`mfem_ex39_named_attributes`](../examples/mfem_ex39_named_attributes.rs) | Named attributes |
| [`mfem_ex43_hdf5_checkpoint`](../examples/mfem_ex43_hdf5_checkpoint.rs) | HDF5 checkpoint |

## Stokes / Navier–Stokes (ex40)

| Example | Description |
|---------|-------------|
| [`mfem_ex40_stokes`](../examples/mfem_ex40_stokes.rs) | Stokes flow |
| [`navier_stokes_kovasznay`](../examples/navier_stokes_kovasznay.rs) | NS Kovasznay MMS |

## IMEX / Thermoelastic / ALE (ex41)

| Example | Description |
|---------|-------------|
| [`mfem_ex41_imex`](../examples/mfem_ex41_imex.rs) | IMEX time stepping |
| [`ex_thermoelastic_coupled`](../examples/ex_thermoelastic_coupled.rs) | Thermoelastic (custom) |
| [`mfem_ex45_moving_mesh_ale`](../examples/mfem_ex45_moving_mesh_ale.rs) | ALE moving mesh |
| [`mfem_ex46_moving_mesh_heat`](../examples/mfem_ex46_moving_mesh_heat.rs) | Moving mesh heat |

## Multiphysics (ex47–53)

| Example | Description |
|---------|-------------|
| [`mfem_ex47_multiphysics_templates`](../examples/mfem_ex47_multiphysics_templates.rs) | Multiphysics template |
| [`mfem_ex48_template_joule_heating`](../examples/mfem_ex48_template_joule_heating.rs) | Joule heating FSI |
| [`mfem_ex49_template_fsi`](../examples/mfem_ex49_template_fsi.rs) | FSI template |
| [`mfem_ex50_template_acoustics_structure`](../examples/mfem_ex50_template_acoustics_structure.rs) | Acoustics–structure |
| [`mfem_ex51_template_em_thermal_stress`](../examples/mfem_ex51_template_em_thermal_stress.rs) | EM–thermal–stress |
| [`mfem_ex52_template_reaction_flow_thermal`](../examples/mfem_ex52_template_reaction_flow_thermal.rs) | Reaction–flow–thermal |
| [`mfem_ex53_3d_electrothermal`](../examples/mfem_ex53_3d_electrothermal.rs) | 3D electrothermal |

## IGA (iga)

| Example | Description |
|---------|-------------|
| [`mfem_ex_iga_poisson_1d`](../examples/mfem_ex_iga_poisson_1d.rs) | IGA Poisson 1D |
| [`mfem_ex_iga_poisson_2d_patch`](../examples/mfem_ex_iga_poisson_2d_patch.rs) | IGA Poisson 2D |
| [`mfem_ex_iga_helmholtz_2d`](../examples/mfem_ex_iga_helmholtz_2d.rs) | IGA Helmholtz 2D |
| [`mfem_ex_iga_poisson_3d`](../examples/mfem_ex_iga_poisson_3d.rs) | IGA Poisson 3D |
| [`mfem_ex_iga_heat_2d`](../examples/mfem_ex_iga_heat_2d.rs) | IGA heat 2D |

## Mesh / Geometry

| Example | Description |
|---------|-------------|
| [`mfem_ex0_mesh_intro`](../examples/mfem_ex0_mesh_intro.rs) | Mesh intro (load/plot) |
| [`mesh_quality_tmop`](../examples/mesh_quality_tmop.rs) | Mesh quality metrics |
| [`meshing_tmop_target_matrix`](../examples/meshing_tmop_target_matrix.rs) | TMOP target matrix |
| [`mfem_tmop_mesh_quality`](../examples/mfem_tmop_mesh_quality.rs) | TMOP mesh quality |
| [`tmop_hex8_optimise`](../examples/tmop_hex8_optimise.rs) | Hex8 TMOP optimisation |
| [`mfem_vec_ref_elem`](../examples/mfem_vec_ref_elem.rs) | Reference element utility |

## Parallel / HPC (pex)

| Example | Description |
|---------|-------------|
| [`mfem_pex1_parallel_poisson`](../examples/mfem_pex1_parallel_poisson.rs) | Parallel Poisson |
| [`mfem_pex2_parallel_elasticity`](../examples/mfem_pex2_parallel_elasticity.rs) | Parallel elasticity |
| [`mfem_pex4_parallel_heat`](../examples/mfem_pex4_parallel_heat.rs) | Parallel heat |
| [`mfem_pex6_parallel_amr`](../examples/mfem_pex6_parallel_amr.rs) | Parallel AMR |
| [`mfem_pex7_parallel_surface`](../examples/mfem_pex7_parallel_surface.rs) | Parallel surface BC |
| [`mfem_pex8_parallel_p_multigrid`](../examples/mfem_pex8_parallel_p_multigrid.rs) | Parallel p-multigrid |
| [`mfem_pex9_parallel_dg_advection`](../examples/mfem_pex9_parallel_dg_advection.rs) | Parallel DG |

## Stokes / Mixed

| Example | Description |
|---------|-------------|
| [`wg_stokes_cavity`](../examples/wg_stokes_cavity.rs) | WG Stokes cavity |
| [`dpg_stokes_2d`](../examples/dpg_stokes_2d.rs) | DPG Stokes 2D |
| [`hdg_stokes_channel`](../examples/hdg_stokes_channel.rs) | HDG Stokes channel |

## Plasticity / Damage

| Example | Description |
|---------|-------------|
| [`plasticity_j2_bar`](../examples/plasticity_j2_bar.rs) | J2 plasticity bar |
| [`plasticity_dp_slope`](../examples/plasticity_dp_slope.rs) | Drucker–Prager slope |
| [`crystal_plasticity_fcc`](../examples/crystal_plasticity_fcc.rs) | Crystal plasticity FCC |

## Phase field / Fracture

| Example | Description |
|---------|-------------|
| [`mfem_phasefield_fracture`](../examples/mfem_phasefield_fracture.rs) | Phase-field fracture |
| [`allen_cahn_evolution`](../examples/allen_cahn_evolution.rs) | Allen–Cahn evolution |
| [`cahn_hilliard_spinodal`](../examples/cahn_hilliard_spinodal.rs) | Cahn–Hilliard spinodal |

## XFEM / Contact

| Example | Description |
|---------|-------------|
| [`xfem_crack_propagation`](../examples/xfem_crack_propagation.rs) | XFEM crack propagation |
| [`contact_active_set_3d`](../examples/contact_active_set_3d.rs) | 3D active-set contact |

## Stochastic / UQ

| Example | Description |
|---------|-------------|
| [`mfem_mc_random_field`](../examples/mfem_mc_random_field.rs) | MC random field generation |
| [`spde_gaussian_field`](../examples/spde_gaussian_field.rs) | SPDE Gaussian field |

## Misc

| Example | Description |
|---------|-------------|
| [`heat_equation`](../examples/heat_equation.rs) | Heat equation (native) |
| [`wave_equation`](../examples/wave_equation.rs) | Wave equation (native) |
| [`fluids_navier_transient`](../examples/fluids_navier_transient.rs) | Transient NS |
| [`maxwell_time_domain`](../examples/maxwell_time_domain.rs) | Time-domain Maxwell |
| [`hyperelastic_hooke`](../examples/hyperelastic_hooke.rs) | Hookean hyperelastic |
| [`solver_comparison`](../examples/solver_comparison.rs) | Solver benchmark |
| [`plor_hex_solve`](../examples/plor_hex_solve.rs) | PLOR hex solve |
| [`shifted_sbm_diffusion`](../examples/shifted_sbm_diffusion.rs) | Shifted SBM diffusion |
| [`gslib_field_transfer`](../examples/gslib_field_transfer.rs) | GSLIB field transfer |
| [`multiphysics_coupled_newton`](../examples/multiphysics_coupled_newton.rs) | Coupled Newton |
| [`dc_current`](../examples/dc_current.rs) | DC current flow |
| [`dpg_poisson_2d`](../examples/dpg_poisson_2d.rs) | DPG Poisson |
| [`hdg_elasticity_beam`](../examples/hdg_elasticity_beam.rs) | HDG elasticity beam |
| [`vem_poisson_polygonal`](../examples/vem_poisson_polygonal.rs) | VEM polygonal |
| [`wgpu_poisson`](../examples/wgpu_poisson.rs) | wgpu Poisson |
