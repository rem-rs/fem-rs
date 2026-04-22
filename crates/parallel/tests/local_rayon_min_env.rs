//! `local_rayon_min()` uses a process-wide `OnceLock`.  Reading
//! `FEM_PARALLEL_LOCAL_RAYON_MIN` is validated here in a **dedicated** integration
//! test binary so no other test initializes the cache first.

#[test]
fn fem_parallel_local_rayon_min_respects_env() {
    std::env::set_var(
        fem_parallel::FEM_PARALLEL_LOCAL_RAYON_MIN,
        "1337",
    );
    assert_eq!(fem_parallel::local_rayon_min(), 1337);
}
