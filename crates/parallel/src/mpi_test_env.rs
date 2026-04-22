//! Test-only helpers for MPI-backed [`crate::Comm`].
//!
//! `MpiLauncher::init().world_comm()` drops the launcher at the end of the
//! statement, which tears down `mpi::environment::Universe` and calls
//! `MPI_Finalize` while the returned [`crate::Comm`] is still used later in the
//! test.  When the `mpi` feature is enabled, [`test_world_comm`] keeps the
//! universe alive for the whole test process.

use crate::Comm;

#[cfg(all(feature = "mpi", not(target_arch = "wasm32")))]
use std::sync::OnceLock;

#[cfg(all(feature = "mpi", not(target_arch = "wasm32")))]
use crate::backend::native::NativeMpiBackend;

/// World communicator for unit tests.
pub(crate) fn test_world_comm() -> Comm {
    #[cfg(all(feature = "mpi", not(target_arch = "wasm32")))]
    {
        static UNIVERSE: OnceLock<::mpi::environment::Universe> = OnceLock::new();
        let uni = UNIVERSE.get_or_init(|| {
            ::mpi::initialize().expect(
                "MPI initialize() returned None (already finalized or unsupported state)",
            )
        });
        Comm::from_backend(Box::new(NativeMpiBackend::from_world(uni)))
    }

    #[cfg(not(all(feature = "mpi", not(target_arch = "wasm32"))))]
    {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use crate::launcher::{Launcher, native::MpiLauncher};
            MpiLauncher::init()
                .expect("MPI already initialised")
                .world_comm()
        }
        #[cfg(target_arch = "wasm32")]
        {
            use crate::launcher::{Launcher, wasm::WorkerLauncher};
            WorkerLauncher::init()
                .expect("WorkerLauncher init")
                .world_comm()
        }
    }
}
