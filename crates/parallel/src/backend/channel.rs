//! In-process channel-based communication backend for [`ThreadLauncher`].
//!
//! [`ChannelBackend`] implements [`CommBackend`] using `Mutex` + `Condvar`
//! primitives so that N OS threads can each hold a `Comm` handle and
//! communicate as if they were MPI ranks — without any MPI installation.
//!
//! ## Design
//!
//! All N backends share a single [`ChannelShared`] via `Arc`.
//! Point-to-point sends push to a per-(src,dst) `VecDeque`; receives block on
//! a `Condvar` until a matching tag arrives.
//! Collectives use generation-counted mutex/condvar rendezvous patterns.
//! Allreduce results are cached in a dedicated field so late-arriving threads
//! can read them after the "last rank" resets the accumulator.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use fem_core::Rank;
use super::CommBackend;

// ── message queue ──────────────────────────────────────────────────────────────

/// Lock+condvar pair protecting a FIFO queue of `(tag, payload)` messages.
type MsgQueue = Arc<(Mutex<VecDeque<(i32, Vec<u8>)>>, Condvar)>;

// ── collective state helpers ──────────────────────────────────────────────────

struct AllReduceF64 {
    sum:     f64,
    result:  f64,   // cache for late threads
    arrived: usize,
    gen:     usize,
}

struct AllReduceI64 {
    sum:     i64,
    result:  i64,
    arrived: usize,
    gen:     usize,
}

// ── ChannelShared ─────────────────────────────────────────────────────────────

/// State shared across all ranks in one [`ThreadLauncher`] invocation.
pub(crate) struct ChannelShared {
    pub n: usize,

    // ── Point-to-point ───────────────────────────────────────────────────────
    // p2p[from][to] — queue of (tag, payload) messages.
    p2p: Vec<Vec<MsgQueue>>,

    // ── AllToAllv staging ────────────────────────────────────────────────────
    // a2av_slots[src][dst] — payload sent by `src` to `dst` (None = not sent).
    a2av_slots:   Vec<Vec<Mutex<Option<Vec<u8>>>>>,
    a2av_send:    Mutex<(usize /*arrived*/, usize /*gen*/)>,
    a2av_send_cv: Condvar,
    a2av_recv:    Mutex<(usize /*done*/,    usize /*gen*/)>,
    a2av_recv_cv: Condvar,

    // ── Barrier ──────────────────────────────────────────────────────────────
    barrier:    Mutex<(usize /*arrived*/, usize /*gen*/)>,
    barrier_cv: Condvar,

    // ── AllReduce f64 ─────────────────────────────────────────────────────────
    reduce_f64:    Mutex<AllReduceF64>,
    reduce_f64_cv: Condvar,

    // ── AllReduce i64 ─────────────────────────────────────────────────────────
    reduce_i64:    Mutex<AllReduceI64>,
    reduce_i64_cv: Condvar,

    // ── Broadcast ────────────────────────────────────────────────────────────
    bcast_data:    Mutex<Option<Vec<u8>>>,  // root deposits here
    bcast_ready:   Mutex<(usize /*count*/, usize /*gen*/)>,
    bcast_ready_cv:Condvar,
    bcast_done:    Mutex<(usize /*count*/, usize /*gen*/)>,
    bcast_done_cv: Condvar,
}

impl ChannelShared {
    /// Construct a fresh shared state for `n` ranks.
    pub fn new(n: usize) -> Arc<Self> {
        let p2p: Vec<Vec<MsgQueue>> = (0..n)
            .map(|_| {
                (0..n)
                    .map(|_| Arc::new((Mutex::new(VecDeque::new()), Condvar::new())))
                    .collect()
            })
            .collect();

        let a2av_slots: Vec<Vec<Mutex<Option<Vec<u8>>>>> = (0..n)
            .map(|_| (0..n).map(|_| Mutex::new(None)).collect())
            .collect();

        Arc::new(ChannelShared {
            n,
            p2p,
            a2av_slots,
            a2av_send:     Mutex::new((0, 0)),
            a2av_send_cv:  Condvar::new(),
            a2av_recv:     Mutex::new((0, 0)),
            a2av_recv_cv:  Condvar::new(),
            barrier:       Mutex::new((0, 0)),
            barrier_cv:    Condvar::new(),
            reduce_f64:    Mutex::new(AllReduceF64 { sum: 0.0, result: 0.0, arrived: 0, gen: 0 }),
            reduce_f64_cv: Condvar::new(),
            reduce_i64:    Mutex::new(AllReduceI64 { sum: 0, result: 0, arrived: 0, gen: 0 }),
            reduce_i64_cv: Condvar::new(),
            bcast_data:     Mutex::new(None),
            bcast_ready:    Mutex::new((0, 0)),
            bcast_ready_cv: Condvar::new(),
            bcast_done:     Mutex::new((0, 0)),
            bcast_done_cv:  Condvar::new(),
        })
    }
}

// ── ChannelBackend ────────────────────────────────────────────────────────────

/// Per-rank handle into the shared in-process communication state.
pub struct ChannelBackend {
    rank:   Rank,
    shared: Arc<ChannelShared>,
}

impl ChannelBackend {
    pub fn new(rank: Rank, shared: Arc<ChannelShared>) -> Self {
        ChannelBackend { rank, shared }
    }

    /// Internal generation-counted barrier used by collectives.
    fn rendezvous(&self, state: &Mutex<(usize, usize)>, cv: &Condvar) {
        let n = self.shared.n;
        let mut g = state.lock().unwrap();
        let my_gen = g.1;
        g.0 += 1;
        if g.0 == n {
            g.0 = 0;
            g.1 = my_gen.wrapping_add(1);
            cv.notify_all();
        } else {
            g = cv.wait_while(g, |s| s.1 == my_gen).unwrap();
        }
        drop(g);
    }
}

impl CommBackend for ChannelBackend {
    fn rank(&self) -> Rank { self.rank }
    fn size(&self) -> usize { self.shared.n }

    // ── barrier ───────────────────────────────────────────────────────────────

    fn barrier(&self) {
        self.rendezvous(&self.shared.barrier, &self.shared.barrier_cv);
    }

    // ── allreduce f64 ─────────────────────────────────────────────────────────

    fn allreduce_sum_f64(&self, local: f64) -> f64 {
        let n = self.shared.n;
        let mut st = self.shared.reduce_f64.lock().unwrap();
        let my_gen = st.gen;
        st.sum     += local;
        st.arrived += 1;
        if st.arrived == n {
            // Compute and cache the result, then reset the accumulator.
            let result = st.sum;
            st.result  = result;
            st.sum     = 0.0;
            st.arrived = 0;
            st.gen     = my_gen.wrapping_add(1);
            self.shared.reduce_f64_cv.notify_all();
            result
        } else {
            // Wait for the generation to advance, then read cached result.
            st = self.shared.reduce_f64_cv
                .wait_while(st, |s| s.gen == my_gen)
                .unwrap();
            st.result
        }
    }

    // ── allreduce i64 ─────────────────────────────────────────────────────────

    fn allreduce_sum_i64(&self, local: i64) -> i64 {
        let n = self.shared.n;
        let mut st = self.shared.reduce_i64.lock().unwrap();
        let my_gen = st.gen;
        st.sum     += local;
        st.arrived += 1;
        if st.arrived == n {
            let result = st.sum;
            st.result  = result;
            st.sum     = 0;
            st.arrived = 0;
            st.gen     = my_gen.wrapping_add(1);
            self.shared.reduce_i64_cv.notify_all();
            result
        } else {
            st = self.shared.reduce_i64_cv
                .wait_while(st, |s| s.gen == my_gen)
                .unwrap();
            st.result
        }
    }

    // ── broadcast ─────────────────────────────────────────────────────────────

    fn broadcast_bytes(&self, root: Rank, buf: &mut Vec<u8>) {
        let n = self.shared.n;

        // Phase 1: root deposits data, then all ranks rendezvous.
        if self.rank == root {
            *self.shared.bcast_data.lock().unwrap() = Some(buf.clone());
        }
        self.rendezvous(&self.shared.bcast_ready, &self.shared.bcast_ready_cv);

        // Phase 2: non-root ranks read from bcast_data.
        if self.rank != root {
            let data = self.shared.bcast_data.lock().unwrap()
                .as_ref()
                .expect("bcast_data must be set by root")
                .clone();
            *buf = data;
        }

        // Phase 3: all rendezvous, then root clears the buffer.
        {
            let mut g = self.shared.bcast_done.lock().unwrap();
            let my_gen = g.1;
            g.0 += 1;
            if g.0 == n {
                g.0 = 0;
                g.1 = my_gen.wrapping_add(1);
                *self.shared.bcast_data.lock().unwrap() = None;
                self.shared.bcast_done_cv.notify_all();
            } else {
                g = self.shared.bcast_done_cv
                    .wait_while(g, |s| s.1 == my_gen)
                    .unwrap();
            }
            drop(g);
        }
    }

    // ── point-to-point ────────────────────────────────────────────────────────

    fn send_bytes(&self, dest: Rank, tag: i32, data: &[u8]) {
        let queue = &self.shared.p2p[self.rank as usize][dest as usize];
        let (lock, cvar) = queue.as_ref();
        lock.lock().unwrap().push_back((tag, data.to_vec()));
        cvar.notify_one();
    }

    fn recv_bytes(&self, src: Rank, tag: i32) -> Vec<u8> {
        let queue = &self.shared.p2p[src as usize][self.rank as usize];
        let (lock, cvar) = queue.as_ref();
        let mut guard = cvar
            .wait_while(lock.lock().unwrap(), |q| !q.iter().any(|(t, _)| *t == tag))
            .unwrap();
        let pos = guard.iter().position(|(t, _)| *t == tag).unwrap();
        guard.remove(pos).unwrap().1
    }

    // ── all-to-all ────────────────────────────────────────────────────────────

    fn alltoallv_bytes(&self, sends: &[(Rank, Vec<u8>)]) -> Vec<(Rank, Vec<u8>)> {
        let n  = self.shared.n;
        let me = self.rank as usize;

        // Phase 1: deposit each send into the staging grid.
        for (dest, data) in sends {
            *self.shared.a2av_slots[me][*dest as usize].lock().unwrap() = Some(data.clone());
        }

        // Phase 2: wait for all ranks to finish depositing.
        self.rendezvous(&self.shared.a2av_send, &self.shared.a2av_send_cv);

        // Phase 3: collect all messages addressed to `me`.
        let mut result = Vec::new();
        for src in 0..n {
            let mut slot = self.shared.a2av_slots[src][me].lock().unwrap();
            if let Some(data) = slot.take() {
                result.push((src as Rank, data));
            }
        }

        // Phase 4: wait for all ranks to finish reading before slots can be
        // reused by a subsequent alltoallv call.
        self.rendezvous(&self.shared.a2av_recv, &self.shared.a2av_recv_cv);

        result
    }

    fn split(&self, color: i32, key: i32) -> Box<dyn CommBackend> {
        let n = self.shared.n;
        let me = self.rank as usize;
        let tag_split = 0x8000i32;

        // Phase 1: gather (color, key) from all ranks to rank 0.
        let my_data = [color.to_le_bytes(), key.to_le_bytes()].concat();
        if me != 0 {
            self.send_bytes(0, tag_split, &my_data);
        }

        let all_colors_keys: Vec<(i32, i32)>;
        if me == 0 {
            let mut ck = vec![(0i32, 0i32); n];
            ck[0] = (color, key);
            for r in 1..n {
                let bytes = self.recv_bytes(r as Rank, tag_split);
                let c = i32::from_le_bytes(bytes[..4].try_into().unwrap());
                let k = i32::from_le_bytes(bytes[4..8].try_into().unwrap());
                ck[r] = (c, k);
            }
            // Broadcast gathered data back.
            let bcast_bytes: Vec<u8> = ck.iter()
                .flat_map(|(c, k)| [c.to_le_bytes(), k.to_le_bytes()].concat())
                .collect();
            for r in 1..n {
                self.send_bytes(r as Rank, tag_split + 1, &bcast_bytes);
            }
            all_colors_keys = ck;
        } else {
            let bcast_bytes = self.recv_bytes(0, tag_split + 1);
            all_colors_keys = bcast_bytes.chunks_exact(8)
                .map(|chunk| {
                    let c = i32::from_le_bytes(chunk[..4].try_into().unwrap());
                    let k = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
                    (c, k)
                })
                .collect();
        }

        // Phase 2: compute sub-groups. Each rank computes the same grouping.
        // Collect unique colors and build (color → sorted list of (key, old_rank)).
        let mut groups: std::collections::BTreeMap<i32, Vec<(i32, usize)>>
            = std::collections::BTreeMap::new();
        for (old_rank, &(c, k)) in all_colors_keys.iter().enumerate() {
            groups.entry(c).or_default().push((k, old_rank));
        }
        for members in groups.values_mut() {
            members.sort_by_key(|(k, _)| *k);
        }

        // Phase 3: rank 0 creates ChannelShared for each group and distributes.
        // The Arc<ChannelShared> must be shared among all threads in the group.
        // Since we're in-process, rank 0 creates them and sends Arc pointers
        // via a shared staging area in the parent ChannelShared.

        // We use a simple approach: rank 0 creates all ChannelShared instances,
        // serializes Arc pointers as raw pointers (safe because all threads are
        // in the same process), and sends to each rank.
        let my_color = color;
        let my_group = &groups[&my_color];
        let new_size = my_group.len();
        let new_rank = my_group.iter()
            .position(|(_, r)| *r == me)
            .unwrap() as Rank;

        if me == 0 {
            // Create one ChannelShared per color group.
            let mut shared_map: std::collections::BTreeMap<i32, Arc<ChannelShared>>
                = std::collections::BTreeMap::new();
            for (&color, members) in &groups {
                shared_map.insert(color, ChannelShared::new(members.len()));
            }

            // Send the raw Arc pointer to each non-zero rank.
            for (old_rank, &(c, _k)) in all_colors_keys.iter().enumerate() {
                if old_rank == 0 { continue; }
                let arc = &shared_map[&c];
                let ptr = Arc::into_raw(Arc::clone(arc));
                let ptr_bytes = (ptr as usize).to_le_bytes();
                self.send_bytes(old_rank as Rank, tag_split + 2, &ptr_bytes);
            }

            // Rank 0's own shared.
            let my_shared = Arc::clone(&shared_map[&my_color]);
            Box::new(ChannelBackend::new(new_rank, my_shared))
        } else {
            // Receive the Arc pointer from rank 0.
            let ptr_bytes = self.recv_bytes(0, tag_split + 2);
            let ptr_val = usize::from_le_bytes(ptr_bytes.try_into().unwrap());
            // SAFETY: we're in the same process, rank 0 sent a valid Arc pointer.
            let my_shared = unsafe { Arc::from_raw(ptr_val as *const ChannelShared) };
            Box::new(ChannelBackend::new(new_rank, my_shared))
        }
    }
}
