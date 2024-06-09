#![feature(allocator_api)]
#![feature(ptr_alignment_type)]
#![feature(btree_cursors)]

use std::alloc::*;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::{BTreeMap, HashSet};
use std::ptr::{Alignment, NonNull};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, AtomicU8, Ordering};

struct AtomicRegion<const WITHIN_LEN: usize> {
    // contains two u32, namely the begin (inclusive) and end (exclusive) of the region region
    data: AtomicU64
}

const LOW_MASK: u64 = u32::MAX as u64;
const HIGH_MASK: u64 = LOW_MASK << u32::BITS;

impl<const WITHIN_LEN: usize> AtomicRegion<WITHIN_LEN> {

    const fn empty() -> Self {
        Self {
            data: AtomicU64::new(0)
        }
    }

    const fn full() -> Self {
        assert!(WITHIN_LEN <= u32::MAX as usize);
        Self {
            data: AtomicU64::new((WITHIN_LEN as u64 - 1) << u32::BITS)
        }
    }

    fn is_empty(region_data: u64) -> bool {
        (region_data >> u32::BITS) == (region_data & LOW_MASK)
    }

    fn dequeue_unchecked(region_data: u64) -> u64 {
        assert!(WITHIN_LEN <= u32::MAX as usize);
        debug_assert!((region_data >> u32::BITS) != (region_data & LOW_MASK));
        (region_data & HIGH_MASK) | (((region_data & LOW_MASK) + 1) % (WITHIN_LEN as u64))
    }

    ///
    /// Atomically updates the region to not include its first element.
    /// Returns `Err` if the region is empty.
    /// 
    fn atomic_dequeue(&self) -> Result<usize, ()> {
        let mut current = self.data.load(Ordering::SeqCst);
        if Self::is_empty(current) {
            return Err(());
        }
        while let Err(new) = self.data.compare_exchange_weak(
            current, 
            Self::dequeue_unchecked(current), 
            Ordering::SeqCst, 
            Ordering::SeqCst
        ) {
            current = new;
            if Self::is_empty(current) {
                return Err(());
            }
        }
        return Ok((current & LOW_MASK) as usize);
    }

    fn is_full(region_data: u64) -> bool {
        ((region_data >> u32::BITS) + 1) % (WITHIN_LEN as u64) == (region_data & LOW_MASK)
    }

    fn enqueue_unchecked(region_data: u64) -> u64 {
        assert!(WITHIN_LEN <= u32::MAX as usize);
        debug_assert!(((region_data >> u32::BITS) + 1) % (WITHIN_LEN as u64) != (region_data & LOW_MASK));
        ((((region_data >> u32::BITS) + 1) % (WITHIN_LEN as u64)) << u32::BITS) | (region_data & LOW_MASK)
    }

    fn start(&self) -> usize {
        (self.data.load(Ordering::SeqCst) & LOW_MASK) as usize
    }

    fn end(&self) -> usize {
        (self.data.load(Ordering::SeqCst) >> u32::BITS) as usize
    }

    ///
    /// Atomically updates the region to include the element following its last one.
    /// Returns `Err` if the region is full within the given size.
    /// 
    /// Assumes that for all states with `self.end() == expected_end`, the queue is
    /// not yet full (checked in debug mode).
    /// 
    fn cond_atomic_enqueue(&self, expected_end: usize) -> Result</* new end after enqueue */ usize, /* loaded end */ usize> {
        let expected_end_u64 = expected_end as u64;
        let mut current = self.data.load(Ordering::SeqCst);
        if (current >> u32::BITS) != expected_end_u64 {
            return Err((current >> u32::BITS) as usize);
        }
        debug_assert!(!Self::is_full(current));
        while let Err(new) = self.data.compare_exchange_weak(
            current, 
            Self::enqueue_unchecked(current), 
            Ordering::SeqCst, 
            Ordering::SeqCst
        ) {
            current = new;
            if (current >> u32::BITS) != expected_end_u64 {
                return Err((current >> u32::BITS) as usize);
            }
            debug_assert!(!Self::is_full(current));
        }
        return Ok((current >> u32::BITS) as usize);
    }
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum OccupationStatus {
    Empty = 0,
    PendingOccupation = 1,
    Occupied = 2,
    PendingEmpty = 3
}

impl OccupationStatus {

    const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Empty,
            1 => Self::PendingOccupation,
            2 => Self::Occupied,
            3 => Self::PendingEmpty,
            _ => panic!("illegal enum value")
        }
    }

    const fn to_u8(self) -> u8 {
        match self {
            OccupationStatus::Empty => 0,
            OccupationStatus::PendingOccupation => 1,
            OccupationStatus::Occupied => 2,
            OccupationStatus::PendingEmpty => 3
        }
    }
}

struct AtomicOccupationStatus {
    status: AtomicU8
}

impl AtomicOccupationStatus {

    const fn empty() -> Self {
        Self {
            status: AtomicU8::new(OccupationStatus::Empty.to_u8())
        }
    }

    fn swap(&self, new: OccupationStatus) -> OccupationStatus {
        OccupationStatus::from_u8(self.status.swap(new.to_u8(), Ordering::SeqCst))
    }

    fn compare_exchange(&self, old: OccupationStatus, new: OccupationStatus) -> Result<OccupationStatus, OccupationStatus> {
        self.status.compare_exchange(old.to_u8(), new.to_u8(), Ordering::SeqCst, Ordering::SeqCst)
            .map(|success_status| OccupationStatus::from_u8(success_status))
            .map_err(|failure_status| OccupationStatus::from_u8(failure_status))
    }

    fn load(&self) -> OccupationStatus {
        OccupationStatus::from_u8(self.status.load(Ordering::SeqCst))
    }
}

///
/// A lock-free, multi-consumer, multi-producer, fixed size buffer.
/// 
/// Note also that the queue can store at most `LEN - 1` elements, 
/// since one entry always has to stay empty to keep up the internal 
/// invariants.
/// 
/// # Implementation contract
///
/// Each entry of `data` has an [`OccupationStatus`]
///  - `Empty` means it may be filled by any thread
///  - `PendingOccupation` means one thread is currently writing to this entry
///  - `Occupied` means it may be emptied by any thread
///  - `PendingEmpty` means some thread is currently taking the element (i.e. reading from the entry)
/// 
/// Entries within `initialized_region` must always be `Occupied`.
/// Entries within `empty_region` must always be `Empty`.
/// Entries after `initialized_region` and before `empty_region` must be either `Occupied` or `PendingOccupation`.
/// Eventually (i.e. after all atomic operations are complete), this area must be of size zero.
/// Entries after `empty_region` and before `initialized_region` must be either `Empty` or `PendingEmpty`.
/// Eventually, this area must be of size zero.
///  
pub struct LockFreeQueue<T, const LEN: usize = 8> {
    data: [(AtomicOccupationStatus, UnsafeCell<Option<T>>); LEN],
    initialized_region: AtomicRegion<LEN>,
    empty_region: AtomicRegion<LEN>,
}

unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T, const LEN: usize> LockFreeQueue<T, LEN> {

    pub fn new() -> Self {
        Self { 
            data: std::array::from_fn(|_| (AtomicOccupationStatus::empty(), UnsafeCell::new(None))), 
            initialized_region: AtomicRegion::empty(), 
            empty_region: AtomicRegion::full() 
        }
    }

    pub fn try_deque(&self) -> Result<T, ()> {
        let mut index = self.initialized_region.atomic_dequeue()?;
        while let Err(_) = self.data[index as usize].0.compare_exchange(OccupationStatus::Occupied, OccupationStatus::PendingEmpty) {
            // this is quite an edge case; However, it can occur that we get an index, but before
            // we set it to `PendingEmpty`, other threads cause `initialized_section` to wrap around
            // just before this retrieved entry. Now an `enqueue` operation in another thread is
            // allowed to add this index again to `initialized_section`, since it remains `Occupied`
            index = self.initialized_region.atomic_dequeue()?;
        }
        let (entry_init, entry_data) = &self.data[index as usize];

        let result = unsafe { (*entry_data.get()).take() }.unwrap();
        assert!(OccupationStatus::PendingEmpty == entry_init.swap(OccupationStatus::Empty));

        let mut current_end = self.empty_region.end();
        while self.data[current_end].0.load() == OccupationStatus::Empty && (current_end + 1) % LEN != self.initialized_region.start() {
            match self.empty_region.cond_atomic_enqueue(current_end) {
                Ok(new_end) => current_end = new_end,
                Err(new_end) => current_end = new_end
            }
        }
        return Ok(result);
    }

    pub fn try_enqueue(&self, x: T) -> Result<(), T> {
        let mut index = self.empty_region.atomic_dequeue();
        if index.is_err() {
            return Err(x);
        }
        while let Err(_) = self.data[index.unwrap() as usize].0.compare_exchange(OccupationStatus::Empty, OccupationStatus::PendingOccupation) {
            // same edge case situation as in `try_deque()`
            index = self.empty_region.atomic_dequeue();
            if index.is_err() {
                return Err(x);
            }
        }
        let index = index.unwrap();
        let (entry_init, entry_data) = &self.data[index as usize];

        unsafe { (*entry_data.get()) = Some(x); }
        assert!(OccupationStatus::PendingOccupation == entry_init.swap(OccupationStatus::Occupied));

        let mut current_end = self.initialized_region.end();
        while self.data[current_end].0.load() == OccupationStatus::Occupied {
            match self.initialized_region.cond_atomic_enqueue(current_end) {
                Ok(new_end) => current_end = new_end,
                Err(new_end) => current_end = new_end
            }
        }
        return Ok(());
    }
}

pub struct FixedSizeMempool<A: Allocator = Global> {
    base_alloc: A,
    layout: Layout,
    mempool: Box<[(AtomicBool, Option<Box<[u8], A>>)]>
}

unsafe impl<A: Allocator> Allocator for FixedSizeMempool<A> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unimplemented!()
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        
    }
}

#[test]
fn test_lockfree_queue() {
    let queue: LockFreeQueue<i32> = LockFreeQueue::new();
    queue.try_enqueue(0).unwrap();
    queue.try_enqueue(1).unwrap();
    queue.try_enqueue(2).unwrap();
    queue.try_enqueue(3).unwrap();
    assert_eq!(Ok(0), queue.try_deque());
    queue.try_enqueue(4).unwrap();
    queue.try_enqueue(5).unwrap();
    queue.try_enqueue(6).unwrap();
    queue.try_enqueue(7).unwrap();
    assert_eq!(Err(8), queue.try_enqueue(8));
    assert_eq!(Ok(1), queue.try_deque());
    queue.try_enqueue(8).unwrap();
    assert_eq!(Ok(2), queue.try_deque());
    assert_eq!(Ok(3), queue.try_deque());
    assert_eq!(Ok(4), queue.try_deque());
    assert_eq!(Ok(5), queue.try_deque());
    assert_eq!(Ok(6), queue.try_deque());
    assert_eq!(Ok(7), queue.try_deque());
    assert_eq!(Ok(8), queue.try_deque());
    assert_eq!(Err(()), queue.try_deque());
}

#[test]
fn test_lockfree_queue_sync() {
    const THREADS: usize = 4;
    const ELS_PER_THREAD: usize = 8096;

    let queue: &'static LockFreeQueue<i32> = Box::leak(Box::new(LockFreeQueue::new()));
    let counter: &'static AtomicI32 = Box::leak(Box::new(AtomicI32::new(0)));

    let mut join_handles = Vec::new();
    for _ in 0..THREADS {
        join_handles.push(std::thread::spawn(|| {
            let mut dequeued = HashSet::new();
            let mut failed_enqueued = HashSet::new();
            for _ in 0..ELS_PER_THREAD {
                match queue.try_enqueue(counter.fetch_add(1, Ordering::Relaxed)) {
                    Ok(()) => {},
                    Err(x) => { assert!(failed_enqueued.insert(x)); }
                }
                if let Ok(x) = queue.try_deque() {
                    assert!(dequeued.insert(x));
                }
            }
            return (dequeued, failed_enqueued);
        }))
    }
    let mut all_els = HashSet::new();
    for handle in join_handles {
        let (dequeued, failed_enqueued) = handle.join().unwrap();
        for x in dequeued {
            assert!(all_els.insert(x));
        }
        for x in failed_enqueued {
            assert!(all_els.insert(x));
        }
    }
    while let Ok(x) = queue.try_deque() {
        assert!(all_els.insert(x));
    }
    for i in 0..(THREADS * ELS_PER_THREAD) {
        assert!(all_els.contains(&(i as i32)));
    }
    assert_eq!(THREADS * ELS_PER_THREAD, all_els.len());
}