
use std::cell::UnsafeCell;
use std::sync::atomic::*;

type Index = u32;
type TwoIndices = u64;
type AtomicTwoIndices = AtomicU64;

const LOW_MASK: TwoIndices = Index::MAX as TwoIndices;

const fn from(combined_index: TwoIndices) -> (Index, Index) {
    ((combined_index & LOW_MASK) as Index, (combined_index >> Index::BITS) as Index)
}

const fn into(indices: (Index, Index)) -> TwoIndices {
    (indices.0 as TwoIndices) | ((indices.1 as TwoIndices) << Index::BITS)
}

struct AtomicRegion<const WITHIN_LEN: usize> {
    // contains two indices, namely the begin (inclusive) and end (exclusive) of the region
    data: AtomicTwoIndices
}

impl<const LEN: usize> AtomicRegion<LEN> {

    const fn new(from: Index, to: Index) -> Self {
        assert!(LEN <= Index::MAX as usize);
        assert!(from <= to);
        assert!(to <= LEN as Index);
        Self {
            data: AtomicTwoIndices::new(into((from, to)))
        }
    }

    fn empty(region_data: TwoIndices) -> bool {
        from(region_data).0 == from(region_data).1
    }

    fn full(region_data: TwoIndices) -> bool {
        from(region_data).0 + LEN as Index == from(region_data).1
    }

    fn exactly_one_entry(region_data: TwoIndices) -> bool {
        from(region_data).0 + 1 == from(region_data).1
    }

    fn exactly_one_free(region_data: TwoIndices) -> bool {
        from(region_data).0 + LEN as Index - 1 == from(region_data).1
    }

    fn dequeue_unchecked(region_data: TwoIndices) -> TwoIndices {
        debug_assert!(!Self::empty(region_data));
        debug_assert!(!Self::exactly_one_entry(region_data));
        into((from(region_data).0.checked_add(1).unwrap(), from(region_data).1))
    }

    fn enqueue_unchecked(region_data: TwoIndices) -> TwoIndices {
        debug_assert!(!Self::full(region_data));
        debug_assert!(!Self::exactly_one_free(region_data));
        into((from(region_data).0, from(region_data).1.checked_add(1).unwrap()))
    }

    #[allow(unused)]
    fn start(&self) -> Index {
        from(self.data.load(Ordering::SeqCst)).0
    }

    fn end(&self) -> Index {
        from(self.data.load(Ordering::SeqCst)).1
    }

    ///
    /// Atomically updates the region to include the element following its last one.
    /// 
    /// Assumes that for all states with `self.end() == expected_end`, the queue is
    /// has at least two spaces left (this is checked only in debug mode).
    /// 
    fn cond_atomic_enqueue(&self, expected_end: Index) -> Result</* new end after enqueue */ Index, /* loaded end */ Index> {
        let mut current = self.data.load(Ordering::SeqCst);
        if from(current).1 != expected_end {
            return Err(from(current).1);
        }
        debug_assert!(!Self::exactly_one_free(current));
        while let Err(new) = self.data.compare_exchange_weak(
            current, 
            Self::enqueue_unchecked(current), 
            Ordering::SeqCst, 
            Ordering::SeqCst
        ) {
            current = new;
            if from(current).1 != expected_end {
                return Err(from(current).1);
            }
            debug_assert!(!Self::exactly_one_free(current));
        }
        return Ok(from(current).1);
    }

    ///
    /// Atomically updates the region to not include its first element.
    /// Returns `Err` if the region has only one element left.
    /// 
    fn atomic_dequeue(&self) -> Result<Index, ()> {
        let mut current = self.data.load(Ordering::SeqCst);
        if Self::exactly_one_entry(current) {
            return Err(());
        }
        while let Err(new) = self.data.compare_exchange_weak(
            current, 
            Self::dequeue_unchecked(current), 
            Ordering::SeqCst, 
            Ordering::SeqCst
        ) {
            current = new;
            if Self::exactly_one_entry(current) {
                return Err(());
            }
        }
        return Ok(from(current).0);
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

    const fn new(status: OccupationStatus) -> Self {
        Self {
            status: AtomicU8::new(status.to_u8())
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
/// A lock-free, multi-consumer, multi-producer, fixed size queue that always contains
/// at least one element.
/// 
/// For internal reasons, this queue always has to contain at least one element. Thus,
/// [`LockfreeQueue::try_dequeue()`] will return `Err()` if there is only one element left.
/// Similarly, the queue's internal buffer is of size `LEN`, but the queue itself can
/// store at most `LEN - 1` elements, since at least one buffer entry must remain empty.
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
/// Entries after  must be `Occupied`, `PendingOccupation` or `Empty`.
/// Eventually (i.e. after all atomic operations are complete), `initialized_region` and `empty_region` must fill the whole space `0..LEN`.
///  
pub struct LockfreeQueue<T, const LEN: usize = 8> {
    data: [(AtomicOccupationStatus, UnsafeCell<Option<T>>); LEN],
    initialized_region: AtomicRegion<LEN>,
    empty_region: AtomicRegion<LEN>,
}

unsafe impl<T: Send> Send for LockfreeQueue<T> {}
unsafe impl<T: Send> Sync for LockfreeQueue<T> {}

impl<T, const LEN: usize> LockfreeQueue<T, LEN> {

    fn data_at(&self, location: Index) -> &(AtomicOccupationStatus, UnsafeCell<Option<T>>) {
        &self.data[location as usize % LEN]
    }

    pub fn new(initial_el: T) -> Self {
        let mut result = Self { 
            data: std::array::from_fn(|_| (AtomicOccupationStatus::new(OccupationStatus::Empty), UnsafeCell::new(None))), 
            initialized_region: AtomicRegion::new(0, 1), 
            empty_region: AtomicRegion::new(1, LEN as Index) 
        };
        result.data[0] = (AtomicOccupationStatus::new(OccupationStatus::Occupied), UnsafeCell::new(Some(initial_el)));
        return result;
    }

    pub fn try_dequeue(&self) -> Result<T, ()> {
        let mut take_location = self.initialized_region.atomic_dequeue()?;

        while let Err(_) = self.data_at(take_location).0.compare_exchange(OccupationStatus::Occupied, OccupationStatus::PendingEmpty) {
            // this is quite an edge case; However, it can occur that we get an index, but before
            // we set it to `PendingEmpty`, other threads cause `initialized_section` to wrap around
            // just before this retrieved entry. Now an `enqueue` operation in another thread is
            // allowed to add this index again to `initialized_section`, since it remains `Occupied`
            take_location = self.initialized_region.atomic_dequeue()?;
        }

        let result = unsafe { (*self.data_at(take_location).1.get()).take() }.unwrap();
        assert!(OccupationStatus::PendingEmpty == self.data_at(take_location).0.swap(OccupationStatus::Empty));

        let mut current_end = self.empty_region.end();

        while self.data_at(current_end).0.load() == OccupationStatus::Empty {
            match self.empty_region.cond_atomic_enqueue(current_end) {
                Ok(new_end) => current_end = new_end,
                Err(new_end) => current_end = new_end
            }
        }
        return Ok(result);
    }

    pub fn try_enqueue(&self, x: T) -> Result<(), T> {
        let mut put_location = self.empty_region.atomic_dequeue();
        if put_location.is_err() {
            return Err(x);
        }
        while let Err(_) = self.data_at(put_location.unwrap()).0.compare_exchange(OccupationStatus::Empty, OccupationStatus::PendingOccupation) {
            // same edge case situation as in `try_take()`
            put_location = self.empty_region.atomic_dequeue();
            if put_location.is_err() {
                return Err(x);
            }
        }
        assert!(unsafe { std::mem::replace(&mut*self.data_at(put_location.unwrap()).1.get(), Some(x)) }.is_none());
        assert!(OccupationStatus::PendingOccupation == self.data_at(put_location.unwrap()).0.swap(OccupationStatus::Occupied));

        let mut current_end = self.initialized_region.end();

        while self.data_at(current_end).0.load() == OccupationStatus::Occupied {
            match self.initialized_region.cond_atomic_enqueue(current_end) {
                Ok(new_end) => current_end = new_end,
                Err(new_end) => current_end = new_end
            }
        }
        return Ok(());
    }

    #[cfg(test)]
    fn check_state_sync(&self) {
        let (init_begin, init_end) = from(self.initialized_region.data.load(Ordering::SeqCst));
        let (empty_begin, empty_end) = from(self.empty_region.data.load(Ordering::SeqCst));
        assert!(init_end > init_begin);
        assert!(empty_end > empty_begin);
        assert_eq!(init_end, empty_begin);
        assert_eq!(init_begin + LEN as Index, empty_end);
        for i in init_begin..init_end {
            assert_eq!(self.data_at(i).0.load(), OccupationStatus::Occupied);
        }
        for i in empty_begin..empty_end {
            assert_eq!(self.data_at(i).0.load(), OccupationStatus::Empty);
        }
    }

    #[cfg(test)]
    fn print(&self) {
        println!("Initialized: {}..{}", self.initialized_region.start(), self.initialized_region.end());
        println!("Empty: {}..{}", self.empty_region.start(), self.empty_region.end());
        print!("[");
        for i in 0..LEN {
            print!("{:?}, ", self.data[i].0.load());
        }
        println!("]");
    }
}

#[cfg(test)]
use std::collections::HashSet;

#[test]
fn test_lockfree_buffer_short_singlethreaded() {
    let buffer: LockfreeQueue<i32, 3> = LockfreeQueue::new(0);
    buffer.try_enqueue(1).unwrap();
    assert_eq!(Err(2), buffer.try_enqueue(2));
    assert_eq!(Ok(0), buffer.try_dequeue());
    assert_eq!(Err(()), buffer.try_dequeue());
}

#[test]
fn test_lockfree_buffer_singlethreaded() {
    let buffer: LockfreeQueue<i32, 9> = LockfreeQueue::new(-1);
    assert_eq!(Err(()), buffer.try_dequeue());
    buffer.try_enqueue(0).unwrap();
    buffer.try_enqueue(1).unwrap();
    buffer.try_enqueue(2).unwrap();
    buffer.try_enqueue(3).unwrap();
    assert_eq!(Ok(-1), buffer.try_dequeue());
    buffer.try_enqueue(4).unwrap();
    buffer.try_enqueue(5).unwrap();
    buffer.try_enqueue(6).unwrap();
    buffer.try_enqueue(7).unwrap();
    assert_eq!(Err(8), buffer.try_enqueue(8));
    assert_eq!(Ok(0), buffer.try_dequeue());
    buffer.try_enqueue(8).unwrap();
    assert_eq!(Ok(1), buffer.try_dequeue());
    assert_eq!(Ok(2), buffer.try_dequeue());
    assert_eq!(Ok(3), buffer.try_dequeue());
    assert_eq!(Ok(4), buffer.try_dequeue());
    assert_eq!(Ok(5), buffer.try_dequeue());
    assert_eq!(Ok(6), buffer.try_dequeue());
    assert_eq!(Ok(7), buffer.try_dequeue());
    assert_eq!(Err(()), buffer.try_dequeue());
}

#[test]
fn test_lockfree_buffer_concurrent_state_singlethreaded() {
    // this state is the result of a thread starting to call `take()` on a one-element
    // buffer, but being interrupted before setting the element to `PendingEmpty`
    let buffer = LockfreeQueue {
        data: [
            (AtomicOccupationStatus::new(OccupationStatus::Occupied), UnsafeCell::new(Some(0))),
            (AtomicOccupationStatus::new(OccupationStatus::Occupied), UnsafeCell::new(Some(1))),
            (AtomicOccupationStatus::new(OccupationStatus::Empty), UnsafeCell::new(None)),
            (AtomicOccupationStatus::new(OccupationStatus::Empty), UnsafeCell::new(None)),
        ],
        empty_region: AtomicRegion { data: AtomicU64::new(into((2, 4))) },
        initialized_region: AtomicRegion { data: AtomicU64::new(into((1, 2))) },
    };
    // this gives `Err()`, as (depending on first thread) it might remove the last element
    assert_eq!(Err(()), buffer.try_dequeue());
    assert_eq!(Ok(()), buffer.try_enqueue(2));
    assert_eq!(Ok(1), buffer.try_dequeue());
    // at this point, there is nothing we can do until the first thread finished `try_take()`
    assert_eq!(Err(()), buffer.try_dequeue());
    assert_eq!(Err(3), buffer.try_enqueue(3));
}

#[test]
fn test_lockfree_buffer_sync() {
    const THREADS: usize = 64;
    const ELS_PER_THREAD: usize = 8096;

    let buffer: &'static LockfreeQueue<i32> = Box::leak(Box::new(LockfreeQueue::new(-1)));
    let counter: &'static AtomicI32 = Box::leak(Box::new(AtomicI32::new(0)));

    let mut join_handles = Vec::new();
    for _ in 0..THREADS {
        join_handles.push(std::thread::spawn(|| {
            let mut dequeued = HashSet::new();
            let mut failed_enqueued = HashSet::new();
            for _ in 0..ELS_PER_THREAD {
                match buffer.try_enqueue(counter.fetch_add(1, Ordering::Relaxed)) {
                    Ok(()) => {},
                    Err(x) => { assert!(failed_enqueued.insert(x)); }
                }
                if let Ok(x) = buffer.try_dequeue() {
                    assert!(dequeued.insert(x));
                }
            }
            return (dequeued, failed_enqueued);
        }))
    }
    let mut all_els = HashSet::new();
    
    let mut failed = false;
    for handle in join_handles {
        match handle.join() {
            Ok((dequeued, failed_enqueued)) => {
                for x in dequeued {
                    assert!(all_els.insert(x));
                }
                for x in failed_enqueued {
                    assert!(all_els.insert(x));
                }
            },
            Err(e) => {
                println!("Error: {:?}", e);
                failed = true;
            }
        }
    }
    assert!(!failed);
    
    buffer.print();
    buffer.check_state_sync();
    while let Ok(x) = buffer.try_dequeue() {
        assert!(all_els.insert(x));
    }
    // the last element remains inside the buffer
    for i in -1..((THREADS * ELS_PER_THREAD) as i32 - 1) {
        assert!(all_els.contains(&(i as i32)));
    }
    assert_eq!(THREADS * ELS_PER_THREAD, all_els.len());
}

#[test]
fn test_lockfree_buffer_single_producer_preserve_order() {
    const THREADS: usize = 64;
    const ELS_PER_THREAD: usize = 8096;

    let buffer: &'static LockfreeQueue<i32> = Box::leak(Box::new(LockfreeQueue::new(-1)));

    let mut join_handles = Vec::new();
    for thread_i in 0..THREADS {
        join_handles.push(std::thread::spawn(move || {
            let mut current = 0;
            let mut last_dequeued_by_thread = [-2; THREADS];
            while current < ELS_PER_THREAD {
                match buffer.try_enqueue(((current * THREADS) | thread_i) as i32) {
                    Err(_) => {},
                    Ok(()) => {
                        current += 1;
                    }
                }
                match buffer.try_dequeue() {
                    Err(()) => {},
                    Ok(-1) => {},
                    Ok(x) => {
                        let queued_thread = (x as usize) % THREADS;
                        assert!(last_dequeued_by_thread[queued_thread] < x);
                        last_dequeued_by_thread[queued_thread] = x;
                    }
                }
            }
        }))
    }
    let mut failed = false;
    for handle in join_handles {
        match handle.join() {
            Ok(()) => {},
            Err(e) => {
                println!("Error: {:?}", e);
                failed = true;
            }
        }
    }
    assert!(!failed);
    buffer.print();
    buffer.check_state_sync();
}