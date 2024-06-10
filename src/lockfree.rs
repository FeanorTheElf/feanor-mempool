
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::*;

type Index = u32;
type TwoIndices = u64;
type AtomicTwoIndices = AtomicU64;

const LOW_MASK: TwoIndices = Index::MAX as TwoIndices;

const fn unpack(combined_index: TwoIndices) -> (Index, Index) {
    ((combined_index & LOW_MASK) as Index, (combined_index >> Index::BITS) as Index)
}

const fn pack(indices: (Index, Index)) -> TwoIndices {
    (indices.0 as TwoIndices) | ((indices.1 as TwoIndices) << Index::BITS)
}

///
/// A range `begin..end` that can be updated atomically.
/// Used to represent initialized and empty regions within
/// the cyclic array buffer used by [`LockfreeQueue`]
/// 
struct AtomicRange<const WITHIN_LEN: usize> {
    // contains two indices, namely the begin (inclusive) and end (exclusive) of the region
    data: AtomicTwoIndices
}

#[derive(Copy, Clone)]
struct OverflowError;

impl<const LEN: usize> AtomicRange<LEN> {

    const fn new(from: Index, to: Index) -> Self {
        assert!(LEN <= Index::MAX as usize);
        assert!(from <= to);
        assert!(to <= LEN as Index);
        Self {
            data: AtomicTwoIndices::new(pack((from, to)))
        }
    }

    fn is_empty(range_data: TwoIndices) -> bool {
        unpack(range_data).0 == unpack(range_data).1
    }

    fn is_full(range_data: TwoIndices) -> bool {
        unpack(range_data).0 + LEN as Index == unpack(range_data).1
    }

    fn has_exactly_one_entry(range_data: TwoIndices) -> bool {
        unpack(range_data).0 + 1 == unpack(range_data).1
    }

    fn has_exactly_one_free(range_data: TwoIndices) -> bool {
        unpack(range_data).0 + LEN as Index - 1 == unpack(range_data).1
    }

    fn dequeue_unchecked(range_data: TwoIndices) -> TwoIndices {
        debug_assert!(!Self::is_empty(range_data));
        debug_assert!(!Self::has_exactly_one_entry(range_data));
        pack((unpack(range_data).0.checked_add(1).unwrap(), unpack(range_data).1))
    }

    fn enqueue_unchecked(range_data: TwoIndices) -> Result<TwoIndices, OverflowError> {
        debug_assert!(!Self::is_full(range_data));
        debug_assert!(!Self::has_exactly_one_free(range_data));
        Ok(pack((unpack(range_data).0, unpack(range_data).1.checked_add(1).ok_or(OverflowError)?)))
    }

    #[allow(unused)]
    fn begin(&self) -> Index {
        unpack(self.data.load(Ordering::SeqCst)).0
    }

    fn end(&self) -> Index {
        unpack(self.data.load(Ordering::SeqCst)).1
    }

    ///
    /// Atomically updates the region to include the element following its last one.
    /// 
    /// Assumes that for all states with `self.end() == expected_end`, the queue is
    /// has at least two spaces left (this is checked only in debug mode).
    /// 
    fn cond_atomic_enqueue(&self, expected_end: Index) -> Result<Result</* new end after enqueue */ Index, /* loaded end */ Index>, OverflowError> {
        let mut current = self.data.load(Ordering::SeqCst);
        if unpack(current).1 != expected_end {
            return Ok(Err(unpack(current).1));
        }
        debug_assert!(!Self::has_exactly_one_free(current));
        while let Err(new) = self.data.compare_exchange_weak(
            current, 
            Self::enqueue_unchecked(current)?, 
            Ordering::SeqCst, 
            Ordering::SeqCst
        ) {
            current = new;
            if unpack(current).1 != expected_end {
                return Ok(Err(unpack(current).1));
            }
            debug_assert!(!Self::has_exactly_one_free(current));
        }
        return Ok(Ok(unpack(current).1));
    }

    ///
    /// Atomically updates the region to not include its first element.
    /// Returns `Err` if the region has only one element left.
    /// 
    fn atomic_dequeue(&self) -> Result<Index, ()> {
        let mut current = self.data.load(Ordering::SeqCst);
        if Self::has_exactly_one_entry(current) {
            return Err(());
        }
        while let Err(new) = self.data.compare_exchange_weak(
            current, 
            Self::dequeue_unchecked(current), 
            Ordering::SeqCst, 
            Ordering::SeqCst
        ) {
            current = new;
            if Self::has_exactly_one_entry(current) {
                return Err(());
            }
        }
        return Ok(unpack(current).0);
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

#[derive(Debug, PartialEq, Eq)]
pub enum EnqueueError<T> {
    Full(T),
    IndexOverflow
}

///
/// A lock-free, multi-consumer, multi-producer, fixed size queue that always contains
/// at least one element.
/// 
/// # External Contract
/// 
/// ## Rule of Thumb
/// 
/// This queue behaves like an atomic queue, except for the fact that added elements might not immediately
/// be eligible to be removed again (i.e. `try_dequeue()` might fail). However, they will eventually be eligible 
/// to be removed.
/// 
/// Situations where this actually matters should be quite rare.
/// 
/// ## Details
/// 
/// `queue.try_enqueue(x)` atomically checks whether the queue has at least two free slots,
/// and if yes, adds `x` to the beginning of the queue. This operation will immediately be
/// visible to all `try_enqueue()` operations (so the queue will never contain more than `LEN - 1`
/// elements), but only eventually to all `try_dequeue()` operations.
/// 
/// Similarly, `queue.try_dequeue()` atomically checks whether the queue has at least two elements,
/// if if yes, removes and returns the last element of the queue. Again, this operation will immediately
/// be visible to all `try_dequeue()` operations (so will never return the same element twice), but only
/// eventuall to all `try_enqueue()` operations.
/// 
/// When we say an event happens "eventually", it means it must happen some time before all atomic operations
/// that were started before the current operation finish.
/// 
/// Consequences:
///  - The queue will never "loose" or "duplicate" elements, i.e. each element passed to a successful call
///    to `enqueue()` will be returned exactly once by a call to `dequeue()` or by `destroy()`.
///  - The queue will preserve the order of elements added, i.e. if `try_enqueue(a)` was called before
///    `try_enqueue(b)`, then it cannot happen that a call to `try_dequeue()` returns `b` and happens before 
///    a a call to `try_dequeue()` return `a`.
///  - The queue will never have less than one element; technically, this means that, if we call `try_dequeue()`
///    and there have been more (or same number) of successful `try_dequeue()` *before* this call than the number
///    of successful `try_enqueue()` either *before* or *concurrent* to this call, then `try_dequeue()` will fail.
///  - The queue can never have more than `LEN - 1` elements; the precise statement is the analogue of the last point
/// 
/// ## Example of a situation where the queue does not behave perfectly atomic
/// 
/// Assume the following:
/// ```text
/// Thread 0:             status_1 = try_enqueue(value);    status_2 = try_dequeue();
/// Threads 1, ..., N:    try_enqueue(value)
/// ```
/// Then it can happen that both `status_1` and `status_2` are failure.
/// In a truly sequentially consistent system, this can never happen.
/// 
/// # Implementation details
///
/// Each entry of `data` has an [`OccupationStatus`]
///  - [`OccupationStatus::Empty`] means it may be filled by any thread
///  - [`OccupationStatus::PendingOccupation`] means one thread is currently writing to this entry
///  - [`OccupationStatus::Occupied`] means it may be emptied by any thread
///  - [`OccupationStatus::PendingEmpty`] means some thread is currently taking the element (i.e. reading from the entry)
/// 
/// Entries within `initialized_region` must always be `Occupied`.
/// Entries within `empty_region` must always be `Empty`.
/// Eventually (i.e. after all atomic operations are complete), `initialized_region` and `empty_region` must fill the whole space `0..LEN`.
///  
pub struct LockfreeNonemptyQueue<T, const LEN: usize = 8> {
    data: [(AtomicOccupationStatus, UnsafeCell<MaybeUninit<T>>); LEN],
    initialized_region: AtomicRange<LEN>,
    empty_region: AtomicRange<LEN>,
}

unsafe impl<T: Send> Send for LockfreeNonemptyQueue<T> {}
unsafe impl<T: Send> Sync for LockfreeNonemptyQueue<T> {}

impl<T, const LEN: usize> LockfreeNonemptyQueue<T, LEN> {

    fn data_at(&self, location: Index) -> &(AtomicOccupationStatus, UnsafeCell<MaybeUninit<T>>) {
        &self.data[location as usize % LEN]
    }

    pub fn new(initial_el: T) -> Self {
        let mut result = Self { 
            data: std::array::from_fn(|_| (AtomicOccupationStatus::new(OccupationStatus::Empty), UnsafeCell::new(MaybeUninit::uninit()))), 
            initialized_region: AtomicRange::new(0, 1), 
            empty_region: AtomicRange::new(1, LEN as Index) 
        };
        result.data[0] = (AtomicOccupationStatus::new(OccupationStatus::Occupied), UnsafeCell::new(MaybeUninit::new(initial_el)));
        return result;
    }

    fn drain(&mut self) -> [Option<T>; LEN] {
        // here we do not worry about synchronization, as we have `&mut self`, thus have exclusive access
        let begin = self.initialized_region.begin();
        let result = std::array::from_fn(|i| {
            let take_location = ((i + begin as usize) % LEN) as u32;
            match self.data[take_location as usize].0.load() {
                OccupationStatus::PendingEmpty | OccupationStatus::PendingOccupation => panic!("internal error"),
                OccupationStatus::Occupied => unsafe {
                    // shouldn't be necessary since `self` gets dropped afterwards, but to be paranoid...
                    self.data_at(take_location).0.swap(OccupationStatus::Empty);
                    Some(std::ptr::read(self.data_at(take_location).1.get()).assume_init())
                },
                OccupationStatus::Empty => None
            }
        });
        let first_none_index = result.iter().enumerate().filter(|(_, x)| x.is_none()).map(|(i, _)| i).next().unwrap_or(LEN);
        debug_assert!(result[..first_none_index].iter().all(|x| x.is_some()));
        debug_assert!(result[first_none_index..].iter().all(|x| x.is_none()));
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

        let result = unsafe { std::ptr::read(self.data_at(take_location).1.get()).assume_init() };
        assert!(OccupationStatus::PendingEmpty == self.data_at(take_location).0.swap(OccupationStatus::Empty));

        let mut current_end = self.empty_region.end();

        while self.data_at(current_end).0.load() == OccupationStatus::Empty {
            match self.empty_region.cond_atomic_enqueue(current_end).ok().unwrap() {
                Ok(new_end) => current_end = new_end,
                Err(new_end) => current_end = new_end
            }
        }
        return Ok(result);
    }

    pub fn try_enqueue(&self, x: T) -> Result<(), EnqueueError<T>> {
        let mut put_location = self.empty_region.atomic_dequeue();
        if put_location.is_err() {
            return Err(EnqueueError::Full(x));
        }
        while let Err(_) = self.data_at(put_location.unwrap()).0.compare_exchange(OccupationStatus::Empty, OccupationStatus::PendingOccupation) {
            // same edge case situation as in `try_take()`
            put_location = self.empty_region.atomic_dequeue();
            if put_location.is_err() {
                return Err(EnqueueError::Full(x));
            }
        }
        unsafe { std::ptr::write(self.data_at(put_location.unwrap()).1.get(), MaybeUninit::new(x)) };
        assert!(OccupationStatus::PendingOccupation == self.data_at(put_location.unwrap()).0.swap(OccupationStatus::Occupied));

        let mut current_end = self.initialized_region.end();

        while self.data_at(current_end).0.load() == OccupationStatus::Occupied {
            match self.initialized_region.cond_atomic_enqueue(current_end) {
                Ok(Ok(new_end)) => current_end = new_end,
                Ok(Err(new_end)) => current_end = new_end,
                Err(OverflowError) => return Err(EnqueueError::IndexOverflow)
            }
        }
        return Ok(());
    }

    #[cfg(test)]
    fn check_state_sync(&self) {
        let (init_begin, init_end) = unpack(self.initialized_region.data.load(Ordering::SeqCst));
        let (empty_begin, empty_end) = unpack(self.empty_region.data.load(Ordering::SeqCst));
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
        println!("Initialized: {}..{}", self.initialized_region.begin(), self.initialized_region.end());
        println!("Empty: {}..{}", self.empty_region.begin(), self.empty_region.end());
        print!("[");
        for i in 0..LEN {
            print!("{:?}, ", self.data[i].0.load());
        }
        println!("]");
    }
}

impl<T, const LEN: usize> Drop for LockfreeNonemptyQueue<T, LEN> {

    fn drop(&mut self) {
        // this will drop the remaining `T`s
        self.drain();
    }
}

#[cfg(test)]
use std::collections::HashSet;

#[test]
fn test_queue_short() {
    let buffer: LockfreeNonemptyQueue<i32, 3> = LockfreeNonemptyQueue::new(0);
    buffer.try_enqueue(1).unwrap();
    assert_eq!(Err(EnqueueError::Full(2)), buffer.try_enqueue(2));
    assert_eq!(Ok(0), buffer.try_dequeue());
    assert_eq!(Err(()), buffer.try_dequeue());
}

#[test]
fn test_queue_basic() {
    let buffer: LockfreeNonemptyQueue<i32, 9> = LockfreeNonemptyQueue::new(-1);
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
    assert_eq!(Err(EnqueueError::Full(8)), buffer.try_enqueue(8));
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
fn test_queue_drain() {
    let mut buffer: LockfreeNonemptyQueue<i32, 9> = LockfreeNonemptyQueue::new(-1);
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
    assert_eq!(Err(EnqueueError::Full(8)), buffer.try_enqueue(8));
    assert_eq!(Ok(0), buffer.try_dequeue());
    buffer.try_enqueue(8).unwrap();
    assert_eq!(Ok(1), buffer.try_dequeue());
    assert_eq!(Ok(2), buffer.try_dequeue());
    assert_eq!([Some(3), Some(4), Some(5), Some(6), Some(7), Some(8), None, None, None], buffer.drain());
}

#[test]
fn test_queue_concurrent_state() {
    // this state is the result of a thread starting to call `try_dequeue()` on a one-element
    // buffer, but being interrupted before setting the element to `PendingEmpty`
    let buffer = LockfreeNonemptyQueue {
        data: [
            (AtomicOccupationStatus::new(OccupationStatus::Occupied), UnsafeCell::new(MaybeUninit::new(0))),
            (AtomicOccupationStatus::new(OccupationStatus::Occupied), UnsafeCell::new(MaybeUninit::new(1))),
            (AtomicOccupationStatus::new(OccupationStatus::Empty), UnsafeCell::new(MaybeUninit::uninit())),
            (AtomicOccupationStatus::new(OccupationStatus::Empty), UnsafeCell::new(MaybeUninit::uninit())),
        ],
        empty_region: AtomicRange { data: AtomicU64::new(pack((2, 4))) },
        initialized_region: AtomicRange { data: AtomicU64::new(pack((1, 2))) },
    };
    // this gives `Err()`, as (depending on first thread) it might remove the last element
    assert_eq!(Err(()), buffer.try_dequeue());
    assert_eq!(Ok(()), buffer.try_enqueue(2));
    assert_eq!(Ok(1), buffer.try_dequeue());
    // at this point, there is nothing we can do until the first thread finished `try_dequeue()`
    assert_eq!(Err(()), buffer.try_dequeue());
    assert_eq!(Err(EnqueueError::Full(3)), buffer.try_enqueue(3));
    // since the beginning state assumes that there is still another thread not finished with calling `try_dequeue()`,
    // we cannot safely call the destructor here, since we conceptually do not have mutable access to `buffer`
    std::mem::forget(buffer);
}

#[test]
fn test_queue_sync() {
    const THREADS: usize = 64;
    const ELS_PER_THREAD: usize = 8096;

    let buffer: &'static LockfreeNonemptyQueue<i32> = Box::leak(Box::new(LockfreeNonemptyQueue::new(-1)));
    let counter: &'static AtomicI32 = Box::leak(Box::new(AtomicI32::new(0)));

    let mut join_handles = Vec::new();
    for _ in 0..THREADS {
        join_handles.push(std::thread::spawn(|| {
            let mut dequeued = HashSet::new();
            let mut failed_enqueued = HashSet::new();
            for _ in 0..ELS_PER_THREAD {
                match buffer.try_enqueue(counter.fetch_add(1, Ordering::Relaxed)) {
                    Ok(()) => {},
                    Err(EnqueueError::Full(x)) => { assert!(failed_enqueued.insert(x)); },
                    Err(EnqueueError::IndexOverflow) => { panic!(); }
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
            Err(_) => {
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
    for i in -1..((THREADS * ELS_PER_THREAD) as i32 - 1) {
        assert!(all_els.contains(&(i as i32)));
    }
    assert_eq!(THREADS * ELS_PER_THREAD, all_els.len());
}

#[test]
fn test_queue_preserve_order() {
    const THREADS: usize = 64;
    const ELS_PER_THREAD: usize = 8096;

    let buffer: &'static LockfreeNonemptyQueue<i32> = Box::leak(Box::new(LockfreeNonemptyQueue::new(-1)));

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
            Err(_) => {
                failed = true;
            }
        }
    }
    assert!(!failed);
    buffer.print();
    buffer.check_state_sync();
}