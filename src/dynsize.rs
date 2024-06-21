use std::alloc::*;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::ptr::{Alignment, NonNull};
use std::cmp::max;

use crate::delayedlock::{MappedReadDelayedWriteLockReadLockGuard, ReadDelayedWriteLock, SendableNonNull};
use crate::fixedsize::FixedLayoutMempool;

const LAYOUT_HEADER: Layout = Layout::for_value(&Header { size: 0 });

struct Header {
    size: usize
}

///
/// An allocator that recycles allocations to improve performance when many temporary
/// allocations are made.
/// 
/// If all allocations are of the same size, prefer using [`FixedLayoutMempool`].
/// 
pub struct DynLayoutMempool<A: 'static + Allocator + Clone + Send + Sync = Global, const CACHE_SIZE: usize = 8> {
    base_alloc: A,
    alignment: Alignment,
    max_different_sizes: usize,
    mempools: ReadDelayedWriteLock<BTreeMap<usize, FixedLayoutMempool<A, CACHE_SIZE>, A>>
}

unsafe impl<A: 'static + Allocator + Clone + Send + Sync, const CACHE_SIZE: usize> Send for DynLayoutMempool<A, CACHE_SIZE> {}

unsafe impl<A: 'static + Allocator + Clone + Send + Sync, const CACHE_SIZE: usize> Sync for DynLayoutMempool<A, CACHE_SIZE> {}

impl<A: 'static + Allocator + Clone + Send + Sync + Default, const CACHE_SIZE: usize> Default for DynLayoutMempool<A, CACHE_SIZE> {

    fn default() -> Self {
        Self::new(Alignment::of::<u64>())
    }
}

impl<const CACHE_SIZE: usize> DynLayoutMempool<Global, CACHE_SIZE> {

    ///
    /// Creates a new [`DynLayoutMempool`] for allocation that require the given maximum
    /// alignment.
    /// 
    pub fn new_global(max_alignment: Alignment) -> Self {
        Self {
            base_alloc: Global,
            max_different_sizes: usize::MAX,
            // can't use max due to `const fn`
            alignment: if max_alignment.as_usize() > Alignment::of::<usize>().as_usize() { max_alignment } else { Alignment::of::<usize>() },
            mempools: ReadDelayedWriteLock::new(BTreeMap::new_in(Global))
        }
    }
}

impl<A: 'static + Allocator + Clone + Send + Sync, const CACHE_SIZE: usize> DynLayoutMempool<A, CACHE_SIZE> {

    ///
    /// Creates a new [`DynLayoutMempool`] for allocation that require the given maximum
    /// alignment.
    /// 
    pub fn new_in(max_alignment: Alignment, base_alloc: A) -> Self {
        Self {
            base_alloc: base_alloc.clone(),
            max_different_sizes: usize::MAX,
            alignment: max(max_alignment, Alignment::of::<usize>()),
            mempools: ReadDelayedWriteLock::new(BTreeMap::new_in(base_alloc))
        }
    }

    ///
    /// Hints to the mempool that allocations of the given layout will be common, and it should
    /// create a suitable memory pool for these allocations.
    /// 
    /// In particular, this will prevent it from over-allocating if allocations of this size are not
    /// available, but larger ones are.
    /// 
    pub fn hint_common_allocation(&self, layout: Layout) {
        let alloc_layout = self.layout_to_allocate(layout).0;
        println!("{}", alloc_layout.size());
        if let Some(_) = self.try_get_allocator_for_exact(alloc_layout.size()) {
            // entry already exists
        } else {
            if self.mempools.read().len() >= self.max_different_sizes {
                panic!("`DynLayoutMempool` was configured to only allocate data with {} different size, but this was exceeded; It will return AllocError", self.max_different_sizes);
            }
            let allocator = self.base_alloc.clone();
            self.mempools.query_write(move |mempool_tree| {
                let entry = mempool_tree.entry(alloc_layout.size());
                entry.or_insert(FixedLayoutMempool::new_in(alloc_layout, allocator));
            });
        }
    }

    fn layout_to_allocate(&self, layout: Layout) -> (Layout, usize) {
        debug_assert!(layout.align() <= self.alignment.as_usize());
        debug_assert!(self.alignment.as_usize() >= LAYOUT_HEADER.align());
        let (result, offset) = LAYOUT_HEADER.extend(layout.align_to(self.alignment.as_usize()).unwrap()).unwrap();
        debug_assert_eq!(offset, self.alignment.as_usize());
        return (result, offset);
    }

    fn try_get_allocator_for_at_least<'a>(&'a self, size: usize) -> Option<MappedReadDelayedWriteLockReadLockGuard<'a, BTreeMap<usize, FixedLayoutMempool<A, CACHE_SIZE>, A>, FixedLayoutMempool<A, CACHE_SIZE>>> {
        self.mempools.read().try_map(|locked| {
            if let Some(value) = locked.lower_bound(Bound::Included(&size)).next() {
                Ok(value.1)
            } else {
                Err(())
            }
        }).ok()
    }

    fn try_get_allocator_for_exact<'a>(&'a self, size: usize) -> Option<MappedReadDelayedWriteLockReadLockGuard<'a, BTreeMap<usize, FixedLayoutMempool<A, CACHE_SIZE>, A>, FixedLayoutMempool<A, CACHE_SIZE>>> {
        self.mempools.read().try_map(|locked| {
            if let Some(value) = locked.get(&size) {
                Ok(value)
            } else {
                Err(())
            }
        }).ok()
    }
}

impl<A: 'static + Allocator + Clone + Send + Sync + Default, const CACHE_SIZE: usize> DynLayoutMempool<A, CACHE_SIZE> {

    pub fn new(max_alignment: Alignment) -> Self {
        Self::new_in(max_alignment, A::default())
    }
}

unsafe impl<A: 'static + Allocator + Clone + Send + Sync, const CACHE_SIZE: usize> Allocator for DynLayoutMempool<A, CACHE_SIZE> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.align() > self.alignment.as_usize() {
            if !cfg!(feature = "disable_print_warnings") {
                eprintln!("Called `DynLayoutMempool` supporting max alignment {} for allocation with min alignment {:?}; It will return AllocError", self.alignment.as_usize(), layout.align());
            }
            return Err(AllocError);
        }
        let (alloc_layout, offset) = self.layout_to_allocate(layout);

        let result = if let Some(allocator) = self.try_get_allocator_for_at_least(alloc_layout.size()) {
            let result = allocator.allocate(allocator.supported_layout()).unwrap();
            debug_assert_eq!(result.len(), allocator.supported_layout().size());
            result
        } else {
            if self.mempools.read().len() >= self.max_different_sizes {
                if !cfg!(feature = "disable_print_warnings") {
                    eprintln!("`DynLayoutMempool` was configured to only allocate data with {} different size, but this was exceeded; It will return AllocError", self.max_different_sizes);
                }
            }
            self.base_alloc.allocate(alloc_layout).unwrap()
        };
        unsafe { std::ptr::write(result.as_mut_ptr() as *mut Header, Header { size: result.len() }); }

        // absolutely necessary during deallocate
        assert_eq!(offset, self.alignment.as_usize());
        let payload_ptr = unsafe { NonNull::slice_from_raw_parts(result.cast().offset(offset as isize), alloc_layout.size() - offset) };
        return Ok(payload_ptr);
    }

    unsafe fn deallocate(&self, payload_ptr: NonNull<u8>, _layout: Layout) {
        let header = std::ptr::read(payload_ptr.as_ptr().offset(-(self.alignment.as_usize() as isize)) as *const Header);
        // TODO: investigate why `layout.align` is wrong when we get called by Vec
        // debug_assert_eq!(layout, Layout::from_size_align(header.size - self.alignment.as_usize(), self.alignment.as_usize()).unwrap());
        let actual_layout = Layout::from_size_align(header.size , self.alignment.as_usize()).unwrap();
        let ptr: NonNull<u8> = unsafe { payload_ptr.cast().offset(-(self.alignment.as_usize() as isize)) };

        if let Some(allocator) = self.try_get_allocator_for_exact(header.size) {
            allocator.deallocate(ptr, actual_layout);
        } else {
            let allocator = self.base_alloc.clone();
            let sendable_ptr = SendableNonNull::new(ptr);
            self.mempools.query_write(move |mempool_tree| {
                let entry = mempool_tree.entry(header.size);
                entry.or_insert(FixedLayoutMempool::new_in(actual_layout, allocator)).deallocate(sendable_ptr.extract(), actual_layout);
            });
        }
    }
}

#[cfg(test)]
use std::ptr;

#[test]
fn test_dyn_layout_mempool_allocate() {
    let allocator: DynLayoutMempool = DynLayoutMempool::default();

    let mut allocation1 = Vec::with_capacity_in(10, &allocator);
    allocation1.extend((0..10).map(|n| n as u64));
    assert!((allocation1.as_ptr() as usize) % align_of::<u64>() == 0);
    let ptr1 = allocation1.as_ptr();
    drop(allocation1);

    let mut allocation2 = Vec::with_capacity_in(10, &allocator);
    allocation2.extend((5..15).map(|n| n as u64));
    let ptr2 = allocation2.as_ptr();
    assert!(ptr::eq(ptr1, ptr2));
    drop(allocation2);

    let mut allocation3 = Vec::with_capacity_in(20, &allocator);
    allocation3.extend((0..10).map(|n| n as u64));
    let ptr3 = allocation3.as_ptr();
    assert!(!ptr::eq(ptr1, ptr3));
    drop(allocation3);

    let mut allocation4 = Vec::with_capacity_in(10, &allocator);
    allocation4.extend((5..15).map(|n| n as u64));
    let ptr4 = allocation4.as_ptr();
    assert!(ptr::eq(ptr1, ptr4));
}

#[test]
fn test_dyn_layout_mempool_various_sizes() {

    fn test<T: Sized>() {
        let allocator: DynLayoutMempool = DynLayoutMempool::new(Alignment::of::<T>());

        let allocation1: Vec<T, _> = Vec::with_capacity_in(10, &allocator);
        assert!((allocation1.as_ptr() as usize) % align_of::<T>() == 0);
        let ptr1 = allocation1.as_ptr();
        drop(allocation1);
    
        let allocation2: Vec<T, _> = Vec::with_capacity_in(10, &allocator);
        assert!((allocation2.as_ptr() as usize) % align_of::<T>() == 0);
        let ptr2 = allocation2.as_ptr();
        assert!(ptr::eq(ptr1, ptr2));
        drop(allocation2);
    
        let allocation3: Vec<T, _> = Vec::with_capacity_in(20, &allocator);
        assert!((allocation3.as_ptr() as usize) % align_of::<T>() == 0);
        let ptr3 = allocation3.as_ptr();
        assert!(!ptr::eq(ptr1, ptr3));
        drop(allocation3);
    
        let allocation4: Vec<T, _> = Vec::with_capacity_in(10, &allocator);
        assert!((allocation4.as_ptr() as usize) % align_of::<T>() == 0);
        let ptr4 = allocation4.as_ptr();
        assert!(ptr::eq(ptr1, ptr4));
    }

    #[repr(align(64))]
    pub struct Dummy {
        _test: u8
    }

    test::<i8>();
    test::<i16>();
    test::<i32>();
    test::<i64>();
    test::<i128>();
    test::<[i128; 2]>();
    test::<Dummy>();
}