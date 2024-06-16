use std::alloc::*;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::ptr::{Alignment, NonNull};
use std::sync::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use std::cmp::max;

use crate::fixedsize::FixedLayoutMempool;

const LAYOUT_USIZE: Layout = Layout::for_value(&1usize);

///
/// An allocator that recycles allocations to improve performance when many temporary
/// allocations are made.
/// 
/// If all allocations are of the same size, prefer using [`FixedLayoutMempool`].
/// 
pub struct DynLayoutMempool<A: Allocator + Clone = Global, const CACHE_SIZE: usize = 8> {
    base_alloc: A,
    alignment: Alignment,
    mempools: RwLock<BTreeMap<usize, FixedLayoutMempool<A, CACHE_SIZE>, A>>
}

impl<A: Allocator + Clone + Default, const CACHE_SIZE: usize> Default for DynLayoutMempool<A, CACHE_SIZE> {

    fn default() -> Self {
        Self::new(Alignment::of::<u64>())
    }
}

impl<const CACHE_SIZE: usize> DynLayoutMempool<Global, CACHE_SIZE> {

    ///
    /// Creates a new [`DynLayoutMempool`] for allocation that require the given maximum
    /// alignment.
    /// 
    pub const fn new_global(max_alignment: Alignment) -> Self {
        Self {
            base_alloc: Global,
            // can't use max due to `const fn`
            alignment: if max_alignment.as_usize() > Alignment::of::<usize>().as_usize() { max_alignment } else { Alignment::of::<usize>() },
            mempools: RwLock::new(BTreeMap::new_in(Global))
        }
    }
}

impl<A: Allocator + Clone, const CACHE_SIZE: usize> DynLayoutMempool<A, CACHE_SIZE> {

    ///
    /// Creates a new [`DynLayoutMempool`] for allocation that require the given maximum
    /// alignment.
    /// 
    pub fn new_in(max_alignment: Alignment, base_alloc: A) -> Self {
        Self {
            base_alloc: base_alloc.clone(),
            alignment: max(max_alignment, Alignment::of::<usize>()),
            mempools: RwLock::new(BTreeMap::new_in(base_alloc))
        }
    }

    fn layout_to_allocate(&self, layout: Layout) -> (Layout, usize) {
        debug_assert!(layout.align() <= self.alignment.as_usize());
        debug_assert!(self.alignment.as_usize() >= LAYOUT_USIZE.align());
        let (result, offset) = LAYOUT_USIZE.extend(layout.align_to(self.alignment.as_usize()).unwrap()).unwrap();
        debug_assert_eq!(offset, self.alignment.as_usize());
        return (result, offset);
    }

    fn get_or_create_allocator_for(&self, layout: Layout) -> MappedRwLockReadGuard<FixedLayoutMempool<A, CACHE_SIZE>> {
        // check whether the entry exists
        if let Ok(result) = RwLockReadGuard::try_map(self.mempools.read().unwrap(), |locked| {
            if let Some(value) = locked.lower_bound(Bound::Included(&layout.size())).next() {
                Some(value.1)
            } else {
                None
            }
        }) {
            return result;
        }
        // if not add it
        {
            let mut locked = self.mempools.write().unwrap();
            if !locked.contains_key(&layout.size()) {
                let old_entry = locked.insert(layout.size(), FixedLayoutMempool::new_in(layout, self.base_alloc.clone()));
                debug_assert!(old_entry.is_none());
            }
        }
        RwLockReadGuard::map(self.mempools.read().unwrap(), |locked| {
            locked.lower_bound(Bound::Included(&layout.size())).next().unwrap().1
        })
    }

    fn get_allocator_for(&self, size: usize) -> MappedRwLockReadGuard<FixedLayoutMempool<A, CACHE_SIZE>> {
        RwLockReadGuard::map(self.mempools.read().unwrap(), |locked| {
            locked.get(&size).unwrap()
        })
    }
}

impl<A: Default + Allocator + Clone, const CACHE_SIZE: usize> DynLayoutMempool<A, CACHE_SIZE> {

    pub fn new(max_alignment: Alignment) -> Self {
        Self::new_in(max_alignment, A::default())
    }
}

unsafe impl<A: Allocator + Clone, const CACHE_SIZE: usize> Allocator for DynLayoutMempool<A, CACHE_SIZE> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.align() > self.alignment.as_usize() {
            return Err(AllocError);
        }
        let (alloc_layout, offset) = self.layout_to_allocate(layout);

        let allocator = self.get_or_create_allocator_for(alloc_layout);
        let result = allocator.allocate(alloc_layout).unwrap();
        debug_assert_eq!(result.len(), allocator.supported_layout().size());
        unsafe { std::ptr::write(result.as_mut_ptr() as *mut usize, result.len()); }

        // absolutely necessary during deallocate
        assert_eq!(offset, self.alignment.as_usize());
        let payload_ptr = unsafe { NonNull::slice_from_raw_parts(result.cast().offset(offset as isize), alloc_layout.size() - offset) };
        return Ok(payload_ptr);
    }

    unsafe fn deallocate(&self, payload_ptr: NonNull<u8>, _layout: Layout) {
        let size = std::ptr::read(payload_ptr.as_ptr().offset(-(self.alignment.as_usize() as isize)) as *const usize);
        let ptr: NonNull<[u8]> = unsafe { NonNull::slice_from_raw_parts(payload_ptr.cast().offset(-(self.alignment.as_usize() as isize)), size) };
        let allocator = self.get_allocator_for(size);
        allocator.deallocate(ptr.cast(), allocator.supported_layout());
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