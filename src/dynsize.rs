use std::alloc::*;
use std::collections::BTreeMap;
use std::cell::{RefCell, RefMut};
use std::ptr::{Alignment, NonNull};

use thread_local::ThreadLocal;

///
/// An allocator that recycles allocations to improve performance when many temporary
/// allocations are made.
/// 
/// If all allocations are of the same size, prefer using [`FixedLayoutMempool`].
/// 
pub struct DynLayoutMempool<A: Allocator + Send + Clone = Global> {
    base_alloc: A,
    alignment: Alignment,
    mempools: ThreadLocal<RefCell<BTreeMap<usize, Vec<SendableNonNull<[u8]>, A>, A>>>
}


impl DynLayoutMempool<Global> {

    pub const fn new_global(max_alignment: Alignment) -> Self {
        Self {
            base_alloc: Global,
            alignment: max_alignment,
            mempools: ThreadLocal::new()
        }
    }
}
impl<A: Allocator + Send + Clone + Default> Default for DynLayoutMempool<A> {

    fn default() -> Self {
        Self::new_in(Alignment::of::<u64>(), A::default())
    }
}
impl<A: Allocator + Send + Clone + Default> DynLayoutMempool<A> {

    pub fn new(max_alignment: Alignment) -> Self {
        Self {
            base_alloc: A::default(),
            alignment: max_alignment,
            mempools: ThreadLocal::new()
        }
    }
}

impl<A: Allocator + Send + Clone> DynLayoutMempool<A> {

    ///
    /// Creates a new [`DynLayoutMempool`] for allocation that require the given maximum
    /// alignment.
    /// 
    pub const fn new_in(max_alignment: Alignment, base_alloc: A) -> Self {
        Self {
            base_alloc: base_alloc,
            alignment: max_alignment,
            mempools: ThreadLocal::new()
        }
    }

    fn layout_to_allocate(&self, layout: Layout) -> Layout {
        assert!(layout.align() <= self.alignment.as_usize());
        layout.align_to(self.alignment.as_usize()).unwrap()
    }

    fn get_or_create_allocator_for(&self, size: usize) -> RefMut<Vec<SendableNonNull<[u8]>, A>> {
        RefMut::map(
            self.mempools.get_or(|| RefCell::new(BTreeMap::new_in(self.base_alloc.clone()))).borrow_mut(), 
            |mempools| mempools.entry(size).or_insert_with(|| Vec::new_in(self.base_alloc.clone()))
        )
    }
}

unsafe impl<A: Allocator + Send + Clone> Allocator for DynLayoutMempool<A> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.align() > self.alignment.as_usize() {
            if !cfg!(feature = "disable_print_warnings") {
                eprintln!("Called `DynLayoutMempool::allocate()` for layout {:?}, but only alignment up to {} is supported; It will return AllocError", layout, self.alignment.as_usize());
            }
            return Err(AllocError);
        }
        let layout_to_allocate = self.layout_to_allocate(layout);
        match self.get_or_create_allocator_for(layout_to_allocate.size()).pop() {
            Some(result) => Ok(result.extract()),
            None => self.base_alloc.allocate(layout_to_allocate)
        }
    }

    unsafe fn deallocate(&self, payload_ptr: NonNull<u8>, layout: Layout) {
        let allocated_layout = self.layout_to_allocate(layout);
        let ptr = NonNull::slice_from_raw_parts(payload_ptr, allocated_layout.size());
        let sendable_ptr = SendableNonNull::new(ptr);
        self.get_or_create_allocator_for(allocated_layout.size()).push(sendable_ptr);
    }
}

#[cfg(test)]
use std::ptr;

use crate::SendableNonNull;

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
        let allocator: DynLayoutMempool = DynLayoutMempool::new_global(Alignment::of::<T>());

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