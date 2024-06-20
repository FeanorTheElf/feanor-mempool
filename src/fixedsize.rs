use std::alloc::*;
use std::ptr::NonNull;

use crate::lockfree::{EnqueueError, LockfreeQueue};

///
/// An allocator that manages allocations of a fixed [`Layout`], and caches a limited
/// number of allocations to improve performance.
/// 
pub struct FixedLayoutMempool<A: Allocator = Global, const CACHE_SIZE: usize = 8> {
    base_alloc: A,
    layout: Layout,
    cached_allocs: LockfreeQueue<NonNull<[u8]>, CACHE_SIZE>
}

unsafe impl<A: Send + Allocator, const CACHE_SIZE: usize> Send for FixedLayoutMempool<A, CACHE_SIZE> {}

unsafe impl<A: Sync + Allocator, const CACHE_SIZE: usize> Sync for FixedLayoutMempool<A, CACHE_SIZE> {}

impl<A: Allocator, const CACHE_SIZE: usize> FixedLayoutMempool<A, CACHE_SIZE> {

    ///
    /// Creates a new [`FixedLayoutMempool`] that supports allocations with the given layout.
    /// 
    pub fn new_in(layout: Layout, base_alloc: A) -> Self {
        Self {
            base_alloc: base_alloc,
            layout: layout,
            cached_allocs: LockfreeQueue::new()
        }
    }

    ///
    /// Creates a new [`FixedLayoutMempool`] that supports allocations of dynamically sized 
    /// arrays `[T]` of length `slice_len`.
    /// 
    pub fn new_for_slice_in<T: Sized>(slice_len: usize, base_alloc: A) -> Self {
        Self::new_in(Layout::array::<T>(slice_len).unwrap(), base_alloc)
    }

    ///
    /// Returns the layout supported by this allocator.
    /// 
    pub fn supported_layout(&self) -> Layout {
        self.layout
    }
}

impl<A: Default + Allocator, const CACHE_SIZE: usize> FixedLayoutMempool<A, CACHE_SIZE> {

    ///
    /// Creates a new [`FixedLayoutMempool`] that supports allocations with the given layout.
    /// 
    pub fn new(layout: Layout) -> Self {
        Self::new_in(layout, A::default())
    }

    ///
    /// Creates a new [`FixedLayoutMempool`] that supports allocations of dynamically sized 
    /// arrays `[T]` of length `slice_len`.
    /// 
    pub fn new_for_slice<T: Sized>(slice_len: usize) -> Self {
        Self::new_in(Layout::array::<T>(slice_len).unwrap(), A::default())
    }
}

unsafe impl<A: Allocator, const CACHE_SIZE: usize> Allocator for FixedLayoutMempool<A, CACHE_SIZE> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout != self.layout {
            return Err(AllocError);
        }
        match self.cached_allocs.try_dequeue() {
            Ok(alloc) => Ok(alloc),
            Err(()) => self.base_alloc.allocate(self.layout)
        }
    }

    unsafe fn deallocate(&self, mut ptr: NonNull<u8>, layout: Layout) {
        debug_assert!(layout == self.layout);
        // this statement currently causes UB according to miri (it only occurs once this slice is actually
        // used again through a reference and not a raw pointer). There currently seems to be no consensus
        // whether this actually should be UB, and the Rust memory model is still not defined. However,
        // it is not flagged if we use miri with `miri-tree-borrows`;
        // for details, see https://github.com/rust-lang/unsafe-code-guidelines/issues/256
        let ptr_as_slice = NonNull::new(std::ptr::slice_from_raw_parts_mut(ptr.as_mut(), self.layout.size())).unwrap();
        match self.cached_allocs.try_enqueue(ptr_as_slice) {
            Ok(()) => {},
            Err(EnqueueError::Full(_)) => {
                self.base_alloc.deallocate(ptr, self.layout);
            },
            Err(EnqueueError::IndexOverflow) => {
                self.base_alloc.deallocate(ptr, self.layout);
                eprintln!("Underlying queue of FixedSizeMempool has exhausted its index space; FixedSizeMempool will stop caching memory now");
            }
        }
    }

    unsafe fn grow(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // not supported, since we have a fixed layout
        Err(AllocError)
    }

    unsafe fn shrink(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // not supported, since we have a fixed layout
        Err(AllocError)
    }
}

impl<A: Allocator, const CACHE_SIZE: usize> Drop for FixedLayoutMempool<A, CACHE_SIZE> {

    fn drop(&mut self) {
        for mem in self.cached_allocs.drain() {
            if let Some(mem) = mem {
                unsafe {
                    self.base_alloc.deallocate(mem.as_non_null_ptr(), self.layout);
                }
            }
        }
    }
}

#[cfg(test)]
use std::ptr;

#[test]
fn test_fixedlayoutallocator() {
    let allocor: FixedLayoutMempool = FixedLayoutMempool::new_for_slice::<u128>(10);

    let mut allocation1 = Vec::with_capacity_in(10, &allocor);
    allocation1.extend((0..10).map(|n| n as u128));
    assert!((allocation1.as_ptr() as usize) % align_of::<u128>() == 0);
    let ptr1 = allocation1.as_ptr();
    drop(allocation1);

    let mut allocation2 = Vec::with_capacity_in(10, &allocor);
    allocation2.extend((5..15).map(|n| n as u128));
    assert!(ptr::eq(ptr1, allocation2.as_ptr()));

    let as_boxed_slice = allocation2.into_boxed_slice();
    assert!(ptr::eq(ptr1, as_boxed_slice.as_ptr()));
}