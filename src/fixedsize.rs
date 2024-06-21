use std::alloc::*;
use std::cell::{RefCell, RefMut};
use std::ptr::NonNull;

use thread_local::ThreadLocal;

use crate::SendableNonNull;

///
/// An allocator that manages allocations of a fixed [`Layout`], and caches a limited
/// number of allocations to improve performance.
/// 
pub struct FixedLayoutMempool<A: Allocator + Clone + Send = Global> {
    base_alloc: A,
    layout: Layout,
    mempool: ThreadLocal<RefCell<Vec<SendableNonNull<[u8]>, A>>>
}

impl<A: Allocator + Clone + Send> FixedLayoutMempool<A> {

    ///
    /// Creates a new [`FixedLayoutMempool`] that supports allocations with the given layout.
    /// 
    pub fn new_in(layout: Layout, base_alloc: A) -> Self {
        Self {
            base_alloc: base_alloc,
            layout: layout,
            mempool: ThreadLocal::new()
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

impl<A: Default + Allocator + Clone + Send> FixedLayoutMempool<A> {

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

    fn get_mempool(&self) -> RefMut<Vec<SendableNonNull<[u8]>, A>> {
        self.mempool.get_or(|| RefCell::new(Vec::new_in(self.base_alloc.clone()))).borrow_mut()
    }
}

unsafe impl<A: Default + Allocator + Clone + Send> Allocator for FixedLayoutMempool<A> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout != self.layout {
            if !cfg!(feature = "disable_print_warnings") {
                eprintln!("Called `FixedLayoutMempool::allocate()` for layout {:?}, but only layout {:?} is supported; It will return AllocError", layout, self.supported_layout());
            }
            return Err(AllocError);
        }
        match self.get_mempool().pop() {
            Some(result) => Ok(result.extract()),
            None => self.base_alloc.allocate(layout)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        debug_assert_eq!(layout, self.layout);
        let ptr = NonNull::slice_from_raw_parts(ptr, layout.size());
        let sendable_ptr = SendableNonNull::new(ptr);
        self.get_mempool().push(sendable_ptr);
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