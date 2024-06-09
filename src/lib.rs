#![feature(allocator_api)]
#![feature(ptr_alignment_type)]
#![feature(btree_cursors)]

use std::alloc::*;
use std::ptr::NonNull;

pub mod lockfree;

pub struct FixedSizeMempool<A: Allocator = Global> {
    base_alloc: A,
    layout: Layout
}

unsafe impl<A: Allocator> Allocator for FixedSizeMempool<A> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unimplemented!()
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        
    }
}
