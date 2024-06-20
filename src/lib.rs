#![feature(allocator_api)]
#![feature(ptr_alignment_type)]
#![feature(btree_cursors)]
#![feature(slice_ptr_get)]
#![feature(layout_for_ptr)]
#![feature(btreemap_alloc)]
#![feature(const_alloc_layout)]
#![feature(mapped_lock_guards)]
#![feature(test)]

extern crate test;

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

///
/// Implementation of a lockfree queue that is used by the allocators to recycle allocations
/// when accessed by multiple threads.
/// 
mod lockfree;
///
/// Implementation of a Read-delayed-write-lock.
/// 
mod delayedlock;
///
/// Implementation of a memory pool supporting only allocations of a fixed layout.
/// 
pub mod fixedsize;
///
/// Implementation of a memory pool supporting arbitrary allocations.
/// 
pub mod dynsize;

///
/// An [`Rc`] pointing to an [`Allocator`]. As opposed to `Rc<A>`, the type `AllocRc<A>`
/// implements again [`Allocator`].
/// 
pub struct AllocRc<A: Allocator, PtrAlloc: Allocator + Clone = Global>(pub Rc<A, PtrAlloc>);

impl<A: Allocator, PtrAlloc: Allocator + Clone> Clone for AllocRc<A, PtrAlloc> {

    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

unsafe impl<A: Allocator, PtrAlloc: Allocator + Clone> Allocator for AllocRc<A, PtrAlloc> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (*self.0).allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        (*self.0).deallocate(ptr, layout)
    }
}

///
/// An [`Arc`] pointing to an [`Allocator`]. As opposed to `Arc<A>`, the type `AllocArc<A>`
/// implements again [`Allocator`].
/// 
pub struct AllocArc<A: Allocator, PtrAlloc: Allocator + Clone = Global>(pub Arc<A, PtrAlloc>);

unsafe impl<A: Allocator, PtrAlloc: Allocator + Clone> Allocator for AllocArc<A, PtrAlloc> {

    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (*self.0).allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        (*self.0).deallocate(ptr, layout)
    }
}

impl<A: Allocator, PtrAlloc: Allocator + Clone> Clone for AllocArc<A, PtrAlloc> {
    
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(test)]
use dynsize::DynLayoutMempool;
#[cfg(test)]
use std::ptr::Alignment;

#[bench]
fn bench_multisize_concurrent_allocations(bencher: &mut test::Bencher) {
    const THREADS: usize = 32;
    const ALLOCATIONS_LOOPS: usize = 1024;
    const SIZE1: usize = 1024;
    const SIZE2: usize = 2048;
    const SIZE3: usize = 65536;

    let memory_provider: DynLayoutMempool<Global, THREADS> = DynLayoutMempool::new(Alignment::of::<u64>());
    bencher.iter(|| {
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    for _ in 0..ALLOCATIONS_LOOPS {
                        let data1: Vec<u64, _> = Vec::with_capacity_in(SIZE1, &memory_provider);
                        let data2: Vec<u64, _> = Vec::with_capacity_in(SIZE2, &memory_provider);
                        let data3: Vec<u64, _> = Vec::with_capacity_in(SIZE1, &memory_provider);
                        let data4: Vec<u64, _> = Vec::with_capacity_in(SIZE3, &memory_provider);
                        drop(data1);
                        drop(data2);
                        drop(data3);
                        drop(data4);
                    }
                });
            }
        });
    })
}