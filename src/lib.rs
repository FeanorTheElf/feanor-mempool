#![feature(allocator_api)]
#![feature(ptr_alignment_type)]
#![feature(btree_cursors)]
#![feature(slice_ptr_get)]
#![feature(layout_for_ptr)]
#![feature(btreemap_alloc)]
#![feature(const_alloc_layout)]
#![feature(mapped_lock_guards)]
#![feature(never_type)]
#![feature(test)]

extern crate test;

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

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

struct SendableNonNull<T: ?Sized> {
    data: NonNull<T>
}

impl<T: ?Sized> SendableNonNull<T> {

    unsafe fn new(data: NonNull<T>) -> Self {
        Self { data }
    }

    fn extract(self) -> NonNull<T> {
        self.data
    }
}

unsafe impl<T: ?Sized> Send for SendableNonNull<T> {}

#[cfg(test)]
use dynsize::DynLayoutMempool;
#[cfg(test)]
use std::ptr::Alignment;

///
/// I try to mimic the allocation behavior when performing operations in 
/// a ring extension. In particular, at the time of writing, the feanor-math
/// implementation required three differently sized allocations:
///  - arrays of size `n` for ring elements
///  - arrays of size `2n` temporarily during multiplication
///  - arrays of size `n^2` for the matrix during complex operations, in particular division
/// 
#[cfg(test)]
fn mock_use_case_algebra_computations<A: Allocator>(allocator: &A) {

    #[inline(never)]
    fn mimic_convolution<A: Allocator>(lhs: &[i64], rhs: &[i64], dst: &mut Vec<i64, A>) {
        dst.clear();
        for i in 0..lhs.len() {
            std::hint::black_box(lhs[i]);
        }
        for i in 0..rhs.len() {
            std::hint::black_box(rhs[i]);
        }
        for _ in 0..(lhs.len() + rhs.len() - 1) {
            dst.push(std::hint::black_box(0));
        }
        dst.push(0);
    }

    #[inline(never)]
    fn mimic_addition<A: Allocator>(lhs: &[i64], rhs: &[i64], dst: &mut Vec<i64, A>) {
        dst.clear();
        assert_eq!(lhs.len(), rhs.len());
        for i in 0..lhs.len() {
            std::hint::black_box(lhs[i]);
            std::hint::black_box(rhs[i]);
            dst.push(std::hint::black_box(0));
        }
    }

    #[inline(never)]
    fn mimic_matrix_construction<A: Allocator>(x: &[i64], y: &[i64], dst: &mut Vec<i64, A>) {
        for i in 0..x.len() {
            for j in 0..y.len() {
                std::hint::black_box(x[i]);
                std::hint::black_box(y[j]);
                dst.push(std::hint::black_box(0));
            }
        }
    }

    const SIZE: usize = 64;
    const SIZE2: usize = 2 * SIZE;
    const SIZE_SQR: usize = SIZE * SIZE;
    
    let mut x = Vec::with_capacity_in(SIZE, allocator);
    x.extend((0..SIZE).map(|n| n as i64));
    let mut y = Vec::with_capacity_in(SIZE, allocator);
    y.extend((0..SIZE).map(|n| 2 * n as i64 + 1));
    // mimic some additions and multiplications, like they would e.g. appear during polynomial
    // evaluation via Horner's schema
    for _ in 0..8 {
        let mut w = Vec::with_capacity_in(SIZE2, allocator);
        mimic_convolution(&x, &y, &mut w);
        let mut z = Vec::with_capacity_in(SIZE, allocator);
        mimic_addition(&w[SIZE..], &w[0..SIZE], &mut z);
        mimic_addition(&z, &x, &mut y);
        std::hint::black_box(w);
        std::hint::black_box(z);
    }
    let mut matrix = Vec::with_capacity_in(SIZE_SQR, allocator);
    mimic_matrix_construction(&x, &y, &mut matrix);
}

#[cfg(test)]
fn benchmark_dynsize_multithreaded<A: Allocator + Sync>(allocator: &A) {

    const THREADS: usize = 16;
    const LOOPS: usize = 16;

    std::thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|| {
                for _ in 0..LOOPS {
                    mock_use_case_algebra_computations(&allocator)
                }
            });
        }
    });
}

#[bench]
fn bench_dynsize_multithreaded_mempool(bencher: &mut test::Bencher) {
    let allocator: DynLayoutMempool = DynLayoutMempool::new_global(Alignment::of::<u64>());
    bencher.iter(|| {
        benchmark_dynsize_multithreaded(&allocator);
    });
}

#[bench]
fn bench_dynsize_multithreaded_global(bencher: &mut test::Bencher) {
    let allocator = Global;
    bencher.iter(|| {
        benchmark_dynsize_multithreaded(&allocator);
    });
}

#[cfg(test)]
fn benchmark_dynsize_singlethreaded<A: Allocator>(allocator: &A) {

    const LOOPS: usize = 256;

    for _ in 0..LOOPS {
        mock_use_case_algebra_computations(&allocator)
    }
}

#[bench]
fn bench_dynsize_singlethreaded_mempool(bencher: &mut test::Bencher) {
    let allocator: DynLayoutMempool = DynLayoutMempool::new_global(Alignment::of::<u64>());
    bencher.iter(|| {
        benchmark_dynsize_singlethreaded(&allocator);
    });
}

#[bench]
fn bench_dynsize_singlethreaded_global(bencher: &mut test::Bencher) {
    let allocator = Global;
    bencher.iter(|| {
        benchmark_dynsize_singlethreaded(&allocator);
    });
}