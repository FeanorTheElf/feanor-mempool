use std::cell::{Cell, RefCell};
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::ptr;

use crate::{MemProviderRc, MemoryProvider, MemoryProviderObject};

///
/// A memory object that will return the owned memory to its creator
/// [`SizedMemoryPool`] after going out of scope.
/// 
pub struct RecyclingMemoryObject<T> {
    data: Box<[MaybeUninit<T>]>,
    recycle_to: Rc<SizedMemoryPool<T>>
}

impl<T> MemoryProviderObject<T> for RecyclingMemoryObject<T> {

    unsafe fn drop_or_recycle(self: Box<Self>) {
        let mut data: Box<[MaybeUninit<T>]> = std::mem::transmute(self.data);
        for i in 0..data.len() {
            data[i].assume_init_drop();
        }
        self.recycle_to.recycle(data);
    }

    unsafe fn deref_initialized(&self) -> &[T] {
        MaybeUninit::slice_assume_init_ref(&*self.data)
    }

    unsafe fn deref_mut_initialized(&mut self) -> &mut [T] {
        MaybeUninit::slice_assume_init_mut(&mut*self.data)
    }

    fn deref_uninit(&mut self) -> &mut [MaybeUninit<T>] {
        &mut*self.data
    }
}

///
/// A memory pool that manages homogeneous memory slices of a fixed size.
/// 
pub struct SizedMemoryPool<T> {
    memories: RefCell<Vec<Box<[MaybeUninit<T>]>>>,
    max_stored: usize,
    mem_size: Cell<usize>
}

impl<T> SizedMemoryPool<T> {

    ///
    /// Creates a new [`SizedMemoryPool`] that caches at most `max_stored` allocations,
    /// each having size `mem_size`.
    /// 
    pub fn new<'a>(max_stored: usize, mem_size: usize) -> MemProviderRc<'a, T>
        where T: 'a
    {
        assert!(mem_size != usize::MAX);
        MemProviderRc { ptr: Rc::new(Self {
            memories: RefCell::new(Vec::new()),
            max_stored: max_stored,
            mem_size: Cell::new(mem_size)
        }) }
    }

    ///
    /// Creates a new [`SizedMemoryPool`] that caches at most `max_stored` allocations.
    /// Note that a [`SizedMemoryPool`] only allows allocations of a fixed size, however
    /// when creating the pool using this function, this size is chosen to be the size 
    /// passed to the first call of [`MemoryProvider::new()`].
    /// 
    pub fn new_dynamic_size<'a>(max_stored: usize) -> MemProviderRc<'a, T>
        where T: 'a
    {
        MemProviderRc { ptr: Rc::new(Self {
            memories: RefCell::new(Vec::new()),
            max_stored: max_stored,
            mem_size: Cell::new(usize::MAX)
        }) }
    }

    fn recycle(&self, data: Box<[MaybeUninit<T>]>) {
        let mut memories = self.memories.borrow_mut();
        if memories.len() < self.max_stored {
            memories.push(data);
        }
    }
}

impl<'a, T> MemoryProvider<'a, T> for SizedMemoryPool<T>
    where T: 'a
{
    unsafe fn new(&self, self_rc: &Rc<dyn 'a + MemoryProvider<T>>, size: usize) -> Box<dyn 'a + MemoryProviderObject<T>> {
        if self.mem_size.get() == usize::MAX {
            self.mem_size.set(size);
        } else {
            assert_eq!(self.mem_size.get(), size, "SizedMemoryPool can only provide elements of one fixed size");
        }
        let mut memories = self.memories.borrow_mut();
        let memory = if let Some(data) = memories.pop() {
            data
        } else {
            Box::new_uninit_slice(size)
        };
        // this is already guaranteed by the method contract, but no real harm in checking again
        assert!(ptr::addr_eq(self as *const SizedMemoryPool<T>, self_rc.as_ref() as *const dyn MemoryProvider<T>));
        let self_rc_ptr = Rc::into_raw(self_rc.clone()) as *const SizedMemoryPool<T>;
        let self_rc = Rc::from_raw(self_rc_ptr);
        Box::new(RecyclingMemoryObject {
            data: memory,
            recycle_to: self_rc
        })
    }
}

#[cfg(test)]
use crate::TraceDrop;
#[cfg(test)]
use std::collections::HashSet;

#[test]
fn test_memory_pool_correct_len() {
    let pool = SizedMemoryPool::new_dynamic_size(2);
    let mem1 = pool.new_init(6, |_| 0);
    let mem2 = pool.new_init(6, |_| 0);
    let mem3 = pool.new_init(6, |_| 0);
    assert_eq!(6, mem1.len());
    drop(mem1);
    let mem4 = pool.new_init(6, |_| 0);

    assert_eq!(6, mem2.len());
    assert_eq!(6, mem3.len());
    assert_eq!(6, mem4.len());
}

#[test]
#[should_panic]
fn test_memory_pool_only_one_len() {
    let pool = SizedMemoryPool::new_dynamic_size(2);
    pool.new_init(5, |_| 0);
    pool.new_init(6, |_| 0);
}

#[test]
fn test_memory_pool_reuse() {
    let pool = SizedMemoryPool::new(2, 6);
    
    let fst = pool.new_init(6, |_| 0);
    let fst_ptr = &*fst as *const [i32];
    let snd = pool.new_init(6, |_| 0);
    let snd_ptr = &*snd as *const [i32];
    let thd = pool.new_init(6, |_| 0);
    let thd_ptr = &*thd as *const [i32];
    drop(fst);
    drop(snd);
    drop(thd);

    // create a new allocation, hoping that it will fill the space of `thd_ptr`
    let _helper_alloc = Box::<[MaybeUninit<i32>]>::new_uninit_slice(6);

    let new_fst = pool.new_init(6, |_| 0);
    let new_snd = pool.new_init(6, |_| 0);
    let new_thd = pool.new_init(6, |_| 0);

    assert!(ptr::eq(&*new_fst as *const [i32], fst_ptr) || ptr::eq(&*new_fst as *const [i32], snd_ptr));
    assert!(ptr::eq(&*new_snd as *const [i32], fst_ptr) || ptr::eq(&*new_snd as *const [i32], snd_ptr));
    assert!(!ptr::addr_eq(&*new_thd as *const [i32], thd_ptr));
}

#[test]
fn test_memory_pool_drop() {
    let pool = SizedMemoryPool::new(2, 12);

    let drop_tracer = Rc::new(RefCell::new(HashSet::new()));

    let a = pool.new_init(12, |i| TraceDrop { content: i as i32, drop_tracer: drop_tracer.clone() });
    drop(a);
    
    assert_eq!(&(0..12).collect::<HashSet<_>>(), &*drop_tracer.borrow());
}