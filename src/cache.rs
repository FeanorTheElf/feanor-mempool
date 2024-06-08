use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::ptr;

use crate::{GrowableMemProviderRc, GrowableMemoryProvider, MemProviderRc, MemoryProvider, MemoryProviderObject};

///
/// A memory object that will return the owned memory to its creator
/// [`SizedMemoryPool`] after going out of scope.
/// 
pub struct SizedRecyclingMemoryObject<T> {
    data: Box<[MaybeUninit<T>]>,
    recycle_to: Rc<SizedMemoryPool<T>>
}

impl<T> MemoryProviderObject<T> for SizedRecyclingMemoryObject<T> {

    unsafe fn drop_or_recycle(mut self: Box<Self>) {
        for i in 0..self.data.len() {
            self.data[i].assume_init_drop();
        }
        self.recycle_to.recycle(self.data);
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

    fn get_creator_identifier(&self) -> &u8 {
        &SIZED_MEMORY_POOL_GENERIC_CREATOR_IDENTIFIER
    }
}

static SIZED_MEMORY_POOL_GENERIC_CREATOR_IDENTIFIER: u8 = 0;

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
        Box::new(SizedRecyclingMemoryObject {
            data: memory,
            recycle_to: self_rc
        })
    }
}

///
/// A memory object that will return the owned memory to its creator
/// [`UnsizedMemoryPool`] after going out of scope.
/// 
pub struct UnsizedRecyclingMemoryObject<T> {
    data: Vec<MaybeUninit<T>>,
    recycle_to: Rc<UnsizedMemoryPool<T>>
}

impl<T> MemoryProviderObject<T> for UnsizedRecyclingMemoryObject<T> {

    unsafe fn drop_or_recycle(mut self: Box<Self>) {
        for i in 0..self.data.len() {
            self.data[i].assume_init_drop();
        }
        self.recycle_to.recycle(self.data);
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

    fn get_creator_identifier(&self) -> &u8 {
        &self.recycle_to.creator_identifier
    }
}

///
/// A memory pool that manages homogeneous memory slices of dynamic size.
/// 
pub struct UnsizedMemoryPool<T> {
    memories: RefCell<BTreeMap<usize, Vec<Vec<MaybeUninit<T>>>>>,
    max_stored: usize,
    currently_stored: Cell<usize>,
    creator_identifier: u8
}

impl<T> UnsizedMemoryPool<T> {

    pub fn new<'a>(max_stored: usize) -> GrowableMemProviderRc<'a, T>
        where T: 'a
    {
        GrowableMemProviderRc { ptr: Rc::new(Self {
            memories: RefCell::new(BTreeMap::new()),
            max_stored: max_stored,
            currently_stored: Cell::new(0),
            creator_identifier: 0
        }) }
    }

    fn recycle(&self, data: Vec<MaybeUninit<T>>) {
        let mut memories = self.memories.borrow_mut();
        if self.currently_stored.get() < self.max_stored {
            self.currently_stored.set(self.currently_stored.get() + 1);
            memories.entry(data.capacity()).or_insert(Vec::new()).push(data);
        }
    }

    fn get_array_of_size(&self, size: usize) -> Option<Vec<MaybeUninit<T>>> {
        let mut memories = self.memories.borrow_mut();
        let mut pos = memories.lower_bound_mut(std::ops::Bound::Included(&size));

        if let Some((key, mems)) = pos.next() {
            debug_assert!(self.currently_stored.get() > 0);
            let mut memory = mems.pop().unwrap();
            debug_assert_eq!(*key, memory.capacity());
            let key = *key;
            if mems.is_empty() {
                memories.remove(&key).unwrap();
            }
            debug_assert!(memory.capacity() >= size);
            memory.resize_with(size, || MaybeUninit::uninit());
            self.currently_stored.set(self.currently_stored.get() - 1);
            return Some(memory);
        } else {
            return None;
        };
    }
}

impl<'a, T> MemoryProvider<'a, T> for UnsizedMemoryPool<T>
    where T: 'a
{
    unsafe fn new(&self, self_rc: &Rc<dyn 'a + MemoryProvider<T>>, size: usize) -> Box<dyn 'a + MemoryProviderObject<T>> {
        let memory = self.get_array_of_size(size).unwrap_or_else(|| Box::new_uninit_slice(size).into_vec());
        // this is already guaranteed by the method contract, but no real harm in checking again
        assert!(ptr::addr_eq(self as *const UnsizedMemoryPool<T>, self_rc.as_ref() as *const dyn MemoryProvider<T>));
        let self_rc_ptr = Rc::into_raw(self_rc.clone()) as *const UnsizedMemoryPool<T>;
        let self_rc = Rc::from_raw(self_rc_ptr);
        Box::new(UnsizedRecyclingMemoryObject {
            data: memory,
            recycle_to: self_rc
        })
    }
}

impl<'a, T> GrowableMemoryProvider<'a, T> for UnsizedMemoryPool<T>
    where T: 'a
{
    unsafe fn new(&self, self_rc: &Rc<dyn 'a + GrowableMemoryProvider<T>>, size: usize) -> Box<dyn 'a + MemoryProviderObject<T>> {
        // copy the code of `<Self as MemoryProvider<T>>::new()` - for some reason delegate gives a lifetime conflict
        let memory = self.get_array_of_size(size).unwrap_or_else(|| Box::new_uninit_slice(size).into_vec());
        // this is already guaranteed by the method contract, but no real harm in checking again
        assert!(ptr::addr_eq(self as *const UnsizedMemoryPool<T>, self_rc.as_ref() as *const dyn GrowableMemoryProvider<T>));
        let self_rc_ptr = Rc::into_raw(self_rc.clone()) as *const UnsizedMemoryPool<T>;
        let self_rc = Rc::from_raw(self_rc_ptr);
        Box::new(UnsizedRecyclingMemoryObject {
            data: memory,
            recycle_to: self_rc
        })
    }

    unsafe fn change_size(&self, data: Box<dyn 'a + MemoryProviderObject<T>>, new_size: usize) -> Box<dyn 'a + MemoryProviderObject<T>> {
        // this ensures that `data` indeed points to us
        assert!(ptr::addr_eq(data.get_creator_identifier(), &self.creator_identifier));
        // this is safe since data is associated with this memory pool
        let mut data_cast = Box::from_raw(Box::into_raw(data) as *mut (dyn 'a + MemoryProviderObject<T>) as *mut () as *mut UnsizedRecyclingMemoryObject<T>);
        if data_cast.data.capacity() >= new_size {
            data_cast.data.resize_with(new_size, || MaybeUninit::uninit());
            return data_cast;
        } else if let Some(mut memory) = self.get_array_of_size(new_size) {
            assert!(data_cast.data.len() < memory.len());
            ptr::copy_nonoverlapping(data_cast.data.as_ptr(), memory.as_mut_ptr(), data_cast.data.len());
            self.recycle(data_cast.data);
            return Box::new(UnsizedRecyclingMemoryObject {
                data: memory,
                recycle_to: data_cast.recycle_to
            });
        } else {
            data_cast.data.resize_with(new_size, || MaybeUninit::uninit());
            return data_cast;
        }
    }

    fn upcast(self: Rc<Self>) -> Rc<dyn 'a + MemoryProvider<'a, T>> {
        unsafe {
            Rc::from_raw(Rc::into_raw(self))
        }
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
    
    assert_eq!(&(0..12).collect::<HashSet<_>>(), &*drop_tracer.as_ref().borrow());
}

#[test]
fn test_unsized_memory_pool_reuse() {
    let pool = UnsizedMemoryPool::new(2);
    
    let fst = pool.new_init(6, |_| 0);
    let fst_ptr = (&*fst).as_ptr();
    let snd = pool.new_init(7, |_| 0);
    let snd_ptr = (&*snd).as_ptr();
    let thd = pool.new_init(8, |_| 0);
    let thd_ptr = (&*thd).as_ptr();
    drop(fst);
    drop(snd);
    drop(thd);

    // create a new allocation, hoping that it will fill the space of `thd_ptr`
    let _helper_alloc = Box::<[MaybeUninit<i32>]>::new_uninit_slice(6);

    let new_fst = pool.new_init(3, |_| 0);
    let new_snd = pool.new_init(4, |_| 0);
    let new_thd = pool.new_init(5, |_| 0);

    assert!(ptr::eq((&*new_fst).as_ptr(), fst_ptr) || ptr::eq((&*new_fst).as_ptr(), snd_ptr) || ptr::eq((&*new_fst).as_ptr(), thd_ptr));
    assert!(ptr::eq((&*new_snd).as_ptr(), fst_ptr) || ptr::eq((&*new_snd).as_ptr(), snd_ptr) || ptr::eq((&*new_snd).as_ptr(), thd_ptr));
    assert!(!ptr::addr_eq((&*new_thd).as_ptr(), fst_ptr) && !ptr::addr_eq((&*new_thd).as_ptr(), snd_ptr) && !ptr::addr_eq((&*new_thd).as_ptr(), thd_ptr));
}

#[test]
fn test_unsized_memory_pool_drop() {
    let pool = UnsizedMemoryPool::new(2);

    let drop_tracer = Rc::new(RefCell::new(HashSet::new()));

    let a = pool.new_init(12, |i| TraceDrop { content: i as i32, drop_tracer: drop_tracer.clone() });
    drop(a);
    
    assert_eq!(&(0..12).collect::<HashSet<_>>(), &*drop_tracer.as_ref().borrow());
}

#[test]
fn test_unsized_memory_pool_grow_reuse() {
    let pool = UnsizedMemoryPool::new(2);
    
    let a = pool.new_init(1000, |_| 1000);
    let a_ptr = (&*a).as_ptr();
    let mut b = pool.new_init(10, |i| i as i32);
    assert!(!ptr::eq((&*b).as_ptr(), a_ptr));
    drop(a);

    pool.grow_init(&mut b, 1000, |_| 0);
    assert!(ptr::eq((&*b).as_ptr(), a_ptr));
    assert_eq!(&(0..10).chain((10..1000).map(|_| 0)).collect::<Vec<_>>(), &*b);
}

#[test]
fn test_unsized_memory_pool_grow_drop() {
    let pool = UnsizedMemoryPool::new(2);
    let drop_tracer = Rc::new(RefCell::new(HashSet::new()));
    
    let a = pool.new_init(1000, |i| TraceDrop { content: -(i as i32), drop_tracer: drop_tracer.clone() });
    let mut b = pool.new_init(10, |i| TraceDrop { content: i as i32, drop_tracer: drop_tracer.clone() });
    assert_eq!(&HashSet::new(), &*drop_tracer.as_ref().borrow());
    drop(a);
    assert_eq!(&(0..1000).map(|x| -x).collect::<HashSet<_>>(), &*drop_tracer.as_ref().borrow());

    pool.grow_init(&mut b, 1000, |i| TraceDrop { content: i as i32, drop_tracer: drop_tracer.clone() });
    assert_eq!(&(0..1000).map(|x| -x).collect::<HashSet<_>>(), &*drop_tracer.as_ref().borrow());
}
