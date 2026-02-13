use std::ops::{Add, Sub, AddAssign, SubAssign};
use std::any::TypeId;
use std::ops::Range;
use shm_tracy::*;
use shm_tracy::zone_scoped;
use crate::runtime::*;
use crate::shimlibs::*;

#[derive(Debug)]
pub struct Config {
    // There are max 2^24 addressable values, each 8 bytes large
    // This value can be up to 2^27-1.
    pub(crate) memory_space_bytes: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            memory_space_bytes: MAX_U24 * 8,
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Hash, Eq, PartialOrd, Ord, Copy, Clone, Debug, PartialEq)]
#[repr(packed)]
pub struct u24(pub(crate) [u8; 3]);
pub(crate) const MAX_U24: u32 = 0xFFFFFF;

impl From<Word> for usize {
    fn from(val: Word) -> Self {
        val.0.into()
    }
}

impl From<usize> for u24 {
    fn from(val: usize) -> Self {
        (val as u32).into()
    }
}

impl From<i32> for u24 {
    fn from(val: i32) -> Self {
        (val as u32).into()
    }
}

impl From<u32> for u24 {
    fn from(val: u32) -> Self {
        let b = val.to_be_bytes();
        u24([b[1], b[2], b[3]])
    }
}

impl From<u24> for u32 {
    fn from(val: u24) -> u32 {
        u32::from_be_bytes([0, val.0[0], val.0[1], val.0[2]])
    }
}

impl From<u24> for usize {
    fn from(val: u24) -> usize {
        u32::from(val) as usize
    }
}

impl From<u24> for u64 {
    fn from(val: u24) -> u64 {
        u32::from(val) as u64
    }
}

/**
 * The interpreter stores memory in 8-byte words. Each `Word` is
 * an index into the interpreter memory.
 */
#[cfg_attr(feature = "facet", derive(Facet))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Word(pub u24);

impl Add<u8> for Word {
    type Output = Word;

    fn add(self, rhs: u8) -> Word {
        self + rhs as u32
    }
}

impl Add<i32> for Word {
    type Output = Word;

    fn add(self, rhs: i32) -> Word {
        let val = (u32::from(self.0) as i32 + rhs) as u32;
        Word(val.into())
    }
}

impl Add<u32> for Word {
    type Output = Word;

    fn add(self, rhs: u32) -> Word {
        Word((u32::from(self.0) + rhs).into())
    }
}

impl Sub<u32> for Word {
    type Output = Word;

    fn sub(self, rhs: u32) -> Word {
        Word((u32::from(self.0) - rhs).into())
    }
}

impl Add<Word> for Word {
    type Output = Word;

    fn add(self, rhs: Word) -> Word {
        Word((u32::from(self.0) + u32::from(rhs.0)).into())
    }
}

impl Sub<Word> for Word {
    type Output = Word;

    fn sub(self, rhs: Word) -> Word {
        Word((u32::from(self.0) - u32::from(rhs.0)).into())
    }
}

impl AddAssign<u32> for Word {
    fn add_assign(&mut self, rhs: u32) {
        self.0 = (u32::from(self.0) + rhs).into()
    }
}

impl SubAssign<u32> for Word {
    fn sub_assign(&mut self, rhs: u32) {
        self.0 = (u32::from(self.0) - rhs).into()
    }
}

impl AddAssign<Word> for Word {
    fn add_assign(&mut self, rhs: Word) {
        self.0 = (u32::from(self.0) + u32::from(rhs.0)).into()
    }
}

impl SubAssign<Word> for Word {
    fn sub_assign(&mut self, rhs: Word) {
        self.0 = (u32::from(self.0) - u32::from(rhs.0)).into()
    }
}

impl From<usize> for Word {
    fn from(val: usize) -> Word {
        Word(val.into())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FreeBlock {
    #[cfg(feature = "dev")]
    pub pos: Word,
    #[cfg(feature = "dev")]
    pub size: Word,

    #[cfg(not(feature = "dev"))]
    pos: Word,
    #[cfg(not(feature = "dev"))]
    size: Word,
}

impl FreeBlock {
    fn new(pos: Word, size: Word) -> Self {
        Self { pos, size }
    }

    pub fn end(&self) -> Word {
        self.pos + self.size
    }
}

#[cfg_attr(feature = "facet", derive(Facet))]
pub struct MMU {
    // This is the raw memory managed by the MMU
    #[cfg(feature = "dev")]
    pub mem: Vec<u64>,
    #[cfg(not(feature = "dev"))]
    mem: Vec<u64>,

    // This is a list of chunks of free memory
    // The first value is the position in words
    // The second value is the number of words
    // Sorted for sanity's sake, though I'm not
    // sure if necessary?
    pub(crate) free_list: Vec<FreeBlock>,
    // We don't store metadata about any allocations
    // It's up to the caller to know how much memory
    // should be freed.
}

macro_rules! alloc {
    ($mmu:expr, $count:expr, $msg:expr) => {
        {
            #[cfg(debug_assertions)]
            {
                //$mmu.alloc_debug($count, $msg)
                $mmu.alloc_no_debug($count)
            }

            #[cfg(not(debug_assertions))]
            {
                $mmu.alloc_no_debug($count)
            }
        }
    };
}

impl MMU {
    fn eprint_free_list(&self) {
        eprintln!("Free list:");
        for block in self.free_list.iter() {
            eprintln!("    {block:?}");
        }
    }

    pub(crate) fn with_capacity(word_count: Word) -> Self {
        let mem = vec![0; usize::from(word_count.0)];
        // Start the free list at word 1, reserving word 0 as a sentinel.
        // This ensures no allocation ever returns position 0, which is used
        // as a "null" / "no scope" sentinel by consumers.
        let free_list = vec![FreeBlock::new(Word(1.into()), word_count - Word(1.into()))];
        Self {
            mem: mem,
            free_list: free_list,
        }
    }

    /*
    fn compact_free_list() {
        todo!("compact_free_list not implemented");
    }
    */

    pub(crate) unsafe fn get<T: 'static>(&self, word: Word) -> &T {
        if TypeId::of::<T>() == TypeId::of::<Word>() {
            panic!("Can't MMU::get<Word>");
        }

        unsafe {
            let ptr: *const T = std::mem::transmute(&self.mem[usize::from(word.0)]);
            &*ptr
        }
    }

    pub(crate) unsafe fn get_mut<T>(&mut self, word: Word) -> &mut T {
        unsafe {
            let ptr: *mut T = std::mem::transmute(&mut self.mem[usize::from(word.0)]);
            &mut *ptr
        }
    }

    pub(crate) fn alloc_and_set<T>(&mut self, value: T, _debug_name: &str) -> Word {
        let word_count = Word((std::mem::size_of::<T>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, _debug_name);
        unsafe {
            let ptr: *mut T = std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(value);
        }
        position
    }

    pub(crate) fn alloc_str_raw(&mut self, contents: &[u8]) -> Word {
        let total_len = contents.len().div_ceil(8);
        let word_count = Word(total_len.into());
        let position = alloc!(self, word_count, &format!("str `{}`", debug_u8s(contents)));

        let bytes: &mut [u8] = unsafe {
            let u64_slice = &mut self.mem[
                usize::from(position.0)..
                (usize::from(position.0)+total_len)
            ];
            std::slice::from_raw_parts_mut(
                u64_slice.as_mut_ptr() as *mut u8,
                contents.len(),
            )
        };

        for (idx, b) in contents.iter().enumerate() {
            bytes[idx] = *b;
        }

        position
    }

    fn alloc_debug(&mut self, words: Word, msg: &str) -> Word {
        let result = self.alloc_no_debug(words);
        eprintln!("Alloc {} {}: {}", usize::from(words.0), msg, usize::from(result));
        result
    }

    pub(crate) fn alloc_no_debug(&mut self, words: Word) -> Word {
        if u32::from(words.0) == 0u32 {
            return Word(0.into());
        }
        for idx in 0..self.free_list.len() {
            if self.free_list[idx].size >= words {
                let returned_pos: Word = self.free_list[idx].pos;

                if self.free_list[idx].size == words {
                    self.free_list.remove(idx);
                } else {
                    self.free_list[idx].pos += words;
                    self.free_list[idx].size -= words;
                }

                // Compaction is handled when it's convenient.
                // Some people might tend towards using a linked list to have
                // constant time insert/deletion without needing a separate
                // compaction step, but I'm guessing that iterating through
                // linear memory is going to be pretty fast.
                //
                // Another option is to allocate from the end of the Vec so
                // that we can at least pop off chunks as they're depleted.
                //
                // Or we could keep track of how many empty elements there are
                // in `free_list` so that we can skip them until the next compaction.
                //
                // There are further enhancements if we split things into buckets,
                // but we can keep things simple for now.

                return returned_pos;
            }
        }
        panic!(
            "Could not allocate {:?} words from free list {:#?} (total: {})",
            words, self.free_list, self.mem.len()
        );
    }

    /**
     * Returns the position in `self.mem` of the block allocated
     */
    fn alloc(&mut self, size: Word) -> Word {
        self.alloc_debug(size, "Unspecified alloc")
    }

    pub(crate) fn free(&mut self, pos: Word, size: Word) {
        if u32::from(size.0) == 0 || u32::from(size.0) == 0 {
            return;
        }

        // eprintln!("Free {}: {}", usize::from(size.0), usize::from(pos));

        // This is the idx of the first free block containing addresses greater than the
        // position we need to free
        let idx = {
            let mut ret = None;
            for idx in 0..self.free_list.len() {
                if pos < self.free_list[idx].end() {
                    ret = Some(idx);
                    break;
                }
            }
                // Technically we could get here if there was no free block at the end
                // of the memory, but we basically don't expect that to happen, so it's
                // not worth addressing.
            ret.expect("Could not find free list position to insert free mem")
        };

        // The data we're freeing is in one of the four categories:
        //   1. needs to be joined to the end of the previous idx
        //   2. joins the previous idx and this idx
        //   3. sits between the previous idx and this idx
        //   4. needs to be joined to the start of this idx
        if idx != 0 {
            if pos == self.free_list[idx-1].end() {
                // Case 1 or 2
                // Since the position matches the end of the previous
                // block we need to join with it
                if pos + size < self.free_list[idx].pos {
                    // Case 1
                    // It's not long enough to reach the idx block, just
                    // add the sizes
                    self.free_list[idx-1].size += size;
                    return;
                } else if pos + size == self.free_list[idx].pos {
                    // Case 2
                    self.free_list[idx-1].size = self.free_list[idx].end() - self.free_list[idx-1].pos;
                    self.free_list.remove(idx);
                    return;
                } else {
                    panic!("Mis-sized free does not fit in gap!");
                }
            }
        }
        if pos + size < self.free_list[idx].pos {
            // Case 3
            self.free_list.insert(idx, FreeBlock::new(pos, size));
            return;
        } else if pos + size == self.free_list[idx].pos {
            // Case 4
            self.free_list[idx].pos = pos;
            self.free_list[idx].size += size;
            return;
        } else {
            panic!("Mis-sized free overlaps with idx block!");
        }
    }
}

// MMU methods that depend on runtime types (ShimValue, ShimDict, ShimList, ShimFn, etc.)
impl MMU {
    pub(crate) fn alloc_str(&mut self, contents: &[u8]) -> ShimValue {
        assert!(contents.len() <= u16::MAX as usize, "String length exceeds u16::MAX");
        let pos = self.alloc_str_raw(contents);
        ShimValue::String(contents.len() as u16, 0, pos.0)
    }

    pub(crate) fn alloc_dict_raw(&mut self) -> Word {
        let word_count = Word((std::mem::size_of::<ShimDict>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, "Dict");
        unsafe {
            let ptr: *mut ShimDict =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(ShimDict::new());
        }
        position
    }

    pub(crate) fn alloc_dict(&mut self) -> ShimValue {
        ShimValue::Dict(self.alloc_dict_raw())
    }

    pub(crate) fn alloc_list(&mut self) -> ShimValue {
        let word_count = Word((std::mem::size_of::<ShimList>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, "List");
        unsafe {
            let ptr: *mut ShimList =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(ShimList::new());
        }
        ShimValue::List(position)
    }

    pub(crate) fn alloc_fn(&mut self, pc: u32, name: &[u8], captured_scope: u32) -> ShimValue {
        let word_count = Word((std::mem::size_of::<ShimFn>() as u32).div_ceil(8).into());
        let position = alloc!(self, word_count, &format!("Fn `{}`", debug_u8s(name)));

        // Allocate the name string
        let name_pos = self.alloc_str_raw(name);

        unsafe {
            let ptr: *mut ShimFn =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(ShimFn { pc, name_len: name.len() as u16, name: name_pos, captured_scope });
        }
        ShimValue::Fn(position)
    }

    pub(crate) fn alloc_native<T: ShimNative>(&mut self, val: T) -> ShimValue {
        assert!(std::mem::size_of::<Box<dyn ShimNative>>() == 16);
        let word_count = Word(2.into());
        let position = alloc!(self, word_count, "Native");
        unsafe {
            let ptr: *mut Box<dyn ShimNative> =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            ptr.write(Box::new(val));
        }
        ShimValue::Native(position)
    }

    pub(crate) fn alloc_bound_native_fn(&mut self, obj: &ShimValue, func: NativeFn) -> ShimValue {
        let position = alloc!(self, Word(2.into()), "Bound Native Fn");
        unsafe {
            let obj_ptr: *mut ShimValue =
                std::mem::transmute(&mut self.mem[usize::from(position.0)]);
            obj_ptr.write(*obj);
            let fn_ptr: *mut NativeFn = std::mem::transmute(
                &mut self.mem[usize::from(position.0) + 1],
            );
            fn_ptr.write(func);

            ShimValue::BoundNativeMethod(position)
        }
    }
}

#[derive(Debug)]
pub struct Bitmask {
    data: Vec<u64>,
}

impl Bitmask {
    pub fn new(num_bits: usize) -> Self {
        // Round up if we don't have a number of bits that's cleanly divisible by 64
        let blocks = num_bits.div_ceil(64);

        Bitmask {
            data: vec![0; blocks],
        }
    }

    pub fn set(&mut self, index: usize) {
        let (block_idx, bit_offset) = self.pos(index);
        self.data[block_idx] |= 1 << bit_offset;
    }

    pub fn is_set(&self, index: usize) -> bool {
        let (block_idx, bit_offset) = self.pos(index);
        (self.data[block_idx] & (1 << bit_offset)) != 0
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    pub fn find_zeros(&self) -> Vec<Range<usize>> {
        let _zone = zone_scoped!("find_zeros");
        let mut ranges = Vec::new();
        let mut start_of_run: Option<usize> = None;

        for (idx, word) in self.data.iter().enumerate() {
            if *word == 0 {
                if start_of_run.is_none() {
                    start_of_run = Some(idx*64);
                }
            } else if *word == u64::MAX {
                if let Some(start_bit) = start_of_run {
                    ranges.push(start_bit..(idx*64));
                    start_of_run = None;
                }
            } else {
                let bit_offset: usize;
                match start_of_run {
                    Some(start_bit) => {
                        bit_offset = word.trailing_zeros() as usize;
                        ranges.push(start_bit..(idx*64 + bit_offset));
                        start_of_run = None
                    }
                    None => {
                        bit_offset = word.trailing_ones() as usize;
                        start_of_run = Some(idx*64 + bit_offset);
                    }
                }
                let mut shifted_word = word >> bit_offset;
                for i in (bit_offset as usize)..64 {
                    let is_zero = (shifted_word & 1) == 0;

                    if is_zero {
                        if start_of_run == None {
                            start_of_run = Some(i);
                        }
                    } else {
                        if let Some(start) = start_of_run {
                            ranges.push(start..i);
                            start_of_run = None;
                        }
                    }
                    shifted_word >>= 1;
                }
            }
        }


        if let Some(start_bit) = start_of_run {
            ranges.push(start_bit..self.data.len()*64);
        }

        ranges
    }

    fn pos(&self, index: usize) -> (usize, usize) {
        (index / 64, index % 64)
    }
}

pub(crate) struct GC<'a> {
    mem: &'a mut MMU,
    mask: Bitmask,
}

impl<'a> GC<'a> {
    pub(crate) fn new(mem: &'a mut MMU) -> Self {
        let last_block_start = mem.free_list[mem.free_list.len()-1].pos;
        let mut mask = Bitmask::new(last_block_start.into());
        // Mark word 0 so the GC never frees the sentinel reserved by MMU::with_capacity
        mask.set(0);
        Self {
            mem,
            mask,
        }
    }

    pub(crate) fn mark(&mut self, mut vals: Vec<ShimValue>) {
        let _zone = zone_scoped!("GC mark");
        unsafe {
            while !vals.is_empty() {
                let _zone = zone_scoped!("GC mark item");
                match vals.pop().unwrap() {
                    ShimValue::Integer(_) | ShimValue::Float(_) | ShimValue::Bool(_) | ShimValue::Unit | ShimValue::None | ShimValue::Uninitialized => (),
                    ShimValue::Fn(fn_pos) => {
                        let pos: usize = fn_pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        // Mark the ShimFn struct (8 bytes = 1 word: u32 pc + Word name)
                        self.mask.set(pos);
                        
                        // Mark the function name string
                        let shim_fn: &ShimFn = self.mem.get(fn_pos);
                        vals.push(ShimValue::String(shim_fn.name_len, 0, shim_fn.name.0));
                    },
                    ShimValue::List(pos) => {
                        let pos: usize = pos.into();
                        let lst: &ShimList = self.mem.get(pos.into());
                        for idx in 0..lst.len() {
                            vals.push(lst.get(self.mem, idx as isize).unwrap());
                        }

                        let contents_pos = usize::from(lst.data);
                        for idx in contents_pos..(contents_pos + lst.capacity()) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::String(len, offset, pos) => {
                        let len = len as usize;
                        let offset = offset as usize;
                        let pos: usize = usize::from(pos);
                        // TODO: check this...
                        for idx in pos..(pos + (offset + len).div_ceil(8)) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::Dict(pos) => {
                        let pos: usize = pos.into();
                        let dict: &ShimDict = std::mem::transmute(&self.mem.mem[pos]);
                        let u64_slice = &self.mem.mem[
                            usize::from(dict.entries)..
                            usize::from(dict.entries)+3*(dict.entry_count as usize)
                        ];
                        let entries: &[DictEntry] = std::slice::from_raw_parts(
                            u64_slice.as_ptr() as *const DictEntry,
                            u64_slice.len() / 3,
                        );

                        // Push the keys/vals
                        let count: usize = dict.entry_count as usize;
                        for entry in &entries[..count] {
                            if !entry.key.is_uninitialized() {
                                vals.push(entry.key);
                                vals.push(entry.value);
                            }
                        }

                        // Mark the space for the dict struct
                        for idx in pos..(pos + std::mem::size_of::<ShimDict>().div_ceil(8)) {
                            self.mask.set(idx);
                        }

                        let size = 1 << dict.size_pow;

                        // Mark the indices array
                        let indices_pos: usize = dict.indices.into();
                        for idx in indices_pos..(indices_pos + size) {
                            self.mask.set(idx);
                        }

                        // Mark the entries array
                        let entries_pos: usize = dict.entries.into();
                        for idx in entries_pos..(entries_pos + size*3) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::StructDef(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let def: &StructDef = self.mem.get(pos.into());
                        for idx in pos..(pos + def.mem_size()) {
                            self.mask.set(idx);
                        }
                    },
                    ShimValue::Struct(pos) => {
                        let pos: usize = pos.into();
                        if self.mask.is_set(pos) {
                            continue;
                        }
                        let def_pos: usize = self.mem.mem[pos] as usize;
                        let def: &StructDef = self.mem.get(def_pos.into());

                        for idx in pos..(pos + def.member_count as usize + 1) {
                            self.mask.set(idx);
                            // Push the members
                            if idx != pos {
                                vals.push(
                                    ShimValue::from_u64(self.mem.mem[idx])
                                );
                            }
                        }
                        vals.push(ShimValue::StructDef(def_pos.into()));
                    },
                    ShimValue::NativeFn(pos) => {
                        let pos: usize = pos.into();
                        self.mask.set(pos);
                    },
                    ShimValue::Native(pos) => {
                        let pos: usize = pos.into();
                        assert!(std::mem::size_of::<Box<dyn ShimNative>>() == 16);
                        self.mask.set(pos);
                        self.mask.set(pos+1);

                        let ptr: &Box<dyn ShimNative> = std::mem::transmute(&self.mem.mem[pos]);

                        vals.extend(ptr.gc_vals());
                    },
                    ShimValue::BoundMethod(pos, fn_pos) => {
                        // Mark the bound struct
                        let val = ShimValue::Struct(pos);
                        vals.push(val);
                        // Mark the function
                        vals.push(ShimValue::Fn(fn_pos));
                    },
                    ShimValue::BoundNativeMethod(pos) => {
                        let pos: usize = pos.into();
                        // Native ShimValue
                        self.mask.set(pos);
                        // Pointer to the fn
                        self.mask.set(pos+1);

                        // Any values that the obj holds
                        let ptr: &Box<dyn ShimNative> = std::mem::transmute(&self.mem.mem[pos]);
                        vals.extend(ptr.gc_vals());
                        vals.push(ShimValue::Native(pos.into()));
                    },
                    ShimValue::Environment(pos) => {
                        let scope: &EnvScope = self.mem.get(pos);

                        // Chunk of memory that store the EnvScope metadata
                        let pos: usize = pos.into();
                        for bit in pos..(pos + std::mem::size_of::<EnvScope>().div_ceil(8)) {
                            self.mask.set(bit);
                        }

                        // Data block
                        let start = usize::from(scope.data);
                        let end = start + scope.capacity as usize;
                        for bit in start..end {
                            self.mask.set(bit);
                        }
                        
                        // Walk the contiguous data block and collect values
                        let bytes = scope.raw_bytes(&self.mem);
                        let mut off = 0usize;
                        while off < bytes.len() {
                            let key_len = bytes[off] as usize;
                            let value_offset = off + 1 + key_len;
                            let val: ShimValue = {
                                let mut val_bytes = [0u8; 8];
                                std::ptr::copy_nonoverlapping(bytes[value_offset..].as_ptr(), val_bytes.as_mut_ptr(), 8);
                                std::mem::transmute(val_bytes)
                            };
                            vals.push(val);
                            off = value_offset + 8;
                        }

                        if scope.parent != 0.into() {
                            vals.push(ShimValue::Environment(Word(scope.parent)));
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn sweep(&mut self) {
        let _zone = zone_scoped!("GC sweep");

        // TODO: need to add the original last block from the free list
        let last_block = self.mem.free_list[self.mem.free_list.len()-1];
        self.mem.free_list = self.mask
            .find_zeros()
            .iter()
            .map(|block| FreeBlock { pos: block.start.into(), size: (block.end-block.start).into() }).collect();
        let new_last_block = self.mem.free_list[self.mem.free_list.len()-1];
        if new_last_block.end() >= last_block.pos {
            // Merge with the new last block
            let len = self.mem.free_list.len();
            self.mem.free_list[len - 1].size = last_block.end() - self.mem.free_list[len - 1].pos;
        } else {
            // Append the previous last free block (which was not included in the bitmask)
            self.mem.free_list.push(last_block);
        }
    }
}
