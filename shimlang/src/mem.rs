use std::ops::{Add, Sub, AddAssign, SubAssign};
use std::any::TypeId;

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
     * Returns the position in `self.mem` of the block allocted
     */
    fn alloc(&mut self, size: Word) -> Word {
        self.alloc_debug(size, "Unspecified alloc")
    }

    pub(crate) fn free(&mut self, pos: Word, size: Word) {
        if u32::from(size.0) == 0 || u32::from(size.0) == 0 {
            return;
        }

        // eprintln!("Free {}: {}", usize::from(size.0), usize::from(pos));

        // This is the idx of the frst free block containing addresses greater than the
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
