use std::collections::HashMap;

#[cfg(feature = "facet")]
use facet::Facet;

use std::ops::{Add, Sub};
use std::ops::{AddAssign, SubAssign};

pub struct Config {
    // There are max 2^24 addressable values, each 8 bytes large
    // This value can be up to 2^32.
    memory_space_bytes: u32,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            memory_space_bytes: 2u32.pow(22) // 4 MB
        }
    }
}

#[cfg_attr(feature = "facet", derive(Facet))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Word(pub u32);

impl Add<u32> for Word {
    type Output = Word;

    fn add(self, rhs: u32) -> Word {
        Word(self.0 + rhs)
    }
}

impl Sub<u32> for Word {
    type Output = Word;

    fn sub(self, rhs: u32) -> Word {
        Word(self.0 - rhs)
    }
}

impl Add<Word> for Word {
    type Output = Word;

    fn add(self, rhs: Word) -> Word {
        Word(self.0 + rhs.0)
    }
}

impl Sub<Word> for Word {
    type Output = Word;

    fn sub(self, rhs: Word) -> Word {
        Word(self.0 - rhs.0)
    }
}

impl AddAssign<u32> for Word {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

impl SubAssign<u32> for Word {
    fn sub_assign(&mut self, rhs: u32) {
        self.0 -= rhs;
    }
}

impl AddAssign<Word> for Word {
    fn add_assign(&mut self, rhs: Word) {
        self.0 += rhs.0;
    }
}

impl SubAssign<Word> for Word {
    fn sub_assign(&mut self, rhs: Word) {
        self.0 -= rhs.0;
    }
}

#[cfg_attr(feature = "facet", derive(Facet))]
#[derive(Debug)]
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
        Self {
            pos,
            size,
        }
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
    pub free_list: Vec<FreeBlock>,

    // We don't store metadata about any allocations
    // It's up to the caller to know how much memory
    // should be freed.
}

impl MMU {
    fn with_capacity(word_count: Word) -> Self {
        let mem = vec![0; word_count.0 as usize];
        let free_list = vec![FreeBlock::new(Word(0), word_count)];
        Self {
            mem: mem,
            free_list: free_list,
        }
    }

    fn compact_free_list() {
        todo!();
    }

    /**
     * Returns the position in `self.mem` of the block allocted
     */
    fn alloc(&mut self, words: Word) -> Word {
        for block in self.free_list.iter_mut() {
            if block.size >= words {
                let returned_pos: Word = block.pos;
                block.pos += words;
                block.size -= words;

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
        panic!("Could not allocate {:?} words from free list {:#?}", words, self.free_list);
    }

    fn free(&mut self, words: u32, ptr: *const u64) {
    }
}

#[derive(Facet)]
pub struct Interpreter {
    pub mem: MMU,
    pub source: HashMap<String, String>,
}

impl Interpreter {
    pub fn create(config: &Config) -> Self {
        let mmu = MMU::with_capacity(Word(config.memory_space_bytes / 8));

        Self {
            mem: mmu,
            source: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2+2, 4);
    }
}
