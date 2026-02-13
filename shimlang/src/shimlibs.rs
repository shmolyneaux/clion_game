use crate::runtime::*;
use crate::lex::debug_u8s;
use std::any::{Any, type_name};
use shm_tracy::*;
use shm_tracy::zone_scoped;
use crate::mem::*;

pub(crate) struct ListIterator {
    pub(crate) lst: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for ListIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_list_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to ListIterator.next()"));
                }

                let itr: &mut ListIterator = args.args[0].as_native(interpreter)?;
                let lst = itr.lst.list(interpreter)?;
                if itr.idx >= lst.len() {
                    Ok(ShimValue::None)
                } else {
                    let result = lst.get(&mut interpreter.mem, itr.idx as isize)?;
                    itr.idx += 1;

                    Ok(result)
                }
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_list_iter_next))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.lst]
    }
}

pub(crate) struct DictKeysIterator {
    pub(crate) dict: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for DictKeysIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_keys_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to DictKeysIterator.next()"));
                }

                let itr: &mut DictKeysIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);
                
                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let result = entries[itr.idx].key;
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }
                
                Ok(ShimValue::None)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_keys_iter_next))
        } else if ident == b"iter" {
            fn shim_dict_keys_iter_iter(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_keys_iter_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

pub(crate) struct DictValuesIterator {
    pub(crate) dict: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for DictValuesIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_values_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to DictValuesIterator.next()"));
                }

                let itr: &mut DictValuesIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);
                
                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let result = entries[itr.idx].value;
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }
                
                Ok(ShimValue::None)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_values_iter_next))
        } else if ident == b"iter" {
            fn shim_dict_values_iter_iter(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_values_iter_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

pub(crate) struct DictEntryNative {
    pub(crate) key: ShimValue,
    pub(crate) value: ShimValue,
}
impl ShimNative for DictEntryNative {
    fn get_attr(&self, _self_as_val: &ShimValue, _interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"key" {
            Ok(self.key)
        } else if ident == b"value" {
            Ok(self.value)
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.key, self.value]
    }
}

pub(crate) struct DictItemsIterator {
    pub(crate) dict: ShimValue,
    pub(crate) idx: usize,
}
impl ShimNative for DictItemsIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_dict_items_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to DictItemsIterator.next()"));
                }

                let itr: &mut DictItemsIterator = args.args[0].as_native(interpreter)?;
                let dict = itr.dict.dict(interpreter)?;
                let entries = dict.entries_array(interpreter);
                
                // Skip invalid entries (tombstones)
                while itr.idx < entries.len() {
                    if entries[itr.idx].is_valid() {
                        let entry = &entries[itr.idx];
                        let result = interpreter.mem.alloc_native(DictEntryNative {
                            key: entry.key,
                            value: entry.value,
                        });
                        itr.idx += 1;
                        return Ok(result);
                    }
                    itr.idx += 1;
                }
                
                Ok(ShimValue::None)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_items_iter_next))
        } else if ident == b"iter" {
            fn shim_dict_items_iter_iter(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_dict_items_iter_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>() ))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.dict]
    }
}

pub(crate) struct RangeNative {
    pub(crate) start: ShimValue,
    pub(crate) end: ShimValue,
}

impl ShimNative for RangeNative {
    fn to_string(&self, interpreter: &mut Interpreter) -> String {
        format!("Range({}, {})", self.start.to_string(interpreter), self.end.to_string(interpreter))
    }

    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"step" {
            fn shim_range_step(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                let step = unpacker.required(b"step")?;
                unpacker.end()?;

                let range: &RangeNative = obj.as_native(interpreter)?;
                
                // Check for zero step
                let is_zero = match step {
                    ShimValue::Integer(0) => true,
                    ShimValue::Float(f) if f == 0.0 => true,
                    _ => false,
                };
                
                if is_zero {
                    return Err(format!("Step cannot be zero"));
                }

                let iterator = RangeIterator {
                    current: range.start,
                    end: range.end,
                    step: step,
                };
                Ok(interpreter.mem.alloc_native(iterator))
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_range_step))
        } else if ident == b"iter" {
            fn shim_range_iter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;

                let range: &RangeNative = obj.as_native(interpreter)?;
                let iterator = RangeIterator {
                    current: range.start,
                    end: range.end,
                    step: ShimValue::Integer(1),
                };
                Ok(interpreter.mem.alloc_native(iterator))
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_range_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>()))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.start, self.end]
    }
}

pub(crate) struct RangeIterator {
    pub(crate) current: ShimValue,
    pub(crate) end: ShimValue,
    pub(crate) step: ShimValue,
}

impl ShimNative for RangeIterator {
    fn get_attr(&self, self_as_val: &ShimValue, interpreter: &mut Interpreter, ident: &[u8]) -> Result<ShimValue, String> {
        if ident == b"next" {
            fn shim_range_iter_next(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                if args.args.len() != 1 {
                    return Err(format!("Can't provide positional args to RangeIterator.next()"));
                }

                let itr: &mut RangeIterator = args.args[0].as_native(interpreter)?;
                
                // Determine if we've reached the end based on step direction
                // For positive steps: iterate while current < end
                // For negative steps: iterate while current > end
                let step_is_positive = match itr.step.gt(interpreter, &ShimValue::Integer(0))? {
                    ShimValue::Bool(b) => b,
                    _ => return Err(format!("Step comparison failed")),
                };
                
                let has_more = if step_is_positive {
                    // current < end
                    match itr.current.lt(interpreter, &itr.end)? {
                        ShimValue::Bool(b) => b,
                        _ => return Err(format!("Range comparison failed")),
                    }
                } else {
                    // current > end
                    match itr.current.gt(interpreter, &itr.end)? {
                        ShimValue::Bool(b) => b,
                        _ => return Err(format!("Range comparison failed")),
                    }
                };
                
                if !has_more {
                    Ok(ShimValue::None)
                } else {
                    let result = itr.current;
                    // current = current + step
                    let mut pending_args = ArgBundle::new();
                    match itr.current.add(interpreter, &itr.step, &mut pending_args)? {
                        CallResult::ReturnValue(new_current) => {
                            itr.current = new_current;
                            Ok(result)
                        }
                        CallResult::PC(pc, captured_scope) => {
                            let mut new_env = Environment::with_scope(captured_scope);
                            let new_current = interpreter.execute_bytecode_extended(
                                &mut (pc as usize),
                                pending_args,
                                &mut new_env,
                            )?;
                            itr.current = new_current;
                            Ok(result)
                        }
                    }
                }
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_range_iter_next))
        } else if ident == b"iter" {
            fn shim_range_iterator_iter(_interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
                let mut unpacker = ArgUnpacker::new(args);
                let obj = unpacker.required(b"obj")?;
                unpacker.end()?;
                Ok(obj)
            }

            Ok(interpreter.mem.alloc_bound_native_fn(self_as_val, shim_range_iterator_iter))
        } else {
            Err(format!("Can't get_attr {} on {}", debug_u8s(ident), type_name::<Self>()))
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    fn gc_vals(&self) -> Vec<ShimValue> {
        vec![self.current, self.end, self.step]
    }
}

const fn generate_size_table() -> [u32; 256] {
    let mut table = [0; 256];

    let mut i = 0;

    while i < 256 {
        table[i] = match i {
            0 => 0,
            1 => 4,
            2 => 16,
            3 => 32,
            // Multiply 1.5 the previous
            4 => 48,
            5 => 72,
            6 => 108,
            7 => 162,
            8 => 243,
            9 => 364,
            10 => 546,
            11 => 819,
            12 => 1228,
            13 => 1842,
            // Multiply x1.2 the previous
            14 => 2210,
            15 => 2652,
            16 => 3182,
            17 => 3818,
            18 => 4581,
            19 => 5497,
            20 => 6596,
            21 => 7915,
            22 => 9498,
            23 => 11397,
            24 => 13676,
            25 => 16411,
            26 => 19693,
            27 => 23631,
            28 => 28357,
            29 => 34028,
            30 => 40833,
            31 => 48999,
            32 => 58798,
            33 => 70557,
            34 => 84668,
            35 => 101601,
            36 => 121921,
            37 => 146305,
            38 => 175566,
            39 => 210679,
            40 => 252814,
            41 => 303376,
            42 => 364051,
            43 => 436861,
            44 => 524233,
            45 => 629079,
            46 => 754894,
            47 => 905872,
            48 => 1087046,
            49 => 1304455,
            50 => 1565346,
            51 => 1878415,
            52 => 2254098,
            53 => 2704917,
            54 => 3245900,
            55 => 3895080,
            56 => 4674096,
            57 => 5608915,
            58 => 6730698,
            59 => 8076837,
            60 => 9692204,
            61 => 11630644,
            62 => 13956772,
            63 => 16748126,
            _ => MAX_U24,
        };
        i += 1;
    }
    table
}

static LIST_CAPACITY_LUT: [u32; 256] = generate_size_table();

#[derive(Debug, Clone, Copy)]
pub(crate) struct DictEntry {
    pub(crate) hash: u64,
    pub(crate) key: ShimValue,
    pub(crate) value: ShimValue,
}

impl DictEntry {
    pub(crate) fn is_valid(&self) -> bool {
        self.hash != 0
            && !self.key.is_uninitialized()
            && !self.value.is_uninitialized()
    }

    fn invalidate(&mut self) {
        self.hash = 0;
        self.key = ShimValue::Uninitialized;
        self.value = ShimValue::Uninitialized;
    }
}

// Minimum non-zero size_pow for ShimDict. When the dict grows from empty,
// it starts with this size_pow value (2^3 = 8 index slots, capacity of ~5 entries).
const MIN_NON_ZERO_SIZE_POW: u8 = 3;

#[derive(Debug)]
pub(crate) struct ShimDict {
    // Size of the index array, always a power of 2
    pub(crate) size_pow: u8,

    // These could be u24, but are u32 to keep things simple

    // Number of valid entries + tombstoned entries
    pub(crate) entry_count: u32,
    // Non-tombstoned entries
    pub(crate) used: u32,

    // Memory position of the dict data
    pub(crate) indices: u24,
    pub(crate) entries: u24,
}

enum DictSlot<'a> {
    Occupied(usize, &'a mut DictEntry),
    // If it's Unoccupied, this is the idx in the indices array
    UnoccupiedU8(u32, usize),
    UnoccupiedU16(u32, usize),
    UnoccupiedU32(u32, usize),
}

#[derive(Debug)]
enum TypedIndices {
    Zero,
    U8(&'static mut [u8]),
    U16(&'static mut [u16]),
    U32(&'static mut [u32]),
}

impl TypedIndices {
    fn get(&self, index: usize) -> usize {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => {data[index] as usize},
            Self::U16(data) => {data[index] as usize},
            Self::U32(data) => {data[index] as usize},
        }
    }

    fn set(&mut self, index: usize, value: usize) {
        match self {
            Self::Zero => panic!("Can't IndexMut empty TypedIndices"),
            Self::U8(data) => {data[index] = value as u8},
            Self::U16(data) => {data[index] = value as u16},
            Self::U32(data) => {data[index] = value as u32},
        }
    }

    fn is_unset(&self, index: usize) -> bool {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => {data[index] == u8::MAX},
            Self::U16(data) => {data[index] == u16::MAX},
            Self::U32(data) => {data[index] == u32::MAX},
        }
    }

    fn is_tombstone(&self, index: usize) -> bool {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => {data[index] == u8::MAX - 1},
            Self::U16(data) => {data[index] == u16::MAX - 1},
            Self::U32(data) => {data[index] == u32::MAX - 1},
        }
    }

    fn set_tombstone(&mut self, index: usize) {
        match self {
            Self::Zero => panic!("Can't index empty TypedIndices"),
            Self::U8(data) => {data[index] = u8::MAX - 1},
            Self::U16(data) => {data[index] = u16::MAX - 1},
            Self::U32(data) => {data[index] = u32::MAX - 1},
        }
    }
}

impl ShimDict {
    pub(crate) fn new() -> Self {
        Self {
            size_pow: 0,
            used: 0,
            entry_count: 0,
            indices: 0.into(),
            entries: 0.into(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.used as usize
    }

    pub(crate) fn get(&self, interpreter: &mut Interpreter, key: ShimValue) -> Result<ShimValue, String> {
        // Check if dict is empty
        if self.size_pow == 0 {
            return Err(format!("Key {key:?} not in dict"));
        }
        
        match self.probe(interpreter, key)? {
            DictSlot::Occupied(_, entry) => {
                Ok(entry.value)
            },
            DictSlot::UnoccupiedU8(..) => {
                Err(format!("Key {key:?} not in dict"))
            },
            _ => todo!(),
        }
    }

    fn print_entries(&self, interpreter: &Interpreter) {
        eprintln!("Entries");
        let _entries: &[DictEntry] = unsafe {
            let u64_slice = &interpreter.mem.mem[
                usize::from(self.entries)..
                usize::from(self.entries)+3*(self.entry_count as usize)
            ];
            std::slice::from_raw_parts(
                u64_slice.as_ptr() as *const DictEntry,
                u64_slice.len() / 3,
            )
        };
    }

    fn expand_capacity(&mut self, interpreter: &mut Interpreter) {
        let _zone = zone_scoped!("ShimDict::expand_capacity");
        let old_size = self.index_size();
        let old_capacity = self.capacity();
        self.size_pow = if old_size == 0 {
            MIN_NON_ZERO_SIZE_POW
        } else {
            self.size_pow + 1
        };


        self.clear_and_alloc_indices(interpreter, old_size);
        self.realloc_entries(interpreter, old_capacity);
    }

    fn realloc_entries(&mut self, interpreter: &mut Interpreter, old_capacity: usize) {
        let old_entries_word = self.entries;
        let old_entries = self.entries_array(interpreter);

        let free_word_count = Word((old_capacity * 3).into());
        let alloc_word_count = Word((self.capacity() * 3).into());
        self.entries = alloc!(interpreter.mem, alloc_word_count, "Dict entry array").0;

        let new_entries = self.entries_mut(interpreter);

        let mut write_idx = 0;
        for read_idx in 0..old_entries.len() {
            let entry = old_entries[read_idx];
            if entry.is_valid() {
                new_entries[write_idx] = entry;
                new_entries[write_idx].is_valid();
                write_idx += 1;
            }
        }
        // This should be equal to or lower than the previous entry_count since
        // it will remove tombstones
        self.entry_count = write_idx as u32;

        let new_entries = self.entries_array(interpreter);
        let mut indices = self.typed_indices(interpreter);
        for (entry_idx, entry) in new_entries.iter().enumerate() {
            let index_idx = self.probe_entry_realloc(interpreter, entry.hash as u32);
            indices.set(index_idx, entry_idx);
        }

        interpreter.mem.free(Word(old_entries_word), free_word_count);
    }

    fn indices_stride_bytes(&self, size: usize) -> usize {
        if size == 0 {
            0
        } else if size <= (u8::MAX as usize) + 1 {
            1
        } else if size <= (u16::MAX as usize) + 1 {
            2
        } else {
            4
        }
    }

    fn typed_indices(&self, interpreter: &Interpreter) -> TypedIndices {
        match self.index_size() {
            0 => TypedIndices::Zero,
            x if x <= (u8::MAX as usize) + 1 => TypedIndices::U8(
                self.indicies_mut::<u8>(interpreter)
            ),
            x if x <= (u16::MAX as usize) + 1 => TypedIndices::U16(
                self.indicies_mut::<u16>(interpreter)
            ),
            x if x <= (u32::MAX as usize) + 1 => TypedIndices::U32(
                self.indicies_mut::<u32>(interpreter)
            ),
            _ => panic!("Size over u64"),
        }
    }

    /**
     * Clear the indices array with current size
     */
    fn clear_and_alloc_indices(&mut self, interpreter: &mut Interpreter, old_size: usize) {
        let new_size = self.index_size();
        let free_word_count = if old_size == 0 {
            Word(0.into())
        } else {
            Word(old_size.div_ceil(8 / self.indices_stride_bytes(old_size)).into())
        };
        let alloc_word_count = if new_size == 0 {
            Word(0.into())
        } else {
            Word(new_size.div_ceil(8 / self.indices_stride_bytes(new_size)).into())
        };


        interpreter.mem.free(Word(self.indices), free_word_count);
        self.indices = alloc!(interpreter.mem, alloc_word_count, "Dict index array").0;

        match self.typed_indices(interpreter) {
            TypedIndices::Zero => (),
            TypedIndices::U8(indices) => {
                for x in indices.iter_mut() {
                    *x = u8::MAX;
                }
            },
            TypedIndices::U16(indices) => {
                for x in indices.iter_mut() {
                    *x = u16::MAX;
                }
            },
            TypedIndices::U32(indices) => {
                for x in indices.iter_mut() {
                    *x = u32::MAX;
                }
            },
        }
    }

    fn capacity(&self) -> usize {
        Self::capacity_for_size_pow(self.size_pow)
    }

    fn capacity_for_size_pow(size_pow: u8) -> usize {
        if size_pow == 0 {
            0
        } else {
            let index_size = 1 << size_pow;
            ((index_size * 2) / 3) as usize
        }
    }

    fn index_size(&self) -> usize {
        if self.size_pow == 0 {
            0
        } else {
            (1 << self.size_pow) as usize
        }
    }

    fn mask(&self) -> usize {
        (self.index_size() - 1) as usize
    }

    fn probe_entry_realloc(&self, interpreter: &Interpreter, longhash: u32) -> usize {
        let mask = self.mask();

        let hash: usize = (longhash as usize) & mask;
        let mut idx = hash & mask;
        match self.typed_indices(interpreter) {
            TypedIndices::Zero => panic!("Can't probe empty dict"),
            TypedIndices::U8(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u8::MAX {
                        return idx;
                    } else if indices[idx] == u8::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            },
            TypedIndices::U16(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u16::MAX {
                        return idx;
                    } else if indices[idx] == u16::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            },
            TypedIndices::U32(indices) => {
                for _ in 0..self.index_size() {
                    if indices[idx] == u32::MAX {
                        return idx;
                    } else if indices[idx] == u32::MAX - 1 {
                        panic!("Found tombstone during dict entry realloc!");
                    }
                    idx = (idx + 1) & mask;
                }
            },
        }

        panic!("Probe entry realloc failed probing!");
    }

    fn probe(&self, interpreter: &mut Interpreter, key: ShimValue) -> Result<DictSlot<'_>, String> {
        let longhash = key.hash(interpreter)? as usize;
        let mask = self.mask();

        let mut idx = longhash & mask;

        let mut freeslot = None;

        let indices = self.typed_indices(interpreter);
        // Linear probe for now
        for _ in 0..self.index_size() {
            if indices.is_unset(idx) {
                if freeslot == None {
                    freeslot = Some(idx);
                }
                break
            } else if indices.is_tombstone(idx) {
                if freeslot == None {
                    freeslot = Some(idx);
                }
            } else {
                // Hash matches, let's check the entry and see if the key matches
                let entry_idx = indices.get(idx);
                let entry = self.get_entry_mut(interpreter, entry_idx as usize);
                if key.equal_inner(interpreter, &entry.key)? {
                    return Ok(DictSlot::Occupied(idx as usize, entry));
                }
                // Otherwise continue probing
            }
            idx = (idx + 1) & mask;
        }
        let idx = match freeslot {
            Some(idx) => idx,
            None => {
                eprintln!("{self:#?}");
                eprintln!("Capacity: {:#?}  Mask: {}", self.capacity(), mask);
                panic!("Could not find free slot");
            },
        };
        match indices {
            TypedIndices::Zero => panic!("probing nothing"),
            TypedIndices::U8(_) => Ok(DictSlot::UnoccupiedU8(longhash as u32, idx as usize)),
            TypedIndices::U16(_) => Ok(DictSlot::UnoccupiedU16(longhash as u32, idx as usize)),
            TypedIndices::U32(_) => Ok(DictSlot::UnoccupiedU32(longhash as u32, idx as usize)),
        }
    }

    pub(crate) fn set(&mut self, interpreter: &mut Interpreter, key: ShimValue, val: ShimValue) -> Result<(), String> {
        if self.entry_count as usize == self.capacity() {
            self.expand_capacity(interpreter);
        }

        match self.probe(interpreter, key)? {
            DictSlot::Occupied(_, entry) => {
                entry.key = key;
                entry.value = val;
            },
            DictSlot::UnoccupiedU8(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u8>(interpreter)[idx] = entry_idx as u8;
                self.entries_mut(interpreter)[entry_idx].is_valid();
                self.entries_array(interpreter)[entry_idx].is_valid();
                self.used += 1;
            },
            DictSlot::UnoccupiedU16(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u16>(interpreter)[idx] = entry_idx as u16;
                self.used += 1;
            },
            DictSlot::UnoccupiedU32(longhash, idx) => {
                let entry_idx = self.set_entry(interpreter, longhash, key, val);
                self.indicies_mut::<u32>(interpreter)[idx] = entry_idx as u32;
                self.used += 1;
            },
        }

        Ok(())
    }

    pub(crate) fn pop(&mut self, interpreter: &mut Interpreter, key: ShimValue, default: Option<ShimValue>) -> Result<ShimValue, String> {
        match self.probe(interpreter, key) {
            Ok(DictSlot::Occupied(indices_idx, entry)) => {
                let value = entry.value;
                entry.hash = 0;
                entry.key = ShimValue::Uninitialized;
                entry.value = ShimValue::Uninitialized;

                let mut indices = self.typed_indices(interpreter);
                indices.set_tombstone(indices_idx);

                // We don't decrement the entry_count since that entry still exists
                self.used -= 1;

                Ok(value)
            },
            Ok(_) => {
                if let Some(default) = default {
                    Ok(default)
                } else {
                    Err(format!("Key {key:?} not found in dict"))
                }
            },
            _ => todo!(),
        }
    }

    fn indicies_mut<T>(&self, interpreter: &Interpreter) -> &'static mut [T] {
        let stride = std::mem::size_of::<T>();
        let size = 1 << self.size_pow;
        let start = usize::from(self.indices);
        let len = size / stride;
        let u64_slice = &interpreter.mem.mem[start..start + len];
        unsafe {
            std::slice::from_raw_parts_mut(
                u64_slice.as_ptr() as *mut T,
                u64_slice.len() * stride,
            )
        }
    }

    /**
     * Return the valid part of the entries array
     */
    pub(crate) fn entries_array(&self, interpreter: &Interpreter) -> &'static [DictEntry] {
        unsafe {
            let u64_slice = &interpreter.mem.mem[
                usize::from(self.entries)..
                usize::from(self.entries)+3*(self.entry_count as usize)
            ];
            std::slice::from_raw_parts(
                u64_slice.as_ptr() as *const DictEntry,
                u64_slice.len() / 3,
            )
        }
    }

    /**
     * Return the entire capacity of the entries table
     */
    fn entries_mut(&self, interpreter: &mut Interpreter) -> &'static mut [DictEntry] {
        unsafe {
            let u64_slice = &mut interpreter.mem.mem[
                usize::from(self.entries)..
                usize::from(self.entries)+3*(self.capacity() as usize)
            ];
            std::slice::from_raw_parts_mut(
                u64_slice.as_mut_ptr() as *mut DictEntry,
                u64_slice.len() / 3,
            )
        }
    }

    fn get_entry(&self, interpreter: &Interpreter, idx: usize) -> &DictEntry {
        unsafe{std::mem::transmute(&interpreter.mem.mem[
            usize::from(self.entries)+3*idx
        ])}
    }

    fn get_entry_mut(&self, interpreter: &mut Interpreter, idx: usize) -> &mut DictEntry {
        unsafe{std::mem::transmute(&mut interpreter.mem.mem[
            usize::from(self.entries)+3*idx
        ])}
    }

    fn set_entry(&mut self, interpreter: &mut Interpreter, hash: u32, key: ShimValue, val: ShimValue) -> usize {
        let entry = self.get_entry_mut(interpreter, self.entry_count as usize);
        entry.hash = hash as u64;
        entry.key = key;
        entry.value = val;

        let entry_idx = self.entry_count;
        self.entry_count += 1;
        entry_idx as usize
    }

    pub(crate) fn shrink_to_fit(&mut self, interpreter: &mut Interpreter) {
        if self.used == 0 {
            // Empty dict - reset to minimal size
            let old_size = self.index_size();
            let old_capacity = self.capacity();
            
            if old_size == 0 {
                return; // Already minimal
            }
            
            self.size_pow = 0;
            self.clear_and_alloc_indices(interpreter, old_size);
            
            // Free the old entries
            let free_word_count = Word((old_capacity * 3).into());
            interpreter.mem.free(Word(self.entries), free_word_count);
            self.entries = 0.into();
            self.entry_count = 0;
            return;
        }
        
        // Calculate the optimal size_pow for the current number of used entries
        // We want capacity to be at least used, and index_size = capacity * 3 / 2
        // Since index_size must be a power of 2, we find the smallest power of 2
        // such that (2^size_pow * 2 / 3) >= used
        let min_capacity = self.used as usize;
        // Start with MIN_NON_ZERO_SIZE_POW, which matches expand_capacity's initial size
        let mut optimal_size_pow = MIN_NON_ZERO_SIZE_POW;
        
        // Upper bound of 31 prevents undefined behavior from 1 << 32 and ensures
        // we stay within u32 limits for entry_count/used fields.
        // Loop condition is <= 31 to allow checking if size_pow=31 is sufficient.
        while optimal_size_pow <= 31 {
            let test_capacity = Self::capacity_for_size_pow(optimal_size_pow);
            if test_capacity >= min_capacity {
                break;
            }
            optimal_size_pow += 1;
        }
        
        // If the optimal size is the same or larger than current, no need to shrink
        if optimal_size_pow >= self.size_pow {
            return;
        }
        
        let old_size = self.index_size();
        let old_capacity = self.capacity();
        self.size_pow = optimal_size_pow;
        
        self.clear_and_alloc_indices(interpreter, old_size);
        self.realloc_entries(interpreter, old_capacity);
    }
}

pub(crate) struct ShimList {
    // The memory is limited to u24, so we know there can't be more than this
    // number of values
    pub(crate) len: u24,
    // We don't really need any more than 64 distinct capacities
    pub(crate) capacity_lut: u8,
    // Add 1 byte of padding so that ShimList is 8 bytes
    _pad: u8,
    // Memory position of the list data
    pub(crate) data: u24,
}

const _: () = {
    assert!(std::mem::size_of::<ShimList>() == 8);
};

impl ShimList {
    pub(crate) fn new() -> Self {
        Self {
            len: 0.into(),
            capacity_lut: 0,
            _pad: 0,
            data: 0.into(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len.into()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn capacity(&self) -> usize {
        LIST_CAPACITY_LUT[self.capacity_lut as usize] as usize
    }

    pub(crate) fn wrap_idx(&self, idx: isize) -> Result<usize, String> {
        if idx >= self.len() as isize {
            return Err(format!("Index {idx} is out of bounds"));
        }

        Ok(
            if idx < 0 {
                let updated_idx = self.len() as isize + idx;
                if updated_idx < 0 {
                    return Err(format!("Index {idx} is out of bounds"));
                } else {
                    updated_idx as usize
                }
            } else {
                idx as usize
            }
        )
    }

    pub(crate) fn get(&self, mem: &MMU, idx: isize) -> Result<ShimValue, String> {
        let idx = self.wrap_idx(idx)?;
        unsafe { Ok(ShimValue::from_u64(mem.mem[usize::from(self.data) + idx])) }
    }

    pub(crate) fn set(&self, mem: &mut MMU, idx: isize, value: ShimValue) -> Result<(), String> {
        let idx = self.wrap_idx(idx)?;
        mem.mem[usize::from(self.data) + idx] = value.to_u64();
        Ok(())
    }

    pub(crate) fn push(&mut self, mem: &mut MMU, val: ShimValue) {
        if self.len() == self.capacity() {
            let old_capacity = self.capacity();
            self.capacity_lut += 1;
            let new_capacity = self.capacity();

            let old_data = usize::from(self.data);
            let word_count: Word = new_capacity.into();
            self.data = alloc!(mem, word_count, "List data").0;

            let new_data = usize::from(self.data);

            for idx in 0..self.len() {
                mem.mem[new_data+idx] = mem.mem[old_data+idx];
            }

            mem.free(old_data.into(), Word(old_capacity.into()));
        }

        mem.mem[usize::from(self.data)+self.len()] = val.to_u64();
        self.len = (usize::from(self.len) + 1).into();
    }
}
const _: () = { assert!(std::mem::size_of::<ShimList>() == 8); };

pub(crate) fn shim_dict(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    if args.args.len() != 0 {
        return Err(format!("Can't provide positional args to dict()"));
    }

    let retval = interpreter.mem.alloc_dict();
    let dict = retval.dict_mut(interpreter)?;

    for (key, val) in args.kwargs.clone().into_iter() {
        let key = interpreter.mem.alloc_str(&key);
        dict.set(interpreter, key, val)?;
    }

    Ok(retval)
}

pub(crate) fn shim_range(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let start = unpacker.required(b"start")?;
    let end = unpacker.required(b"end")?;
    unpacker.end()?;

    let range = RangeNative {
        start: start,
        end: end,
    };
    Ok(interpreter.mem.alloc_native(range))
}

pub(crate) fn shim_print(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let _zone = zone_scoped!("shim_print");
    for (idx, arg) in args.args.iter().enumerate() {
        if idx != 0 {
            print!(" ");
        }
        print!("{}", arg.to_string(interpreter));
    }

    println!();
    Ok(ShimValue::None)
}

pub(crate) fn shim_assert(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    if !args.kwargs.is_empty() {
        return Err(format!("Assert doesn't take keyword arguments"));
    }
    if args.len() > 2 {
        return Err(format!("Assert got more than two arguments! {:?}", args));
    }
    if args.len() == 0 {
        return Ok(ShimValue::None);
    }

    if !args.args[0].is_truthy(interpreter)? {
        let msg = if args.len() > 1 {
            args.args[1].to_string(interpreter)
        } else {
            format!("Assert Failed: {:?} not truthy", args.args[0])
        };
        Err(msg)
    } else {
        Ok(ShimValue::None)
    }
}

pub(crate) fn shim_panic(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut out = String::new();
    for (idx, arg) in args.args.iter().enumerate() {
        if idx != 0 {
            out.push(' ');
        }
        out.push_str(&format!("{}", arg.to_string(interpreter)));
    }

    out.push('\n');
    Err(out)
}

//enum ShimSortKey {
//    Bytes(&[u8]),
//    Int(i32),
//    Float(f32),
//}

pub(crate) fn shim_list_sort(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.optional(b"key");
    unpacker.end()?;

    // Create a vector of (index, value, sort_key) tuples to maintain stability
    let mut items_with_keys: Vec<(usize, ShimValue, ShimValue)> = Vec::new();
    
    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        
        let sort_key = if let Some(key) = key {
            let mut args = ArgBundle::new();
            args.args.push(item);
            match key.call(interpreter, &mut args)? {
                CallResult::ReturnValue(val) => val,
                CallResult::PC(pc, captured_scope) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        args,
                        &mut new_env,
                    )?
                },
            }
        } else {
            item
        };
        
        items_with_keys.push((idx, item, sort_key));
    }
    
    // Perform stable sort by comparing sort keys
    items_with_keys.sort_by(|a, b| {
        let (idx_a, _, key_a) = a;
        let (idx_b, _, key_b) = b;
        
        // Try to compare the keys
        match compare_values(interpreter, key_a, key_b) {
            Ok(ordering) => ordering,
            Err(_) => {
                // If comparison fails, maintain original order (stability)
                idx_a.cmp(idx_b)
            }
        }
    });
    
    // Mutate the list in place
    let lst_mut = obj.list_mut(interpreter)?;
    for (idx, (_, item, _)) in items_with_keys.iter().enumerate() {
        lst_mut.set(&mut interpreter.mem, idx as isize, *item)?;
    }
    
    Ok(ShimValue::None)
}

// Helper function to compare two ShimValues for sorting/ordering purposes.
// This function returns an Ordering to determine relative position in a sorted sequence.
// For equality checks, use ShimValue::equal_inner instead.
pub(crate) fn compare_values(interpreter: &mut Interpreter, a: &ShimValue, b: &ShimValue) -> Result<std::cmp::Ordering, String> {
    use std::cmp::Ordering;
    
    match (a, b) {
        (ShimValue::Integer(x), ShimValue::Integer(y)) => Ok(x.cmp(y)),
        (ShimValue::Float(x), ShimValue::Float(y)) => {
            // Handle NaN comparison by treating NaN as equal to itself
            if x.is_nan() && y.is_nan() {
                Ok(Ordering::Equal)
            } else if x.is_nan() {
                Ok(Ordering::Greater)
            } else if y.is_nan() {
                Ok(Ordering::Less)
            } else if x < y {
                Ok(Ordering::Less)
            } else if x > y {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        },
        (ShimValue::Integer(x), ShimValue::Float(y)) => {
            let x_f = *x as f32;
            if x_f < *y {
                Ok(Ordering::Less)
            } else if x_f > *y {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        },
        (ShimValue::Float(x), ShimValue::Integer(y)) => {
            let y_f = *y as f32;
            if *x < y_f {
                Ok(Ordering::Less)
            } else if *x > y_f {
                Ok(Ordering::Greater)
            } else {
                Ok(Ordering::Equal)
            }
        },
        (ShimValue::String(..), ShimValue::String(..)) => {
            let str_a = a.string(interpreter)?;
            let str_b = b.string(interpreter)?;
            Ok(str_a.cmp(&str_b))
        },
        (ShimValue::Bool(x), ShimValue::Bool(y)) => Ok(x.cmp(y)),
        (ShimValue::None, ShimValue::None) => Ok(Ordering::Equal),
        (ShimValue::List(_), ShimValue::List(_)) => {
            // Compare lists lexicographically
            let lst_a = a.list(interpreter)?;
            let lst_b = b.list(interpreter)?;
            
            let min_len = std::cmp::min(lst_a.len(), lst_b.len());
            for i in 0..min_len {
                let item_a = lst_a.get(&interpreter.mem, i as isize)?;
                let item_b = lst_b.get(&interpreter.mem, i as isize)?;
                match compare_values(interpreter, &item_a, &item_b)? {
                    Ordering::Equal => continue,
                    other => return Ok(other),
                }
            }
            Ok(lst_a.len().cmp(&lst_b.len()))
        },
        _ => Err(format!("Cannot compare {:?} and {:?}", a, b)),
    }
}

pub(crate) fn shim_list_filter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.optional(b"key");
    unpacker.end()?;

    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;

    for idx in 0..lst.len() {
        let input = lst.get(&interpreter.mem, idx as isize)?;
        let result = if let Some(key) = key {
            let mut args = ArgBundle::new();
            args.args.push(input);
            match key.call(interpreter, &mut args)? {
                CallResult::ReturnValue(val) => {
                    val
                },
                CallResult::PC(pc, captured_scope) => {
                    let mut new_env = Environment::with_scope(captured_scope);
                    let val = interpreter.execute_bytecode_extended(
                        &mut (pc as usize),
                        args,
                        &mut new_env,
                    )?;
                    val
                },
            } 
        } else {
            input
        };
        if result.is_truthy(interpreter)? {
            new_lst.push(&mut interpreter.mem, input);
        }
    }

    Ok(new_lst_val)
}

pub(crate) fn shim_list_map(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.required(b"key")?;
    unpacker.end()?;

    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;

    for idx in 0..lst.len() {
        let input = lst.get(&interpreter.mem, idx as isize)?;
        let mut args = ArgBundle::new();
        args.args.push(input);
        let output = match key.call(interpreter, &mut args)? {
            CallResult::ReturnValue(val) => {
                val
            },
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                let val = interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    args,
                    &mut new_env,
                )?;
                val
            },
        };
        new_lst.push(&mut interpreter.mem, output);
    }

    Ok(new_lst_val)
}

pub(crate) fn shim_list_len(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(lst.len() as i32))
}

pub(crate) fn shim_list_append(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    let item = unpacker.required(b"item")?;
    unpacker.end()?;

    lst.push(&mut interpreter.mem, item);

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_iter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(ListIterator {lst: obj, idx: 0}))
}

pub(crate) fn shim_list_clear(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    unpacker.end()?;

    lst.len = 0.into();
    
    Ok(ShimValue::None)
}

pub(crate) fn shim_list_extend(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let iterable = unpacker.required(b"iterable")?;
    unpacker.end()?;

    // Get the iterator for the iterable
    let mut iter_args = ArgBundle::new();
    let iterator = iterable.get_attr(interpreter, b"iter")?.call(interpreter, &mut iter_args)?;
    let iterator = match iterator {
        CallResult::ReturnValue(val) => val,
        CallResult::PC(pc, captured_scope) => {
            let mut new_env = Environment::with_scope(captured_scope);
            interpreter.execute_bytecode_extended(
                &mut (pc as usize),
                iter_args,
                &mut new_env,
            )?
        },
    };

    // Get the next method
    let next_method = iterator.get_attr(interpreter, b"next")?;

    // Iterate and append each item
    loop {
        let mut next_args = ArgBundle::new();
        
        let result = match next_method.call(interpreter, &mut next_args)? {
            CallResult::ReturnValue(val) => val,
            CallResult::PC(pc, captured_scope) => {
                let mut new_env = Environment::with_scope(captured_scope);
                interpreter.execute_bytecode_extended(
                    &mut (pc as usize),
                    next_args,
                    &mut new_env,
                )?
            },
        };

        // Break if we get None (end of iteration)
        if result.is_none() {
            break;
        }

        // Append the item to the list
        let lst = obj.list_mut(interpreter)?;
        lst.push(&mut interpreter.mem, result);
    }

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_index(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let value = unpacker.required(b"value")?;
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        if item.equal_inner(interpreter, &value)? {
            return Ok(ShimValue::Integer(idx as i32));
        }
    }

    Ok(default.unwrap_or(ShimValue::None))
}

pub(crate) fn shim_list_insert(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let index = unpacker.required(b"index")?;
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let idx = index.integer()? as isize;

    let lst = obj.list_mut(interpreter)?;
    let len = lst.len();

    // Handle negative and out-of-bounds indices like Python
    let insert_idx = if idx < 0 {
        // Negative indices count from the end
        (len as isize + idx).max(0) as usize
    } else if idx as usize > len {
        // Positive indices beyond length append at the end
        len
    } else {
        idx as usize
    };

    // Add a new element at the end (this will resize if needed)
    lst.push(&mut interpreter.mem, ShimValue::None);

    // Shift elements to make room
    for i in (insert_idx..len).rev() {
        let val = lst.get(&interpreter.mem, i as isize)?;
        lst.set(&mut interpreter.mem, (i + 1) as isize, val)?;
    }

    // Insert the value
    lst.set(&mut interpreter.mem, insert_idx as isize, value)?;

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_pop(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let index = unpacker.optional(b"index");
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    let lst = obj.list_mut(interpreter)?;
    
    if lst.is_empty() {
        return Ok(default.unwrap_or(ShimValue::None));
    }

    // Determine which index to pop
    let pop_idx = if let Some(idx_val) = index {
        let idx = idx_val.integer()? as isize;
        lst.wrap_idx(idx)?
    } else {
        // Default to last element
        lst.len() - 1
    };

    // Get the value at the index
    let value = lst.get(&interpreter.mem, pop_idx as isize)?;

    // Shift elements after pop_idx to the left
    for i in pop_idx..(lst.len() - 1) {
        let next_val = lst.get(&interpreter.mem, (i + 1) as isize)?;
        lst.set(&mut interpreter.mem, i as isize, next_val)?;
    }

    // Decrease the length
    lst.len = (lst.len() - 1).into();

    Ok(value)
}

pub(crate) fn shim_list_sorted(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    let key = unpacker.optional(b"key");
    unpacker.end()?;

    // Create a new list with the same elements
    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;
    
    for idx in 0..lst.len() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        new_lst.push(&mut interpreter.mem, item);
    }

    // Sort the new list using the existing sort logic
    let mut sort_args = ArgBundle::new();
    sort_args.args.push(new_lst_val);
    if let Some(k) = key {
        sort_args.kwargs.push((b"key".to_vec(), k));
    }
    shim_list_sort(interpreter, &sort_args)?;

    Ok(new_lst_val)
}

pub(crate) fn shim_list_reverse(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list_mut(interpreter)?;
    unpacker.end()?;

    let len = lst.len();
    for i in 0..(len / 2) {
        let left = lst.get(&interpreter.mem, i as isize)?;
        let right = lst.get(&interpreter.mem, (len - 1 - i) as isize)?;
        lst.set(&mut interpreter.mem, i as isize, right)?;
        lst.set(&mut interpreter.mem, (len - 1 - i) as isize, left)?;
    }

    Ok(ShimValue::None)
}

pub(crate) fn shim_list_reversed(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    let lst = obj.list(interpreter)?;
    unpacker.end()?;

    // Create a new list with reversed elements
    let new_lst_val = interpreter.mem.alloc_list();
    let new_lst = new_lst_val.list_mut(interpreter)?;
    
    for idx in (0..lst.len()).rev() {
        let item = lst.get(&interpreter.mem, idx as isize)?;
        new_lst.push(&mut interpreter.mem, item);
    }

    Ok(new_lst_val)
}

pub(crate) fn shim_dict_iter(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(DictKeysIterator {dict: obj, idx: 0}))
}

pub(crate) fn shim_dict_keys(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(DictKeysIterator {dict: obj, idx: 0}))
}

pub(crate) fn shim_dict_values(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(DictValuesIterator {dict: obj, idx: 0}))
}

pub(crate) fn shim_dict_items(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let obj = unpacker.required(b"obj")?;
    unpacker.end()?;

    Ok(interpreter.mem.alloc_native(DictItemsIterator {dict: obj, idx: 0}))
}

pub(crate) fn shim_dict_pop(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    let default = unpacker.optional(b"default");
    unpacker.end()?;

    dict.pop(interpreter, key, default)
}

pub(crate) fn shim_dict_index_set(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    dict.set(interpreter, key, value)?;

    Ok(ShimValue::None)
}

pub(crate) fn shim_dict_index_get(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    unpacker.end()?;

    dict.get(interpreter, key)
}

pub(crate) fn shim_dict_index_has(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    let key = unpacker.required(b"key")?;
    unpacker.end()?;

    Ok(ShimValue::Bool(dict.get(interpreter, key).is_ok()))
}

pub(crate) fn shim_dict_len(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(dict.len() as i32))
}

pub(crate) fn shim_dict_shrink_to_fit(
    interpreter: &mut Interpreter,
    args: &ArgBundle,
) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let dict = binding.dict_mut(interpreter)?;
    unpacker.end()?;

    dict.shrink_to_fit(interpreter);
    Ok(ShimValue::None)
}

pub(crate) fn shim_str_len(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let binding = unpacker.required(b"obj")?;
    let s = binding.string(interpreter)?;
    unpacker.end()?;

    Ok(ShimValue::Integer(s.len() as i32))
}

pub(crate) fn get_type_name(value: &ShimValue) -> &'static str {
    match value {
        ShimValue::Uninitialized => "uninitialized",
        ShimValue::Unit => "unit",
        ShimValue::None => "none",
        ShimValue::Integer(_) => "int",
        ShimValue::Float(_) => "float",
        ShimValue::Bool(_) => "bool",
        ShimValue::Fn(_) => "function",
        ShimValue::BoundMethod(_, _) => "bound method",
        ShimValue::BoundNativeMethod(_) => "bound native method",
        ShimValue::NativeFn(_) => "native function",
        ShimValue::String(..) => "string",
        ShimValue::List(_) => "list",
        ShimValue::Dict(_) => "dict",
        ShimValue::StructDef(_) => "struct definition",
        ShimValue::Struct(_) => "struct",
        ShimValue::Native(_) => "native object",
        ShimValue::Environment(_) => "environment",
    }
}

fn trim_bytes(s: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = s.len();
    
    // Trim from start
    while start < end && s[start].is_ascii_whitespace() {
        start += 1;
    }
    
    // Trim from end
    while end > start && s[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    
    &s[start..end]
}

fn parse_string_to<T: std::str::FromStr>(
    s: &[u8],
    type_name: &str,
) -> Result<T, String> {
    let trimmed = trim_bytes(s);
    unsafe {
        std::str::from_utf8_unchecked(trimmed).parse::<T>()
            .map_err(|_| {
                let string_repr = std::str::from_utf8(s).unwrap_or("<invalid utf8>");
                format!("Cannot convert string '{}' to {}", string_repr, type_name)
            })
    }
}

pub(crate) fn shim_str(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let string_repr = value.to_string(interpreter);
    let bytes = string_repr.as_bytes();
    Ok(interpreter.mem.alloc_str(bytes))
}

pub(crate) fn shim_int(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Integer(i)),
        ShimValue::Float(f) => Ok(ShimValue::Integer(f as i32)),
        ShimValue::Bool(b) => Ok(ShimValue::Integer(if b { 1 } else { 0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<i32>(s, "int").map(ShimValue::Integer)
        },
        _ => Err(format!("Cannot convert {} to int", get_type_name(&value)))
    }
}

pub(crate) fn shim_float(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    match value {
        ShimValue::Integer(i) => Ok(ShimValue::Float(i as f32)),
        ShimValue::Float(f) => Ok(ShimValue::Float(f)),
        ShimValue::Bool(b) => Ok(ShimValue::Float(if b { 1.0 } else { 0.0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<f32>(s, "float").map(ShimValue::Float)
        },
        _ => Err(format!("Cannot convert {} to float", get_type_name(&value)))
    }
}

pub(crate) fn shim_try_int(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let result = match value {
        ShimValue::Integer(i) => Some(ShimValue::Integer(i)),
        ShimValue::Float(f) => Some(ShimValue::Integer(f as i32)),
        ShimValue::Bool(b) => Some(ShimValue::Integer(if b { 1 } else { 0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<i32>(s, "int")
                .map(ShimValue::Integer)
                .ok()
        },
        _ => None
    };

    Ok(result.unwrap_or(ShimValue::None))
}

pub(crate) fn shim_try_float(interpreter: &mut Interpreter, args: &ArgBundle) -> Result<ShimValue, String> {
    let mut unpacker = ArgUnpacker::new(args);
    let value = unpacker.required(b"value")?;
    unpacker.end()?;

    let result = match value {
        ShimValue::Integer(i) => Some(ShimValue::Float(i as f32)),
        ShimValue::Float(f) => Some(ShimValue::Float(f)),
        ShimValue::Bool(b) => Some(ShimValue::Float(if b { 1.0 } else { 0.0 })),
        ShimValue::String(..) => {
            let s = value.string(interpreter)?;
            parse_string_to::<f32>(s, "float")
                .map(ShimValue::Float)
                .ok()
        },
        _ => None
    };

    Ok(result.unwrap_or(ShimValue::None))
}
