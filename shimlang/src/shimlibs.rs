use crate::lex::debug_u8s;
use crate::runtime::*;
use shm_tracy::*;
use shm_tracy::zone_scoped;

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
                        // TODO: this doesn't even have print...
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
                    // TODO: this doesn't even have print...
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
