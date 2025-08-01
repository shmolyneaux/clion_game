//! Pretty printer implementation for Facet types

use alloc::collections::VecDeque;
use core::{
    fmt::{self, Write},
    hash::{Hash, Hasher},
    str,
};
use std::{collections::HashMap, hash::DefaultHasher};

use facet_core::{
    Def, Facet, FieldFlags, PointerType, PrimitiveType, SequenceType, StructKind, TextualType,
    Type, TypeNameOpts, UserType,
};
use facet_reflect::{Peek, ValueId};

use crate::color::ColorGenerator;

/// A formatter for pretty-printing Facet types
pub struct PrettyPrinter {
    indent_size: usize,
    max_depth: Option<usize>,
    color_generator: ColorGenerator,
    use_colors: bool,
    list_u8_as_bytes: bool,
}

impl Default for PrettyPrinter {
    fn default() -> Self {
        Self {
            indent_size: 2,
            max_depth: None,
            color_generator: ColorGenerator::default(),
            use_colors: std::env::var_os("NO_COLOR").is_none(),
            list_u8_as_bytes: true,
        }
    }
}

/// Stack state for iterative formatting
enum StackState {
    Start,
    ProcessStructField { field_index: usize },
    ProcessSeqItem { item_index: usize, kind: SeqKind },
    ProcessBytesItem { item_index: usize },
    ProcessMapEntry,
    Finish,
    OptionFinish,
}

enum SeqKind {
    List,
    Tuple,
}

/// Stack item for iterative traversal
struct StackItem<'mem, 'facet, 'shape> {
    value: Peek<'mem, 'facet, 'shape>,
    format_depth: usize,
    type_depth: usize,
    state: StackState,
}

impl PrettyPrinter {
    /// Create a new PrettyPrinter with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the indentation size
    pub fn with_indent_size(mut self, size: usize) -> Self {
        self.indent_size = size;
        self
    }

    /// Set the maximum depth for recursive printing
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Set the color generator
    pub fn with_color_generator(mut self, generator: ColorGenerator) -> Self {
        self.color_generator = generator;
        self
    }

    /// Enable or disable colors
    pub fn with_colors(mut self, use_colors: bool) -> Self {
        self.use_colors = use_colors;
        self
    }

    /// Format a value to a string
    pub fn format<'a, T: Facet<'a>>(&self, value: &T) -> String {
        let value = Peek::new(value);

        let mut output = String::new();
        self.format_peek_internal(value, &mut output, &mut HashMap::new())
            .expect("Formatting failed");

        output
    }

    /// Format a value to a formatter
    pub fn format_to<'a, T: Facet<'a>>(
        &self,
        value: &T,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let value = Peek::new(value);
        self.format_peek_internal(value, f, &mut HashMap::new())
    }

    /// Format a value to a string
    pub fn format_peek(&self, value: Peek<'_, '_, '_>) -> String {
        let mut output = String::new();
        self.format_peek_internal(value, &mut output, &mut HashMap::new())
            .expect("Formatting failed");
        output
    }

    /// Internal method to format a Peek value
    pub(crate) fn format_peek_internal<'shape>(
        &self,
        initial_value: Peek<'_, '_, 'shape>,
        f: &mut impl Write,
        visited: &mut HashMap<ValueId<'shape>, usize>,
    ) -> fmt::Result {
        // Create a queue for our stack items
        let mut stack = VecDeque::new();

        // Push the initial item
        stack.push_back(StackItem {
            value: initial_value,
            format_depth: 0,
            type_depth: 0,
            state: StackState::Start,
        });

        // Process items until the stack is empty
        while let Some(mut item) = stack.pop_back() {
            match item.state {
                StackState::Start => {
                    // Check if we've reached the maximum depth
                    if let Some(max_depth) = self.max_depth {
                        if item.format_depth > max_depth {
                            self.write_punctuation(f, "[")?;
                            write!(f, "...")?;
                            continue;
                        }
                    }

                    // Check for cycles - if we've seen this value before at a different type_depth
                    if let Some(&ptr_type_depth) = visited.get(&item.value.id()) {
                        // If the current type_depth is significantly deeper than when we first saw this value,
                        // we have a true cycle, not just a transparent wrapper
                        if item.type_depth > ptr_type_depth + 1 {
                            self.write_type_name(f, &item.value)?;
                            self.write_punctuation(f, " { ")?;
                            self.write_comment(
                                f,
                                &format!(
                                    "/* cycle detected at {} (first seen at type_depth {}) */",
                                    item.value.id(),
                                    ptr_type_depth
                                ),
                            )?;
                            self.write_punctuation(f, " }")?;
                            continue;
                        }
                    } else {
                        // First time seeing this value, record its type_depth
                        visited.insert(item.value.id(), item.type_depth);
                    }

                    // Process based on the peek variant and type
                    match (item.value.shape().def, item.value.shape().ty) {
                        // Handle scalar values
                        (Def::Scalar(_def), _) => {
                            self.format_scalar(item.value, f)?;
                        }
                        // Handle option types
                        (Def::Option(_def), _) => {
                            let option = item.value.into_option().unwrap();

                            // Print the Option name
                            self.write_type_name(f, &item.value)?;

                            if option.is_some() {
                                self.write_punctuation(f, "::Some(")?;

                                if let Some(inner_value) = option.value() {
                                    // Create a custom stack item for Option::Some value
                                    let start_item = StackItem {
                                        value: inner_value,
                                        format_depth: item.format_depth,
                                        type_depth: item.type_depth + 1,
                                        state: StackState::Start,
                                    };

                                    // Add a special close parenthesis item
                                    let close_paren_item = StackItem {
                                        value: item.value,
                                        format_depth: item.format_depth,
                                        type_depth: item.type_depth,
                                        state: StackState::OptionFinish,
                                    };

                                    // Process the value first, then handle closing
                                    stack.push_back(close_paren_item);
                                    stack.push_back(start_item);
                                }

                                // Skip to next item
                                continue;
                            } else {
                                self.write_punctuation(f, "::None")?;
                            }
                        }
                        // Handle tuple struct types
                        (_, Type::User(UserType::Struct(struct_def)))
                            if struct_def.kind == StructKind::Tuple =>
                        {
                            self.write_type_name(f, &item.value)?;
                            item.state = StackState::ProcessSeqItem {
                                item_index: 0,
                                kind: SeqKind::Tuple,
                            };
                            self.write_punctuation(f, " (")?;
                            writeln!(f)?;
                            item.format_depth += 1;
                            item.type_depth += 1;
                            stack.push_back(item);
                        }
                        // Handle regular struct types
                        (_, Type::User(UserType::Struct(_))) => {
                            let struct_ = item.value.into_struct().unwrap();

                            // Get struct doc comments from the shape
                            let doc_comments = item.value.shape().doc;
                            if !doc_comments.is_empty() {
                                for line in doc_comments {
                                    self.write_comment(f, &format!("///{}", line))?;
                                    writeln!(f)?;
                                }
                            }

                            // Print the struct name
                            self.write_type_name(f, &item.value)?;
                            self.write_punctuation(f, " {")?;

                            if struct_.field_count() == 0 {
                                self.write_punctuation(f, "}")?;
                                continue;
                            }

                            writeln!(f)?;

                            // Push back the item with the next state to continue processing fields
                            item.state = StackState::ProcessStructField { field_index: 0 };
                            item.format_depth += 1;
                            stack.push_back(item);
                        }
                        (Def::List(_), _) => {
                            self.handle_list(&mut stack, item, f)?;
                            continue;
                        }
                        (_, Type::Pointer(PointerType::Reference(r))) => {
                            'handle: {
                                let target = (r.target)();
                                match target.ty {
                                    Type::Sequence(
                                        SequenceType::Slice(_) | SequenceType::Array(_),
                                    ) => {
                                        self.handle_list(&mut stack, item, f)?;
                                        break 'handle;
                                    }
                                    Type::Primitive(primitive_type) => match primitive_type {
                                        PrimitiveType::Boolean => {}
                                        PrimitiveType::Numeric(_numeric_type) => {}
                                        PrimitiveType::Textual(textual_type) => {
                                            match textual_type {
                                                TextualType::Char => todo!(),
                                                TextualType::Str => {
                                                    // well we can print a string slice, that's no issue.
                                                    // `Peek` implements `Display` which forwards to the
                                                    // `Display` implementation of the underlying type.
                                                    if self.use_colors {
                                                        write!(f, "\x1b[33m{}\x1b[0m", item.value)?; // yellow
                                                    } else {
                                                        write!(f, "{}", item.value)?;
                                                    }
                                                    break 'handle;
                                                }
                                            }
                                        }
                                        PrimitiveType::Never => {}
                                    },
                                    _ => {
                                        write!(f, "unsupported reference type: {:?}", item.value)?;
                                    }
                                }
                            }
                        }
                        (Def::Map(_), _) => {
                            let _map = item.value.into_map().unwrap();
                            // Print the map name
                            self.write_type_name(f, &item.value)?;
                            self.write_punctuation(f, " {")?;
                            writeln!(f)?;

                            // Push back the item with the next state to continue processing map
                            item.state = StackState::ProcessMapEntry;
                            item.format_depth += 1;
                            // When recursing into a map, always increment format_depth
                            item.type_depth += 1; // Always increment type_depth for map operations
                            stack.push_back(item);
                        }
                        (_, Type::User(UserType::Enum(_))) => {
                            // When recursing into an enum, increment format_depth
                            // Only increment type_depth if we're moving to a different address
                            let enum_peek = item.value.into_enum().unwrap();

                            // Get the active variant or handle error
                            let variant = match enum_peek.active_variant() {
                                Ok(v) => v,
                                Err(_) => {
                                    // Print the enum name
                                    self.write_type_name(f, &item.value)?;
                                    write!(f, " /* cannot determine variant */")?;
                                    continue;
                                }
                            };

                            // Get enum and variant doc comments
                            let doc_comments = item.value.shape().doc;

                            // Display doc comments before the type name
                            for line in doc_comments {
                                self.write_comment(f, &format!("///{}", line))?;
                                writeln!(f)?;
                            }

                            // Show variant docs
                            for line in variant.doc {
                                self.write_comment(f, &format!("///{}", line))?;
                                writeln!(f)?;
                            }

                            // Print the enum name and separator
                            self.write_type_name(f, &item.value)?;
                            self.write_punctuation(f, "::")?;

                            // Variant docs are already handled above

                            // Get the active variant name - we've already checked above that we can get it
                            // This is the same variant, but we're repeating the code here to ensure consistency

                            // Apply color for variant name
                            if self.use_colors {
                                if self.use_colors {
                                    write!(f, "\x1b[1m{}\x1b[0m", variant.name)?; // bold
                                } else {
                                    write!(f, "{}", variant.name)?;
                                }
                            } else {
                                write!(f, "{}", variant.name)?;
                            }

                            // Process the variant fields based on the variant kind
                            match variant.data.kind {
                                StructKind::Unit => {
                                    // Unit variant has no fields, nothing more to print
                                }
                                StructKind::Tuple => {
                                    // Tuple variant, print the fields like a tuple
                                    self.write_punctuation(f, "(")?;

                                    // Check if there are any fields to print
                                    if variant.data.fields.is_empty() {
                                        self.write_punctuation(f, ")")?;
                                        continue;
                                    }

                                    writeln!(f)?;

                                    // Push back item to process fields
                                    item.state = StackState::ProcessStructField { field_index: 0 };
                                    item.format_depth += 1;
                                    stack.push_back(item);
                                }
                                StructKind::Struct => {
                                    // Struct variant, print the fields like a struct
                                    self.write_punctuation(f, " {")?;

                                    // Check if there are any fields to print
                                    let has_fields = !variant.data.fields.is_empty();

                                    if !has_fields {
                                        self.write_punctuation(f, " }")?;
                                        continue;
                                    }

                                    writeln!(f)?;

                                    // Push back item to process fields
                                    item.state = StackState::ProcessStructField { field_index: 0 };
                                    item.format_depth += 1;
                                    stack.push_back(item);
                                }
                                _ => {
                                    // Other variant kinds that might be added in the future
                                    write!(f, " /* unsupported variant kind */")?;
                                }
                            }
                        }
                        (_, Type::Pointer(PointerType::Function(_))) => {
                            // Just print the type name for function pointers
                            self.write_type_name(f, &item.value)?;
                            write!(f, " /* function pointer (not yet supported) */")?;
                        }
                        _ => {
                            write!(f, "unsupported peek variant: {:?}", item.value)?;
                        }
                    }
                }
                StackState::ProcessStructField { field_index } => {
                    // Handle both struct and enum fields
                    if let Type::User(UserType::Struct(struct_)) = item.value.shape().ty {
                        let peek_struct = item.value.into_struct().unwrap();
                        if field_index >= struct_.fields.len() {
                            // All fields processed, write closing brace
                            write!(
                                f,
                                "{:width$}{}",
                                "",
                                self.style_punctuation("}"),
                                width = (item.format_depth - 1) * self.indent_size
                            )?;
                            continue;
                        }

                        let field = struct_.fields[field_index];
                        let field_value = peek_struct.field(field_index).unwrap();

                        // Field doc comment
                        if !field.doc.is_empty() {
                            // Only add new line if not the first field
                            if field_index > 0 {
                                writeln!(f)?;
                            }
                            // Hard-code consistent indentation for doc comments
                            for line in field.doc {
                                // Use exactly the same indentation as fields (2 spaces)
                                write!(
                                    f,
                                    "{:width$}",
                                    "",
                                    width = item.format_depth * self.indent_size
                                )?;
                                self.write_comment(f, &format!("///{}", line))?;
                                writeln!(f)?;
                            }
                        }

                        // Field name
                        write!(
                            f,
                            "{:width$}",
                            "",
                            width = item.format_depth * self.indent_size
                        )?;
                        self.write_field_name(f, field.name)?;
                        self.write_punctuation(f, ": ")?;

                        // Check if field is sensitive
                        if field.flags.contains(FieldFlags::SENSITIVE) {
                            // Field value is sensitive, use write_redacted
                            self.write_redacted(f, "[REDACTED]")?;
                            self.write_punctuation(f, ",")?;
                            writeln!(f)?;

                            item.state = StackState::ProcessStructField {
                                field_index: field_index + 1,
                            };
                            stack.push_back(item);
                        } else {
                            // Field value is not sensitive, format normally
                            // Push back current item to continue after formatting field value
                            item.state = StackState::ProcessStructField {
                                field_index: field_index + 1,
                            };

                            let finish_item = StackItem {
                                value: field_value,
                                format_depth: item.format_depth,
                                type_depth: item.type_depth + 1,
                                state: StackState::Finish,
                            };
                            let start_item = StackItem {
                                value: field_value,
                                format_depth: item.format_depth,
                                type_depth: item.type_depth + 1,
                                state: StackState::Start,
                            };

                            stack.push_back(item);
                            stack.push_back(finish_item);
                            stack.push_back(start_item);
                        }
                    } else if let Type::User(UserType::Enum(_def)) = item.value.shape().ty {
                        let enum_val = item.value.into_enum().unwrap();

                        // Get active variant or skip this field processing
                        let variant = match enum_val.active_variant() {
                            Ok(v) => v,
                            Err(_) => {
                                // Skip field processing for this enum
                                continue;
                            }
                        };
                        if field_index >= variant.data.fields.len() {
                            // Determine variant kind to use the right closing delimiter
                            match variant.data.kind {
                                StructKind::Tuple => {
                                    // Close tuple variant with )
                                    write!(
                                        f,
                                        "{:width$}{}",
                                        "",
                                        self.style_punctuation(")"),
                                        width = (item.format_depth - 1) * self.indent_size
                                    )?;
                                }
                                StructKind::Struct => {
                                    // Close struct variant with }
                                    write!(
                                        f,
                                        "{:width$}{}",
                                        "",
                                        self.style_punctuation("}"),
                                        width = (item.format_depth - 1) * self.indent_size
                                    )?;
                                }
                                _ => {}
                            }
                            continue;
                        }

                        let field = variant.data.fields[field_index];

                        // Get field value or skip this field
                        let field_value = match enum_val.field(field_index) {
                            Ok(Some(v)) => v,
                            _ => {
                                // Can't get the field value, skip this field
                                item.state = StackState::ProcessStructField {
                                    field_index: field_index + 1,
                                };
                                stack.push_back(item);
                                continue;
                            }
                        };

                        // Add field doc comments if available
                        // Only add new line if not the first field
                        write!(
                            f,
                            "{:width$}",
                            "",
                            width = item.format_depth * self.indent_size
                        )?;

                        if !field.doc.is_empty() {
                            for line in field.doc {
                                self.write_comment(f, &format!("///{}", line))?;
                                write!(
                                    f,
                                    "\n{:width$}",
                                    "",
                                    width = item.format_depth * self.indent_size
                                )?;
                            }
                        }

                        // For struct variants, print field name
                        if let StructKind::Struct = variant.data.kind {
                            self.write_field_name(f, field.name)?;
                            self.write_punctuation(f, ": ")?;
                        }

                        // Set up to process the next field after this one
                        item.state = StackState::ProcessStructField {
                            field_index: field_index + 1,
                        };

                        // Create finish and start items for processing the field value
                        let finish_item = StackItem {
                            value: field_value,
                            format_depth: item.format_depth,
                            type_depth: item.type_depth + 1,
                            state: StackState::Finish,
                        };
                        let start_item = StackItem {
                            value: field_value,
                            format_depth: item.format_depth,
                            type_depth: item.type_depth + 1,
                            state: StackState::Start,
                        };

                        // Push items to stack in the right order
                        stack.push_back(item);
                        stack.push_back(finish_item);
                        stack.push_back(start_item);
                    }
                }
                StackState::ProcessSeqItem { item_index, kind } => {
                    let (len, elem) = match kind {
                        SeqKind::List => {
                            let list = item.value.into_list_like().unwrap();
                            (list.len(), list.get(item_index))
                        }
                        SeqKind::Tuple => {
                            let tuple = item.value.into_tuple().unwrap();
                            (tuple.len(), tuple.field(item_index))
                        }
                    };
                    if item_index >= len {
                        // All items processed, write closing bracket
                        write!(
                            f,
                            "{:width$}",
                            "",
                            width = (item.format_depth - 1) * self.indent_size
                        )?;
                        self.write_punctuation(
                            f,
                            match kind {
                                SeqKind::List => "]",
                                SeqKind::Tuple => ")",
                            },
                        )?;
                        continue;
                    }

                    // Indent
                    write!(
                        f,
                        "{:width$}",
                        "",
                        width = item.format_depth * self.indent_size
                    )?;

                    // Push back current item to continue after formatting list item
                    item.state = StackState::ProcessSeqItem {
                        item_index: item_index + 1,
                        kind,
                    };
                    let next_format_depth = item.format_depth;
                    let next_type_depth = item.type_depth + 1;
                    stack.push_back(item);

                    let elem = elem.unwrap();

                    // Push list item to format first
                    stack.push_back(StackItem {
                        value: elem,
                        format_depth: next_format_depth,
                        type_depth: next_type_depth,
                        state: StackState::Finish,
                    });

                    // When we push a list item to format, we need to process it from the beginning
                    stack.push_back(StackItem {
                        value: elem,
                        format_depth: next_format_depth,
                        type_depth: next_type_depth,
                        state: StackState::Start, // Use Start state to properly process the item
                    });
                }
                StackState::ProcessBytesItem { item_index } => {
                    let list = item.value.into_list().unwrap();
                    if item_index >= list.len() {
                        // All items processed, write closing bracket
                        write!(
                            f,
                            "{:width$}",
                            "",
                            width = (item.format_depth - 1) * self.indent_size
                        )?;
                        continue;
                    }

                    // On the first byte, write the opening byte sequence indicator
                    if item_index == 0 {
                        write!(f, " ")?;
                    }

                    // Only display 16 bytes per line
                    if item_index > 0 && item_index % 16 == 0 {
                        writeln!(f)?;
                        write!(
                            f,
                            "{:width$}",
                            "",
                            width = item.format_depth * self.indent_size
                        )?;
                    } else if item_index > 0 {
                        write!(f, " ")?;
                    }

                    // Get the byte
                    let byte_value = list.get(item_index).unwrap();
                    // Get the byte value as u8
                    let byte = byte_value.get::<u8>().unwrap_or(&0);

                    // Generate a color for this byte based on its value
                    let mut hasher = DefaultHasher::new();
                    byte.hash(&mut hasher);
                    let hash = hasher.finish();
                    let color = self.color_generator.generate_color(hash);

                    // Apply color if needed
                    if self.use_colors {
                        write!(f, "\x1b[38;2;{};{};{}m", color.r, color.g, color.b)?;
                    }

                    // Display the byte in hex format
                    write!(f, "{:02x}", *byte)?;

                    // Reset color if needed
                    // Reset color already handled by stylize

                    // Push back current item to continue after formatting byte
                    item.state = StackState::ProcessBytesItem {
                        item_index: item_index + 1,
                    };
                    stack.push_back(item);
                }
                StackState::ProcessMapEntry => {
                    // TODO: Implement proper map iteration when available in facet

                    // Indent
                    write!(
                        f,
                        "{:width$}",
                        "",
                        width = item.format_depth * self.indent_size
                    )?;
                    write!(f, "{}", self.style_comment("/* Map contents */"))?;
                    writeln!(f)?;

                    // Closing brace with proper indentation
                    write!(
                        f,
                        "{:width$}{}",
                        "",
                        self.style_punctuation("}"),
                        width = (item.format_depth - 1) * self.indent_size
                    )?;
                }
                StackState::Finish => {
                    // Add comma and newline for struct fields and list items
                    self.write_punctuation(f, ",")?;
                    writeln!(f)?;
                }
                StackState::OptionFinish => {
                    // Just close the Option::Some parenthesis, with no comma
                    self.write_punctuation(f, ")")?;
                }
            }
        }

        Ok(())
    }

    fn handle_list<'mem, 'facet, 'shape>(
        &self,
        stack: &mut VecDeque<StackItem<'mem, 'facet, 'shape>>,
        mut item: StackItem<'mem, 'facet, 'shape>,
        f: &mut impl Write,
    ) -> fmt::Result {
        let list = item.value.into_list_like().unwrap();

        // When recursing into a list, always increment format_depth
        // Only increment type_depth if we're moving to a different address
        let new_type_depth =
            // Incrementing type_depth for all list operations
            item.type_depth + 1; // Always increment type_depth for list operations

        // Print the list name
        self.write_type_name(f, &item.value)?;

        if list.def().t().is_type::<u8>() && self.list_u8_as_bytes {
            // Push back the item with the next state to continue processing list items
            item.state = StackState::ProcessBytesItem { item_index: 0 };
            writeln!(f)?;
            write!(f, " ")?;

            // TODO: write all the bytes here instead?
        } else {
            // Push back the item with the next state to continue processing list items
            item.state = StackState::ProcessSeqItem {
                item_index: 0,
                kind: SeqKind::List,
            };
            self.write_punctuation(f, " [")?;
            writeln!(f)?;
        }

        item.format_depth += 1;
        item.type_depth = new_type_depth;
        stack.push_back(item);

        Ok(())
    }

    /// Format a scalar value
    fn format_scalar(&self, value: Peek, f: &mut impl Write) -> fmt::Result {
        // Generate a color for this shape
        let mut hasher = DefaultHasher::new();
        value.shape().id.hash(&mut hasher);
        let hash = hasher.finish();
        let color = self.color_generator.generate_color(hash);

        // Display the value
        struct DisplayWrapper<'mem, 'facet, 'shape>(&'mem Peek<'mem, 'facet, 'shape>);

        impl fmt::Display for DisplayWrapper<'_, '_, '_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.0.shape().is_display() {
                    write!(f, "{}", self.0)?;
                } else if self.0.shape().is_debug() {
                    write!(f, "{:?}", self.0)?;
                } else {
                    write!(f, "{}", self.0.shape())?;
                    write!(f, "(⋯)")?;
                }
                Ok(())
            }
        }

        // Apply color if needed and display
        if self.use_colors {
            // We need to use direct ANSI codes for RGB colors
            write!(
                f,
                "\x1b[38;2;{};{};{}m{}",
                color.r,
                color.g,
                color.b,
                DisplayWrapper(&value)
            )?;
            write!(f, "\x1b[0m")?;
        } else {
            write!(f, "{}", DisplayWrapper(&value))?;
        }

        Ok(())
    }

    /// Write styled type name to formatter
    fn write_type_name<W: fmt::Write>(&self, f: &mut W, peek: &Peek) -> fmt::Result {
        struct TypeNameWriter<'mem, 'facet, 'shape>(&'mem Peek<'mem, 'facet, 'shape>);

        impl core::fmt::Display for TypeNameWriter<'_, '_, '_> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                self.0.type_name(f, TypeNameOpts::infinite())
            }
        }
        let type_name = TypeNameWriter(peek);

        if self.use_colors {
            write!(f, "\x1b[1m{}\x1b[0m", type_name) // bold
        } else {
            write!(f, "{}", type_name)
        }
    }

    /// Style a type name and return it as a string
    #[allow(dead_code)]
    fn style_type_name(&self, peek: &Peek) -> String {
        let mut result = String::new();
        self.write_type_name(&mut result, peek).unwrap();
        result
    }

    /// Write styled field name to formatter
    fn write_field_name<W: fmt::Write>(&self, f: &mut W, name: &str) -> fmt::Result {
        if self.use_colors {
            // Use cyan color for field names (approximating original RGB color)
            write!(f, "\x1b[36m{}\x1b[0m", name) // cyan
        } else {
            write!(f, "{}", name)
        }
    }

    /// Write styled punctuation to formatter
    fn write_punctuation<W: fmt::Write>(&self, f: &mut W, text: &str) -> fmt::Result {
        if self.use_colors {
            write!(f, "\x1b[2m{}\x1b[0m", text) // dim
        } else {
            write!(f, "{}", text)
        }
    }

    /// Style punctuation and return it as a string
    fn style_punctuation(&self, text: &str) -> String {
        let mut result = String::new();
        self.write_punctuation(&mut result, text).unwrap();
        result
    }

    /// Write styled comment to formatter
    fn write_comment<W: fmt::Write>(&self, f: &mut W, text: &str) -> fmt::Result {
        if self.use_colors {
            write!(f, "\x1b[2m{}\x1b[0m", text) // dim
        } else {
            write!(f, "{}", text)
        }
    }

    /// Style a comment and return it as a string
    fn style_comment(&self, text: &str) -> String {
        let mut result = String::new();
        self.write_comment(&mut result, text).unwrap();
        result
    }

    /// Write styled redacted value to formatter
    fn write_redacted<W: fmt::Write>(&self, f: &mut W, text: &str) -> fmt::Result {
        if self.use_colors {
            // Use bright red and bold for redacted values
            if self.use_colors {
                write!(f, "\x1b[91;1m{}\x1b[0m", text) // bright red + bold
            } else {
                write!(f, "{}", text)
            }
        } else {
            write!(f, "{}", text)
        }
    }

    /// Style a redacted value and return it as a string
    #[allow(dead_code)]
    fn style_redacted(&self, text: &str) -> String {
        let mut result = String::new();
        self.write_redacted(&mut result, text).unwrap();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic tests for the PrettyPrinter
    #[test]
    fn test_pretty_printer_default() {
        let printer = PrettyPrinter::default();
        assert_eq!(printer.indent_size, 2);
        assert_eq!(printer.max_depth, None);
        assert!(printer.use_colors);
    }

    #[test]
    fn test_pretty_printer_with_methods() {
        let printer = PrettyPrinter::new()
            .with_indent_size(4)
            .with_max_depth(3)
            .with_colors(false);

        assert_eq!(printer.indent_size, 4);
        assert_eq!(printer.max_depth, Some(3));
        assert!(!printer.use_colors);
    }
}
