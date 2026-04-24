use facet::*;
use facet_reflect::*;
use std::borrow::Cow;
use std::ffi::CString;
use std::fmt;
use std::fmt::Formatter;
use std::os::raw::c_int;

pub type ImGuiWindowFlags = c_int;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ImVec2 {
    pub x: f32,
    pub y: f32,
}

pub fn im_col32(r: u32, g: u32, b: u32, a: u32) -> u32 {
    (a << 24) | (b << 16) | (g << 8) | r
}

#[cfg(not(test))]
unsafe extern "C" {
    pub fn igBegin(
        name: *const core::ffi::c_char,
        p_open: *mut bool,
        flags: ImGuiWindowFlags,
    ) -> bool;
    pub fn igEnd();
    pub fn igBeginDisabled();
    pub fn igEndDisabled();
    pub fn igIsItemHovered(flags: i32) -> bool;
    pub fn igSetTooltip(text: *const core::ffi::c_char);
    pub fn igInputText(
        label: *const core::ffi::c_char,
        buffer: *mut core::ffi::c_char,
        buffer_size: i32,
        flags: i32,
    ) -> bool;
    fn igTextC(fmt: *const core::ffi::c_char, ...);
    fn igTextColoredC(r: f32, g: f32, b: f32, a: f32, fmt: *const core::ffi::c_char, ...);
    pub fn igTextColoredBC(
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        br: f32,
        bg: f32,
        bb: f32,
        ba: f32,
        text: *const core::ffi::c_char,
    );
    pub fn igRemoveSpacingH();
    pub fn igButton(label: *const core::ffi::c_char) -> bool;
    pub fn igSliderFloat(
        label: *const core::ffi::c_char,
        v: *mut f32,
        v_min: f32,
        v_max: f32,
        format: *const core::ffi::c_char,
    );
    pub fn igCheckbox(label: *const core::ffi::c_char, value: *mut bool) -> bool;
    pub fn igWantCaptureKeyboard() -> bool;
    pub fn igWantCaptureMouse() -> bool;
    pub fn igTreeNode(label: *const core::ffi::c_char) -> bool;
    pub fn igTreePop();
    pub fn igSHMNextItemOpenOnce();
    pub fn igSameLine();
    pub fn igSetKeyboardFocusHere();
    pub fn igSeparator();
    pub fn shmConsoleFooterHeight() -> f32;
    pub fn igBeginTable(label: *const core::ffi::c_char, columns: i32) -> bool;
    pub fn igTableSetupColumn(label: *const core::ffi::c_char);
    pub fn igTableHeadersRow();
    pub fn igTableNextRow();
    pub fn igTableSetColumnIndex(index: i32);
    pub fn igEndTable();
    pub fn igFrameRate() -> f32;
    pub fn igGetCursorScreenPos(pOut: *mut ImVec2);
    pub fn igDrawRectFilled(min: ImVec2, max: ImVec2, col: u32);
    pub fn igDummy(size: ImVec2);
    pub fn igBeginTooltip();
    pub fn igEndTooltip();
    pub fn igGetMousePos(pOut: *mut ImVec2);
    pub fn igIsMouseHoveringRect(min: ImVec2, max: ImVec2, clip: bool) -> bool;
}

pub fn igText(fmt: *const core::ffi::c_char) {
    #[cfg(not(test))]
    unsafe {
        igTextC(fmt);
    }
}

pub fn igTextColored(r: f32, g: f32, b: f32, a: f32, fmt: *const core::ffi::c_char) {
    #[cfg(not(test))]
    unsafe {
        igTextColoredC(r, g, b, a, fmt);
    }
}

#[cfg(test)]
mod test_mocks {
    use super::ImGuiWindowFlags;
    use super::ImVec2;

    pub fn igBegin(
        name: *const core::ffi::c_char,
        p_open: *mut bool,
        flags: ImGuiWindowFlags,
    ) -> bool {
        panic!("Can't call igBegin in test context")
    }
    pub fn igEnd() {
        panic!("Can't call igEnd in test context")
    }
    pub fn igTextColoredBC(
        r: f32,
        g: f32,
        b: f32,
        a: f32,
        br: f32,
        bg: f32,
        bb: f32,
        ba: f32,
        text: *const core::ffi::c_char,
    ) {
        panic!("Can't call igTextColoredBC in test context")
    }
    pub fn igRemoveSpacingH() {}
    pub fn igBeginDisabled() {
        panic!("Can't call igBeginDisabled in test context")
    }
    pub fn igEndDisabled() {
        panic!("Can't call igEndDisabled in test context")
    }
    pub fn igIsItemHovered(flags: i32) -> bool {
        panic!("Can't call igIsItemHovered in test context")
    }
    pub fn igSetTooltip(text: *const core::ffi::c_char) {
        panic!("Can't call igSetTooltip in test context")
    }
    pub fn igInputText(
        label: *const core::ffi::c_char,
        buffer: *mut core::ffi::c_char,
        buffer_size: i32,
        flags: i32,
    ) -> bool {
        panic!("Can't call igInputText in test context")
    }
    pub fn igButton(label: *const core::ffi::c_char) -> bool {
        panic!("Can't call igButton in test context")
    }
    pub fn igSliderFloat(
        label: *const core::ffi::c_char,
        v: *mut f32,
        v_min: f32,
        v_max: f32,
        format: *const core::ffi::c_char,
    ) {
        panic!("Can't call igSliderFloat in test context")
    }
    pub fn igCheckbox(label: *const core::ffi::c_char, value: *mut bool) -> bool {
        panic!("Can't call igCheckbox in test context")
    }
    pub fn igWantCaptureKeyboard() -> bool {
        panic!("Can't call igWantCaptureKeyboard in test context")
    }
    pub fn igWantCaptureMouse() -> bool {
        panic!("Can't call igWantCaptureMouse in test context")
    }
    pub fn igTreeNode(label: *const core::ffi::c_char) -> bool {
        panic!("Can't call igTreeNode in test context")
    }
    pub fn igTreePop() {
        panic!("Can't call igTreePop in test context")
    }
    pub fn igSHMNextItemOpenOnce() {
        panic!("Can't call igSHMNextItemOpenOnce in test context")
    }
    pub fn igSameLine() {
        panic!("Can't call igSameLine in test context")
    }
    pub fn igSetKeyboardFocusHere() {
        panic!("Can't call igSetKeyboardFocusHere in test context")
    }
    pub fn igSeparator() {
        panic!("Can't call igSeparator in test context")
    }
    pub fn shmConsoleFooterHeight() -> f32 {
        panic!("Can't call shmConsoleFooterHeight in test context")
    }
    pub fn igBeginTable(label: *const core::ffi::c_char, columns: i32) -> bool {
        panic!("Can't call igBeginTable in test context")
    }
    pub fn igTableSetupColumn(label: *const core::ffi::c_char) {
        panic!("Can't call igTableSetupColumn in test context")
    }
    pub fn igTableHeadersRow() {
        panic!("Can't call igTableHeadersRow in test context")
    }
    pub fn igTableNextRow() {
        panic!("Can't call igTableNextRow in test context")
    }
    pub fn igTableSetColumnIndex(index: i32) {
        panic!("Can't call igTableSetColumnIndex in test context")
    }
    pub fn igEndTable() {
        panic!("Can't call igEndTable in test context")
    }
    pub fn igFrameRate() -> f32 {
        panic!("Can't call igFrameRate in test context")
    }
    pub fn igGetCursorScreenPos(pOut: *mut ImVec2) {
        panic!("Can't call igGetCursorScreenPos in test context")
    }
    pub fn igDrawRectFilled(min: ImVec2, max: ImVec2, col: u32) {
        panic!("Can't call igDrawRectFilled in test context")
    }
    pub fn igDummy(size: ImVec2) {
        panic!("Can't call igDummy in test context")
    }
    pub fn igBeginTooltip() {
        panic!("Can't call igBeginTooltip in test context")
    }
    pub fn igEndTooltip() {
        panic!("Can't call igEndTooltip in test context")
    }
    pub fn igGetMousePos(pOut: *mut ImVec2) {
        panic!("Can't call igGetMousePos in test context")
    }
    pub fn igIsMouseHoveringRect(min: ImVec2, max: ImVec2, clip: bool) -> bool {
        panic!("Can't call igIsMouseHoveringRect in test context")
    }
}

#[cfg(test)]
pub use test_mocks::*;

pub static IMGUI_WINDOW_FLAGS_NONE: c_int = 0;
pub static IMGUI_WINDOW_FLAGS_NO_TITLE_BAR: c_int = 1 << 0;
pub static IMGUI_WINDOW_FLAGS_NO_RESIZE: c_int = 1 << 1;
pub static IMGUI_WINDOW_FLAGS_NO_MOVE: c_int = 1 << 2;
pub static IMGUI_WINDOW_FLAGS_NO_SCROLLBAR: c_int = 1 << 3;
pub static IMGUI_WINDOW_FLAGS_NO_SCROLL_WITH_MOUSE: c_int = 1 << 4;
pub static IMGUI_WINDOW_FLAGS_NO_COLLAPSE: c_int = 1 << 5;
pub static IMGUI_WINDOW_FLAGS_ALWAYS_AUTO_RESIZE: c_int = 1 << 6;
pub static IMGUI_WINDOW_FLAGS_NO_BACKGROUND: c_int = 1 << 7;
pub static IMGUI_WINDOW_FLAGS_NO_SAVED_SETTINGS: c_int = 1 << 8;
pub static IMGUI_WINDOW_FLAGS_NO_MOUSE_INPUTS: c_int = 1 << 9;
pub static IMGUI_WINDOW_FLAGS_MENU_BAR: c_int = 1 << 10;
pub static IMGUI_WINDOW_FLAGS_HORIZONTAL_SCROLLBAR: c_int = 1 << 11;
pub static IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING: c_int = 1 << 12;
pub static IMGUI_WINDOW_FLAGS_NO_BRING_TO_FRONT_ON_FOCUS: c_int = 1 << 13;
pub static IMGUI_WINDOW_FLAGS_ALWAYS_VERTICAL_SCROLLBAR: c_int = 1 << 14;
pub static IMGUI_WINDOW_FLAGS_ALWAYS_HORIZONTAL_SCROLLBAR: c_int = 1 << 15;
pub static IMGUI_WINDOW_FLAGS_NO_NAV_INPUTS: c_int = 1 << 16;
pub static IMGUI_WINDOW_FLAGS_NO_NAV_FOCUS: c_int = 1 << 17;
pub static IMGUI_WINDOW_FLAGS_UNSAVED_DOCUMENT: c_int = 1 << 18;
pub static IMGUI_WINDOW_FLAGS_NO_DOCKING: c_int = 1 << 19;

pub static IMGUI_HOVERED_FLAGS_ALLOW_WHEN_DISABLED: c_int = 1 << 10;

pub static IMGUI_INPUT_TEXT_FLAGS_NONE: c_int = 0;
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_DECIMAL: c_int = 1 << 0;
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_HEXADECIMAL: c_int = 1 << 1;
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_SCIENTIFIC: c_int = 1 << 2;
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_UPPERCASE: c_int = 1 << 3;
pub static IMGUI_INPUT_TEXT_FLAGS_CHARS_NOBLANK: c_int = 1 << 4;
pub static IMGUI_INPUT_TEXT_FLAGS_ALLOW_TAB_INPUT: c_int = 1 << 5;
pub static IMGUI_INPUT_TEXT_FLAGS_ENTER_RETURNS_TRUE: c_int = 1 << 6;
pub static IMGUI_INPUT_TEXT_FLAGS_ESCAPE_CLEARS_ALL: c_int = 1 << 7;
pub static IMGUI_INPUT_TEXT_FLAGS_CTRL_ENTER_FOR_NEWLINE: c_int = 1 << 8;
pub static IMGUI_INPUT_TEXT_FLAGS_READONLY: c_int = 1 << 9;
pub static IMGUI_INPUT_TEXT_FLAGS_PASSWORD: c_int = 1 << 10;
pub static IMGUI_INPUT_TEXT_FLAGS_ALWAYSOVERWRITE: c_int = 1 << 11;
pub static IMGUI_INPUT_TEXT_FLAGS_AUTOSELECTALL: c_int = 1 << 12;
pub static IMGUI_INPUT_TEXT_FLAGS_PARSEEMPTYREFVAL: c_int = 1 << 13;
pub static IMGUI_INPUT_TEXT_FLAGS_DISPLAYEMPTYREFVAL: c_int = 1 << 14;
pub static IMGUI_INPUT_TEXT_FLAGS_NOHORIZONTALSCROLL: c_int = 1 << 15;
pub static IMGUI_INPUT_TEXT_FLAGS_NOUNDOREDO: c_int = 1 << 16;
pub static IMGUI_INPUT_TEXT_FLAGS_ELIDELEFT: c_int = 1 << 17;
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKCOMPLETION: c_int = 1 << 18;
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKHISTORY: c_int = 1 << 19;
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKALWAYS: c_int = 1 << 20;
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKCHARFILTER: c_int = 1 << 21;
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKRESIZE: c_int = 1 << 22;
pub static IMGUI_INPUT_TEXT_FLAGS_CALLBACKEDIT: c_int = 1 << 23;

fn parse_facet_range(s: &str) -> Option<(f32, f32)> {
    let prefix = "range = (";

    let rest = s.strip_prefix(prefix)?.trim_start();

    let comma_index = rest.find(',')?;

    let (first_part, second_part_with_paren) = rest.split_at(comma_index);
    let first = first_part.trim().parse::<f32>().ok()?;

    let second_part = second_part_with_paren[1..].trim();

    if !second_part.ends_with(')') {
        return None;
    }

    let second_str = &second_part[..second_part.len() - 1];
    let second = second_str.trim().parse::<f32>().ok()?;

    Some((first, second))
}

struct FormatWrapper {
    func: TypeNameFn,
}

impl fmt::Display for FormatWrapper {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        (self.func)(f, TypeNameOpts::infinite())
    }
}

fn format_shape_typename(shape: &Shape) -> String {
    let mut output = String::new();
    fmt::write(
        &mut output,
        format_args!(
            "{}",
            FormatWrapper {
                func: shape.vtable.type_name()
            }
        ),
    )
    .unwrap();
    output
}

pub fn imgui_debug<'a, T: Facet<'a>>(obj: &mut T) {
    let poke = Peek::new(obj);

    let mut open = true;
    unsafe {
        igBegin(
            c"State".as_ptr(),
            &mut open as *mut bool,
            IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING,
        );
        imgui_debug_inner(&poke, &[], "");
        igEnd();
    }
}

pub unsafe fn imgui_debug_inner(peek: &Peek, attributes: &[FieldAttribute], path: &str) {
    let shape = peek.shape();
    match (peek.scalar_type(), shape.def) {
        (Some(ScalarType::Unit), _) => igText(CString::new(format!("()")).unwrap().as_ptr()),
        (Some(ScalarType::Bool), _) => {
            let mut readonly = false;
            for attr in attributes.iter() {
                let attr_str = match attr {
                    FieldAttribute::Arbitrary(s) => s,
                    _ => continue,
                };
                if *attr_str == "readonly" {
                    readonly = true;
                    break;
                }
            }

            if readonly {
                igText(
                    CString::new(format!("{}", peek.get::<bool>().unwrap()))
                        .unwrap()
                        .as_ptr(),
                );
            } else {
                let ptr = unsafe { peek.data().thin().unwrap().as_ptr::<bool>() as *mut bool };
                igCheckbox(CString::new(path).unwrap().as_ptr(), ptr);
            }
        }
        (Some(ScalarType::Char), _) => igText(
            CString::new(format!("{}", peek.get::<char>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::Str), _) => igText(
            CString::new(format!("{}", peek.get::<&str>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::String), _) => igText(
            CString::new(format!("{}", peek.get::<String>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::CowStr), _) => igText(
            CString::new(format!("{}", peek.get::<Cow<str>>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::F32), _) => {
            let ptr = unsafe { peek.data().thin().unwrap().as_ptr::<f32>() as *mut f32 };
            let mut min = 0.0;
            let mut max = 1.0;

            for attr in attributes.iter() {
                let attr_str = match attr {
                    FieldAttribute::Arbitrary(s) => s,
                    _ => continue,
                };
                if let Some(v) = parse_facet_range(attr_str) {
                    min = v.0;
                    max = v.1;
                }
            }
            igSliderFloat(
                CString::new(path).unwrap().as_ptr(),
                ptr,
                min,
                max,
                c"%.3f".as_ptr(),
            );
        }
        (Some(ScalarType::F64), _) => igText(
            CString::new(format!("{}f64", peek.get::<f64>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::U8), _) => igText(
            CString::new(format!("{}u8", peek.get::<u8>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::U16), _) => igText(
            CString::new(format!("{}u16", peek.get::<u16>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::U32), _) => igText(
            CString::new(format!("{}u32", peek.get::<u32>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::U64), _) => igText(
            CString::new(format!("{}u64", peek.get::<u64>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::U128), _) => igText(
            CString::new(format!("{}u128", peek.get::<u128>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::USize), _) => igText(
            CString::new(format!("{}usize", peek.get::<usize>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::I8), _) => igText(
            CString::new(format!("{}i8", peek.get::<i8>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::I16), _) => igText(
            CString::new(format!("{}i16", peek.get::<i16>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::I32), _) => {
            let ptr = unsafe { peek.data().thin().unwrap().as_ptr::<i32>() as *mut i32 };
            igText(
                CString::new(format!("{}i32", peek.get::<i32>().unwrap()))
                    .unwrap()
                    .as_ptr(),
            );
        }
        (Some(ScalarType::I64), _) => igText(
            CString::new(format!("{}i64", peek.get::<i64>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::I128), _) => igText(
            CString::new(format!("{}128", peek.get::<i128>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (Some(ScalarType::ISize), _) => igText(
            CString::new(format!("{}isize", peek.get::<isize>().unwrap()))
                .unwrap()
                .as_ptr(),
        ),
        (_, Def::Scalar(def)) => igText(CString::new(format!("Def::Scalar")).unwrap().as_ptr()),
        (_, Def::List(def)) => {
            let shape_typename = format_shape_typename(peek.shape());
            let peek = peek.into_list_like().unwrap();
            if peek.len() == 0 {
                igText(CString::new(format!("[]")).unwrap().as_ptr());
            } else {
                if igTreeNode(
                    CString::new(format!("{}##{}", shape_typename, path,))
                        .unwrap()
                        .as_ptr(),
                ) {
                    for (idx, item) in peek.iter().enumerate() {
                        imgui_debug_inner(&item, &[], &format!("{}[{}]", path, idx));
                    }
                    igTreePop();
                }
            }
        }
        (_, Def::Map(def)) => igText(CString::new(format!("Def::Map")).unwrap().as_ptr()),
        (_, Def::Set(def)) => igText(CString::new(format!("Def::Set")).unwrap().as_ptr()),
        (_, Def::Array(def)) => {
            let shape_typename = format_shape_typename(peek.shape());
            let peek = peek.into_list_like().unwrap();
            if peek.len() == 0 {
                igText(CString::new(format!("[]")).unwrap().as_ptr());
            } else {
                if igTreeNode(
                    CString::new(format!("{}##{}", shape_typename, path,))
                        .unwrap()
                        .as_ptr(),
                ) {
                    for (idx, item) in peek.iter().enumerate() {
                        imgui_debug_inner(&item, &[], &format!("{}[{}]", path, idx));
                    }
                    igTreePop();
                }
            }
        }
        (_, Def::Slice(def)) => igText(CString::new(format!("Def::Slice")).unwrap().as_ptr()),
        (_, Def::Option(def)) => igText(CString::new(format!("Def::Option")).unwrap().as_ptr()),
        (_, Def::SmartPointer(def)) => {
            igText(CString::new(format!("Def::SmartPointer")).unwrap().as_ptr())
        }
        (_, Def::Undefined) => {
            let ty = shape.ty;
            match ty {
                Type::Primitive(ty) => {
                    igText(CString::new(format!("Type::Primitive")).unwrap().as_ptr())
                }
                Type::Sequence(ty) => {
                    igText(CString::new(format!("Type::Sequence")).unwrap().as_ptr())
                }
                Type::User(UserType::Struct(ty)) => {
                    if path == "" {
                        igSHMNextItemOpenOnce();
                    }
                    if igTreeNode(
                        CString::new(format!("{}##{}", format_shape_typename(peek.shape()), path,))
                            .unwrap()
                            .as_ptr(),
                    ) {
                        let peek = peek.into_struct().unwrap();

                        if igBeginTable(CString::new(path).unwrap().as_ptr(), 2) {
                            igTableSetupColumn(c"Attribute".as_ptr());
                            igTableSetupColumn(c"Value".as_ptr());

                            for field_idx in 0..peek.field_count() {
                                igTableNextRow();

                                igTableSetColumnIndex(0);
                                let field = peek.field(field_idx).unwrap();
                                let field_shape = ty.fields[field_idx];
                                let field_name = field_shape.name;
                                igText(CString::new(format!("{field_name}: ")).unwrap().as_ptr());

                                igTableSetColumnIndex(1);
                                imgui_debug_inner(
                                    &field,
                                    &field_shape.attributes,
                                    &format!("{}.{}", path, field_name),
                                );
                            }

                            igEndTable();
                        }
                        igTreePop();
                    }
                }
                Type::User(UserType::Enum(ty)) => {
                    igText(CString::new(format!("UserType::Enum")).unwrap().as_ptr())
                }
                Type::User(UserType::Union(ty)) => {
                    igText(CString::new(format!("UserType::Union")).unwrap().as_ptr())
                }
                Type::User(UserType::Opaque) => {
                    igText(CString::new(format!("UserType::Opaque")).unwrap().as_ptr())
                }
                Type::Pointer(ty) => {
                    igText(CString::new(format!("Type::Pointer")).unwrap().as_ptr())
                }
                _ => igText(
                    CString::new(format!(
                        "Can't debug {}",
                        format_shape_typename(peek.shape())
                    ))
                    .unwrap()
                    .as_ptr(),
                ),
            }
        }
        _ => igText(
            CString::new(format!(
                "Can't debug {}",
                format_shape_typename(peek.shape())
            ))
            .unwrap()
            .as_ptr(),
        ),
    }
}

pub fn draw_log_window() {
    crate::DEBUG_LOG.with_borrow(|logs| unsafe {
        let mut open = true;
        igBegin(
            c"Rust Log Window".as_ptr(),
            &mut open as *mut bool,
            IMGUI_WINDOW_FLAGS_NO_FOCUS_ON_APPEARING,
        );
        for line in logs.iter() {
            igText(line.as_ptr());
        }
        igEnd();
    });
}
