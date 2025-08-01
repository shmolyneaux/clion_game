use crate::{
    Def, Facet, ScalarAffinity, ScalarDef, Shape, Type, UserType, ValueVTable, value_vtable,
};

#[cfg(feature = "alloc")]
unsafe impl Facet<'_> for alloc::string::String {
    const VTABLE: &'static ValueVTable = &const {
        value_vtable!(alloc::string::String, |f, _opts| write!(
            f,
            "{}",
            Self::SHAPE.type_identifier
        ))
    };

    const SHAPE: &'static Shape<'static> = &const {
        Shape::builder_for_sized::<Self>()
            .def(Def::Scalar(
                ScalarDef::builder()
                    // `String` is always on the heap
                    .affinity(&const { ScalarAffinity::string().max_inline_length(0).build() })
                    .build(),
            ))
            .type_identifier("String")
            .ty(Type::User(UserType::Opaque))
            .build()
    };
}

unsafe impl<'a> Facet<'a> for alloc::borrow::Cow<'a, str> {
    const VTABLE: &'static ValueVTable = &const {
        value_vtable!(alloc::borrow::Cow<'_, str>, |f, _opts| write!(
            f,
            "Cow<'_, str>"
        ))
    };

    const SHAPE: &'static Shape<'static> = &const {
        Shape::builder_for_sized::<Self>()
            .def(Def::Scalar(
                ScalarDef::builder()
                    .affinity(&const { ScalarAffinity::string().build() })
                    .build(),
            ))
            .type_identifier("Cow")
            .ty(Type::User(UserType::Opaque))
            .build()
    };
}
