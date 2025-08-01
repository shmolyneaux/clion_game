use alloc::string::String;

use ulid::Ulid;

use crate::{
    Def, Facet, ParseError, PtrConst, PtrMut, PtrUninit, ScalarAffinity, ScalarDef, Shape,
    TryFromError, TryIntoInnerError, Type, UserType, ValueVTable, value_vtable,
};

unsafe impl Facet<'_> for Ulid {
    const VTABLE: &'static ValueVTable = &const {
        // Functions to transparently convert between Ulid and String
        unsafe fn try_from<'shape, 'dst>(
            src_ptr: PtrConst<'_>,
            src_shape: &'shape Shape<'shape>,
            dst: PtrUninit<'dst>,
        ) -> Result<PtrMut<'dst>, TryFromError<'shape>> {
            if src_shape.id != <String as Facet>::SHAPE.id {
                return Err(TryFromError::UnsupportedSourceShape {
                    src_shape,
                    expected: &[<String as Facet>::SHAPE],
                });
            }
            let s = unsafe { src_ptr.read::<String>() };
            match Ulid::from_string(&s) {
                Ok(ulid) => Ok(unsafe { dst.put(ulid) }),
                Err(_) => Err(TryFromError::UnsupportedSourceShape {
                    src_shape,
                    expected: &[<String as Facet>::SHAPE],
                }),
            }
        }

        unsafe fn try_into_inner<'dst>(
            src_ptr: PtrMut<'_>,
            dst: PtrUninit<'dst>,
        ) -> Result<PtrMut<'dst>, TryIntoInnerError> {
            let ulid = unsafe { src_ptr.read::<Ulid>() };
            Ok(unsafe { dst.put(ulid.to_string()) })
        }

        let mut vtable = value_vtable!(Ulid, |f, _opts| write!(
            f,
            "{}",
            Self::SHAPE.type_identifier
        ));
        {
            let vtable = vtable.sized_mut().unwrap();
            vtable.parse = || {
                Some(|s, target| match Ulid::from_string(s) {
                    Ok(ulid) => Ok(unsafe { target.put(ulid) }),
                    Err(_) => Err(ParseError::Generic("ULID parsing failed")),
                })
            };
            vtable.try_from = || Some(try_from);
            vtable.try_into_inner = || Some(try_into_inner);
        }
        vtable
    };

    const SHAPE: &'static Shape<'static> = &const {
        // Return the Shape of the inner type (String)
        fn inner_shape() -> &'static Shape<'static> {
            <String as Facet>::SHAPE
        }

        Shape::builder_for_sized::<Self>()
            .type_identifier("Ulid")
            .ty(Type::User(UserType::Opaque))
            .def(Def::Scalar(
                ScalarDef::builder()
                    .affinity(&const { ScalarAffinity::ulid().build() })
                    .build(),
            ))
            .inner(inner_shape)
            .build()
    };
}
