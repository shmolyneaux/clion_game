use crate::*;
use core::arch::x86_64::__m128;

unsafe impl<'a> Facet<'a> for __m128
{
    const VTABLE: &'static ValueVTable = <[f32; 4]>::VTABLE;
    const SHAPE: &'static Shape<'static> = &const {
        Shape::builder_for_sized::<Self>()
            .type_identifier("core::arch::x86_64::__m128")
            .ty(Type::Sequence(SequenceType::Array(ArrayType {
                t: f32::SHAPE,
                n: 4,
            })))
            .def(Def::Array(
                ArrayDef::builder()
                    .vtable(
                        &const {
                            ArrayVTable::builder()
                                .as_ptr(|ptr| unsafe {
                                    let array = ptr.get::<[f32; 4]>();
                                    PtrConst::new(array.as_ptr())
                                })
                                .as_mut_ptr(|ptr| unsafe {
                                    let array = ptr.as_mut::<[f32; 4]>();
                                    PtrMut::new(array.as_mut_ptr())
                                })
                                .build()
                        },
                    )
                    .t(f32::SHAPE)
                    .n(4)
                    .build(),
            ))
            .build()
    };
}
