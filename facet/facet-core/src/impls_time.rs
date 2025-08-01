use alloc::string::String;
use time::{OffsetDateTime, UtcDateTime};

use crate::{
    Def, Facet, ParseError, PtrConst, PtrUninit, ScalarAffinity, ScalarDef, Shape, Type, UserType,
    ValueVTable, value_vtable,
};

unsafe impl Facet<'_> for UtcDateTime {
    const VTABLE: &'static ValueVTable = &const {
        let mut vtable = value_vtable!(UtcDateTime, |f, _opts| write!(
            f,
            "{}",
            Self::SHAPE.type_identifier
        ));
        {
            let vtable = vtable.sized_mut().unwrap();
            vtable.try_from = || {
                Some(
                    |source: PtrConst, source_shape: &Shape, target: PtrUninit| {
                        if source_shape.is_type::<String>() {
                            let source = unsafe { source.read::<String>() };
                            let parsed = UtcDateTime::parse(
                                &source,
                                &time::format_description::well_known::Rfc3339,
                            )
                            .map_err(|_| ParseError::Generic("could not parse date"));
                            match parsed {
                                Ok(val) => Ok(unsafe { target.put(val) }),
                                Err(_e) => {
                                    Err(crate::TryFromError::Generic("could not parse date"))
                                }
                            }
                        } else {
                            Err(crate::TryFromError::UnsupportedSourceShape {
                                src_shape: source_shape,
                                expected: &[String::SHAPE],
                            })
                        }
                    },
                )
            };
            vtable.parse = || {
                Some(|s: &str, target: PtrUninit| {
                    let parsed =
                        UtcDateTime::parse(s, &time::format_description::well_known::Rfc3339)
                            .map_err(|_| ParseError::Generic("could not parse date"))?;
                    Ok(unsafe { target.put(parsed) })
                })
            };
            vtable.display = || {
                Some(|value, f| unsafe {
                    let udt = value.get::<UtcDateTime>();
                    match udt.format(&time::format_description::well_known::Rfc3339) {
                        Ok(s) => write!(f, "{s}"),
                        Err(_) => write!(f, "<invalid UtcDateTime>"),
                    }
                })
            };
        }
        vtable
    };

    const SHAPE: &'static Shape<'static> = &const {
        Shape::builder_for_sized::<Self>()
            .type_identifier("UtcDateTime")
            .ty(Type::User(UserType::Opaque))
            .def(Def::Scalar(
                ScalarDef::builder()
                    .affinity(&const { ScalarAffinity::time().build() })
                    .build(),
            ))
            .build()
    };
}

unsafe impl Facet<'_> for OffsetDateTime {
    const VTABLE: &'static ValueVTable = &const {
        let mut vtable = value_vtable!(OffsetDateTime, |f, _opts| write!(
            f,
            "{}",
            Self::SHAPE.type_identifier
        ));
        {
            let vtable = vtable.sized_mut().unwrap();
            vtable.try_from = || {
                Some(
                    |source: PtrConst, source_shape: &Shape, target: PtrUninit| {
                        if source_shape.is_type::<String>() {
                            let source = unsafe { source.read::<String>() };
                            let parsed = OffsetDateTime::parse(
                                &source,
                                &time::format_description::well_known::Rfc3339,
                            )
                            .map_err(|_| ParseError::Generic("could not parse date"));
                            match parsed {
                                Ok(val) => Ok(unsafe { target.put(val) }),
                                Err(_e) => {
                                    Err(crate::TryFromError::Generic("could not parse date"))
                                }
                            }
                        } else {
                            Err(crate::TryFromError::UnsupportedSourceShape {
                                src_shape: source_shape,
                                expected: &[String::SHAPE],
                            })
                        }
                    },
                )
            };
            vtable.parse = || {
                Some(|s: &str, target: PtrUninit| {
                    let parsed =
                        OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
                            .map_err(|_| ParseError::Generic("could not parse date"))?;
                    Ok(unsafe { target.put(parsed) })
                })
            };
            vtable.display = || {
                Some(|value, f| unsafe {
                    let odt = value.get::<OffsetDateTime>();
                    match odt.format(&time::format_description::well_known::Rfc3339) {
                        Ok(s) => write!(f, "{s}"),
                        Err(_) => write!(f, "<invalid OffsetDateTime>"),
                    }
                })
            };
        }
        vtable
    };

    const SHAPE: &'static Shape<'static> = &const {
        Shape::builder_for_sized::<Self>()
            .type_identifier("OffsetDateTime")
            .ty(Type::User(UserType::Opaque))
            .def(Def::Scalar(
                ScalarDef::builder()
                    .affinity(&const { ScalarAffinity::time().build() })
                    .build(),
            ))
            .build()
    };
}

#[cfg(test)]
mod tests {
    use core::fmt;

    use time::OffsetDateTime;

    use crate::{Facet, PtrConst};

    #[test]
    fn parse_offset_date_time() -> eyre::Result<()> {
        facet_testhelpers::setup();

        let target = OffsetDateTime::SHAPE.allocate()?;
        unsafe {
            ((OffsetDateTime::VTABLE.sized().unwrap().parse)().unwrap())(
                "2023-03-14T15:09:26Z",
                target,
            )?;
        }
        let odt: OffsetDateTime = unsafe { target.assume_init().read() };
        assert_eq!(
            odt,
            OffsetDateTime::parse(
                "2023-03-14T15:09:26Z",
                &time::format_description::well_known::Rfc3339
            )
            .unwrap()
        );

        struct DisplayWrapper<'a>(PtrConst<'a>);

        impl fmt::Display for DisplayWrapper<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                unsafe { ((OffsetDateTime::VTABLE.sized().unwrap().display)().unwrap())(self.0, f) }
            }
        }

        let s = format!("{}", DisplayWrapper(PtrConst::new(&odt as *const _)));
        assert_eq!(s, "2023-03-14T15:09:26Z");

        // Deallocate the heap allocation to avoid memory leaks under Miri
        unsafe {
            OffsetDateTime::SHAPE.deallocate_uninit(target)?;
        }

        Ok(())
    }
}
