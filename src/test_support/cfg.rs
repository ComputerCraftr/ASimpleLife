use syn::{Attribute, Meta, Token, punctuated::Punctuated};

/// Prove exclusion when `test` is false; unknown target/feature predicates may
/// take either value. `cfg_attr(test, ...)` alone never excludes production.
pub(crate) fn attributes_are_test_only(attributes: &[Attribute]) -> bool {
    let mut pending: Vec<_> = attributes
        .iter()
        .map(|attribute| attribute.meta.clone())
        .collect();
    while let Some(meta) = pending.pop() {
        let Meta::List(list) = meta else {
            continue;
        };
        if list.path.is_ident("cfg") {
            if list
                .parse_args::<Meta>()
                .is_ok_and(|meta| !possibilities(&meta).0)
            {
                return true;
            }
        } else if list.path.is_ident("cfg_attr") {
            let Ok(children) =
                list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            else {
                continue;
            };
            let mut children = children.into_iter();
            if children
                .next()
                .is_some_and(|condition| possibilities(&condition) == (true, false))
            {
                pending.extend(children);
            }
        }
    }
    false
}

fn possibilities(root: &Meta) -> (bool, bool) {
    let mut pending = vec![(root.clone(), false)];
    let mut values = Vec::new();
    while let Some((meta, reduce)) = pending.pop() {
        match meta {
            Meta::Path(path) if path.is_ident("test") => values.push((false, true)),
            Meta::List(list) => {
                let Ok(children) =
                    list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                else {
                    values.push((true, true));
                    continue;
                };
                if !reduce {
                    pending.push((Meta::List(list), true));
                    pending.extend(children.into_iter().rev().map(|child| (child, false)));
                    continue;
                }
                let begin = values.len() - children.len();
                let children = &values[begin..];
                let value = if list.path.is_ident("all") {
                    (children.iter().all(|v| v.0), children.iter().any(|v| v.1))
                } else if list.path.is_ident("any") {
                    (children.iter().any(|v| v.0), children.iter().all(|v| v.1))
                } else if list.path.is_ident("not") && children.len() == 1 {
                    (children[0].1, children[0].0)
                } else {
                    (true, true)
                };
                values.truncate(begin);
                values.push(value);
            }
            _ => values.push((true, true)),
        }
    }
    values.first().copied().unwrap_or((true, true))
}

#[test]
fn cfg_test_exclusion_is_proven_from_boolean_syntax_not_token_text() {
    use crate::RequiredExt;
    for (attribute, expected) in [
        ("#[cfg(test)]", true),
        ("#[cfg(all(test, feature = \"x\"))]", true),
        ("#[cfg(any(test, unix))]", false),
        ("#[cfg(not(test))]", false),
        ("#[cfg(not(not(test)))]", true),
        ("#[cfg_attr(test, inline)]", false),
        ("#[cfg_attr(test, cfg(test))]", false),
        ("#[cfg_attr(not(test), cfg(test))]", true),
        ("#[cfg_attr(unix, cfg(test))]", false),
    ] {
        let item: syn::ItemFn =
            syn::parse_str(&format!("{attribute} fn f() {{}}")).or_invariant("cfg fixture");
        assert_eq!(
            attributes_are_test_only(&item.attrs),
            expected,
            "{attribute}"
        );
    }
}
