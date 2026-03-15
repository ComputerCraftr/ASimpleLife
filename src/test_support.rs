use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CfgPredicate {
    Flag(String),
    KeyValue { key: String, value: String },
    All(Vec<Self>),
    Any(Vec<Self>),
    Not(Box<Self>),
}

pub(crate) struct RustSource {
    syntax: syn::File,
}

impl RustSource {
    pub(crate) fn parse(source: &str) -> Result<Self, syn::Error> {
        let normalized = source.replace("\r\n", "\n");
        syn::parse_file(&normalized).map(|syntax| Self { syntax })
    }

    pub(crate) fn function_has_cfg_flag(&self, function: &str, flag: &str) -> Option<bool> {
        let function = self.function(function)?;
        Some(function.attrs.iter().any(|attribute| {
            if !attribute.path().is_ident("cfg") {
                return false;
            }
            let mut contains_flag = false;
            attribute
                .parse_nested_meta(|meta| {
                    contains_flag |= meta.path.is_ident(flag);
                    Ok(())
                })
                .is_ok()
                && contains_flag
        }))
    }

    pub(crate) fn function_path_references(&self, function: &str) -> Option<Vec<String>> {
        let function = self.function(function)?;
        let mut visitor = PathReferenceVisitor::default();
        visitor.visit_item_fn(function);
        Some(visitor.references)
    }

    pub(crate) fn struct_field_names(&self, name: &str) -> Option<Vec<String>> {
        self.syntax.items.iter().find_map(|item| {
            let syn::Item::Struct(item_struct) = item else {
                return None;
            };
            if item_struct.ident != name {
                return None;
            }
            Some(
                item_struct
                    .fields
                    .iter()
                    .filter_map(|field| field.ident.as_ref().map(ToString::to_string))
                    .collect(),
            )
        })
    }

    pub(crate) fn method_call_count(&self, method: &str) -> usize {
        let mut visitor = MethodCallVisitor { method, count: 0 };
        visitor.visit_file(&self.syntax);
        visitor.count
    }

    pub(crate) fn macro_cfg_predicate(&self, macro_name: &str) -> Option<CfgPredicate> {
        self.syntax.items.iter().find_map(|item| {
            let syn::Item::Macro(item_macro) = item else {
                return None;
            };
            if !item_macro.mac.path.is_ident(macro_name) {
                return None;
            }
            item_macro.attrs.iter().find_map(|attribute| {
                if !attribute.path().is_ident("cfg") {
                    return None;
                }
                let syn::Meta::List(list) = &attribute.meta else {
                    return None;
                };
                let nested = list
                    .parse_args_with(
                        syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                    )
                    .ok()?;
                (nested.len() == 1)
                    .then(|| nested.first().and_then(cfg_predicate))
                    .flatten()
            })
        })
    }

    fn contains_callable(&self, name: &str) -> bool {
        self.syntax.items.iter().any(|item| match item {
            syn::Item::Fn(function) => function.sig.ident == name,
            syn::Item::Impl(item_impl) => item_impl.items.iter().any(
                |item| matches!(item, syn::ImplItem::Fn(function) if function.sig.ident == name),
            ),
            _ => false,
        })
    }

    fn contains_macro(&self, name: &str) -> bool {
        self.syntax.items.iter().any(
            |item| matches!(item, syn::Item::Macro(item_macro) if item_macro.mac.path.is_ident(name)),
        )
    }

    fn function(&self, name: &str) -> Option<&syn::ItemFn> {
        self.syntax.items.iter().find_map(|item| {
            let syn::Item::Fn(function) = item else {
                return None;
            };
            (function.sig.ident == name).then_some(function)
        })
    }
}

pub(crate) struct RustRepository {
    sources: Vec<(PathBuf, RustSource)>,
}

impl RustRepository {
    pub(crate) fn discover(root: &Path) -> Result<Self, String> {
        let mut paths = Vec::new();
        let mut directories = vec![root.join("src")];
        while let Some(directory) = directories.pop() {
            let entries = fs::read_dir(&directory)
                .map_err(|error| format!("failed to read {}: {error}", directory.display()))?;
            for entry in entries {
                let entry = entry.map_err(|error| {
                    format!(
                        "failed to read an entry in {}: {error}",
                        directory.display()
                    )
                })?;
                let path = entry.path();
                if path.is_dir() {
                    directories.push(path);
                } else if path.extension().is_some_and(|extension| extension == "rs") {
                    paths.push(path);
                }
            }
        }
        paths.sort();

        let mut sources = Vec::with_capacity(paths.len());
        for path in paths {
            let source = fs::read_to_string(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            let syntax = RustSource::parse(&source)
                .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
            sources.push((path, syntax));
        }
        Ok(Self { sources })
    }

    pub(crate) fn source_containing_callable(&self, name: &str) -> Option<&RustSource> {
        unique_source(
            self.sources
                .iter()
                .filter(|(_, source)| source.contains_callable(name)),
        )
    }

    pub(crate) fn source_containing_macro(&self, name: &str) -> Option<&RustSource> {
        unique_source(
            self.sources
                .iter()
                .filter(|(_, source)| source.contains_macro(name)),
        )
    }
}

fn unique_source<'a>(
    mut matches: impl Iterator<Item = &'a (PathBuf, RustSource)>,
) -> Option<&'a RustSource> {
    let (_, source) = matches.next()?;
    matches.next().is_none().then_some(source)
}

fn cfg_predicate(meta: &syn::Meta) -> Option<CfgPredicate> {
    match meta {
        syn::Meta::Path(path) => Some(CfgPredicate::Flag(path.get_ident()?.to_string())),
        syn::Meta::NameValue(name_value) => {
            let key = name_value.path.get_ident()?.to_string();
            let syn::Expr::Lit(expression) = &name_value.value else {
                return None;
            };
            let syn::Lit::Str(value) = &expression.lit else {
                return None;
            };
            Some(CfgPredicate::KeyValue {
                key,
                value: value.value(),
            })
        }
        syn::Meta::List(list) => {
            let operator = list.path.get_ident()?.to_string();
            let nested = list
                .parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                )
                .ok()?
                .iter()
                .map(cfg_predicate)
                .collect::<Option<Vec<_>>>()?;
            match operator.as_str() {
                "all" => Some(CfgPredicate::All(nested)),
                "any" => Some(CfgPredicate::Any(nested)),
                "not" if nested.len() == 1 => nested
                    .into_iter()
                    .next()
                    .map(Box::new)
                    .map(CfgPredicate::Not),
                _ => None,
            }
        }
    }
}

#[derive(Default)]
struct PathReferenceVisitor {
    references: Vec<String>,
}

impl<'ast> Visit<'ast> for PathReferenceVisitor {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.references.extend(
            path.segments
                .iter()
                .map(|segment| segment.ident.to_string()),
        );
        visit::visit_path(self, path);
    }
}

struct MethodCallVisitor<'a> {
    method: &'a str,
    count: usize,
}

impl<'ast> Visit<'ast> for MethodCallVisitor<'_> {
    fn visit_expr_method_call(&mut self, expression: &'ast syn::ExprMethodCall) {
        if expression.method == self.method {
            self.count += 1;
        }
        visit::visit_expr_method_call(self, expression);
    }
}

#[cfg(test)]
mod tests {
    use super::{CfgPredicate, RustSource};
    use crate::RequiredExt;

    #[test]
    fn rust_source_queries_use_syntax_instead_of_comments_or_string_literals() {
        let source = RustSource::parse(
            r#"
                // fake.live_cells(); compile_life_scaffold();
                const TEXT: &str = "fake.live_cells(); ReferenceLifeScaffold";

                fn run() {
                    grid.occupied_chunks();
                    compile_to_life_circuit();
                }

                #[cfg(test)]
                fn compile_life_scaffold() {}

                struct CompiledLifeProgram {
                    initial_grid: Grid,
                    cell_bits: u8,
                }
            "#,
        )
        .or_invariant("source-query fixture should parse");

        assert_eq!(source.method_call_count("live_cells"), 0);
        assert_eq!(source.method_call_count("occupied_chunks"), 1);
        assert_eq!(
            source.function_has_cfg_flag("compile_life_scaffold", "test"),
            Some(true)
        );
        assert_eq!(
            source
                .function_path_references("run")
                .or_invariant("run function"),
            ["grid", "compile_to_life_circuit"]
        );
        assert_eq!(
            source
                .struct_field_names("CompiledLifeProgram")
                .or_invariant("fixture struct"),
            ["initial_grid", "cell_bits"]
        );
    }

    #[test]
    fn rust_source_queries_accept_windows_line_endings() {
        let source = "#[cfg(test)]\r\nfn scaffold() { value.real_call(); }\r\n";
        let source = RustSource::parse(source).or_invariant("CRLF source should parse");

        assert_eq!(source.function_has_cfg_flag("scaffold", "test"), Some(true));
        assert_eq!(source.method_call_count("real_call"), 1);
    }

    #[test]
    fn rust_source_queries_parse_cfg_predicates_structurally() {
        let source = RustSource::parse(
            r#"
                #[cfg(all(not(doc), any(target_env = "musl", target_env = "msvc")))]
                compile_error!("static CRT required");
            "#,
        )
        .or_invariant("cfg fixture should parse");

        assert_eq!(
            source.macro_cfg_predicate("compile_error"),
            Some(CfgPredicate::All(vec![
                CfgPredicate::Not(Box::new(CfgPredicate::Flag("doc".to_string()))),
                CfgPredicate::Any(vec![
                    CfgPredicate::KeyValue {
                        key: "target_env".to_string(),
                        value: "musl".to_string(),
                    },
                    CfgPredicate::KeyValue {
                        key: "target_env".to_string(),
                        value: "msvc".to_string(),
                    },
                ]),
            ]))
        );
    }
}
