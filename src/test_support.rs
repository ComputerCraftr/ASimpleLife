use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

pub(crate) mod c;
mod cfg;
pub(crate) mod compiled_c;
mod discovery;
pub(crate) use cfg::attributes_are_test_only;
pub(crate) use discovery::source_files;

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

    pub(crate) fn function_is_test_only(&self, function: &str) -> Option<bool> {
        let (test_only, _) = self.callable(function)?;
        Some(test_only)
    }

    pub(crate) fn function_path_references(&self, function: &str) -> Option<Vec<String>> {
        let (_, block) = self.callable(function)?;
        let mut visitor = PathReferenceVisitor::default();
        visitor.visit_block(block);
        Some(visitor.references)
    }

    pub(crate) fn struct_field_names(&self, name: &str) -> Option<Vec<String>> {
        let matches = self.structs(name);
        if matches.len() != 1 {
            return None;
        }
        Some(
            matches[0]
                .fields
                .iter()
                .filter_map(|field| field.ident.as_ref().map(ToString::to_string))
                .collect(),
        )
    }

    fn structs(&self, name: &str) -> Vec<&syn::ItemStruct> {
        let mut visitor = StructVisitor {
            name,
            found: Vec::new(),
        };
        visitor.visit_file(&self.syntax);
        visitor.found
    }

    pub(crate) fn method_call_count(&self, method: &str) -> usize {
        let mut visitor = MethodCallVisitor { method, count: 0 };
        visitor.visit_file(&self.syntax);
        visitor.count
    }

    pub(crate) fn callable_method_count(&self, callable: &str, method: &str) -> Option<usize> {
        let (_, block) = self.callable(callable)?;
        let mut visitor = MethodCallVisitor { method, count: 0 };
        visitor.visit_block(block);
        Some(visitor.count)
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
        !self.callables(name).is_empty()
    }

    fn contains_macro(&self, name: &str) -> bool {
        self.syntax.items.iter().any(
            |item| matches!(item, syn::Item::Macro(item_macro) if item_macro.mac.path.is_ident(name)),
        )
    }

    fn callable(&self, name: &str) -> Option<(bool, &syn::Block)> {
        let matches = self.callables(name);
        (matches.len() == 1).then(|| matches[0])
    }

    fn callables(&self, name: &str) -> Vec<(bool, &syn::Block)> {
        let mut visitor = CallableVisitor {
            name,
            found: Vec::new(),
            test_only: false,
        };
        visitor.visit_file(&self.syntax);
        visitor.found
    }
}

struct StructVisitor<'name, 'ast> {
    name: &'name str,
    found: Vec<&'ast syn::ItemStruct>,
}

impl<'ast> Visit<'ast> for StructVisitor<'_, 'ast> {
    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        if item.ident == self.name {
            self.found.push(item);
        }
        visit::visit_item_struct(self, item);
    }
}

struct CallableVisitor<'name, 'ast> {
    name: &'name str,
    found: Vec<(bool, &'ast syn::Block)>,
    test_only: bool,
}

impl<'ast> Visit<'ast> for CallableVisitor<'_, 'ast> {
    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        let inherited = self.test_only;
        self.test_only |= attributes_are_test_only(&function.attrs);
        if function.sig.ident == self.name {
            self.found.push((self.test_only, &function.block));
        }
        visit::visit_item_fn(self, function);
        self.test_only = inherited;
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        let inherited = self.test_only;
        self.test_only |= attributes_are_test_only(&function.attrs);
        if function.sig.ident == self.name {
            self.found.push((self.test_only, &function.block));
        }
        visit::visit_impl_item_fn(self, function);
        self.test_only = inherited;
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let inherited = self.test_only;
        self.test_only |= attributes_are_test_only(&item.attrs);
        visit::visit_item_mod(self, item);
        self.test_only = inherited;
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let inherited = self.test_only;
        self.test_only |= attributes_are_test_only(&item.attrs);
        visit::visit_item_impl(self, item);
        self.test_only = inherited;
    }
}

pub(crate) struct RustRepository {
    sources: Vec<(PathBuf, RustSource)>,
}

impl RustRepository {
    pub(crate) fn discover(root: &Path) -> Result<Self, String> {
        let paths = source_files(root, &["rs"])?;

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

    pub(crate) fn source_containing_struct(&self, name: &str) -> Option<&RustSource> {
        unique_source(
            self.sources
                .iter()
                .filter(|(_, source)| !source.structs(name).is_empty()),
        )
    }

    pub(crate) fn source_containing_function_reference(
        &self,
        function: &str,
        referenced_path: &str,
    ) -> Option<&RustSource> {
        unique_source(self.sources.iter().filter(|(_, source)| {
            source
                .function_path_references(function)
                .is_some_and(|references| {
                    references
                        .iter()
                        .any(|reference| reference == referenced_path)
                })
        }))
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
    fn rust_queries_follow_nested_modules_but_keep_callable_scope_and_ambiguity() {
        let source = RustSource::parse(
            r#"
            mod moved {
                struct Artifact { cells: Grid }
                impl Engine { fn embed(&self) { grid.occupied_chunks(); } }
                fn unrelated() { grid.live_cells(); }
            }
        "#,
        )
        .or_invariant("nested AST fixture");
        assert_eq!(source.callable_method_count("embed", "live_cells"), Some(0));
        assert_eq!(
            source.callable_method_count("embed", "occupied_chunks"),
            Some(1)
        );
        assert_eq!(
            source.struct_field_names("Artifact"),
            Some(vec!["cells".to_string()])
        );
        let duplicate = RustSource::parse("mod a { fn f() {} } mod b { fn f() {} }")
            .or_invariant("ambiguous fixture");
        assert!(
            duplicate.function_path_references("f").is_none(),
            "ambiguous names must not silently select the first declaration"
        );
    }

    #[test]
    fn rust_callable_cfg_inherits_module_and_impl_conditions_without_leaking_to_siblings() {
        let source = RustSource::parse(
            r#"
            #[cfg(test)] mod fixtures { fn hidden() {} }
            #[cfg(test)] impl Engine { fn probe(&self) {} }
            #[cfg_attr(test, inline)] fn production() {}
        "#,
        )
        .or_invariant("inherited cfg fixture");
        assert_eq!(source.function_is_test_only("hidden"), Some(true));
        assert_eq!(source.function_is_test_only("probe"), Some(true));
        assert_eq!(source.function_is_test_only("production"), Some(false));
    }

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
            source.function_is_test_only("compile_life_scaffold"),
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

        assert_eq!(source.function_is_test_only("scaffold"), Some(true));
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
