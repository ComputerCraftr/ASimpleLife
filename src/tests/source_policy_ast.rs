use crate::RequiredExt;
use crate::test_support::attributes_are_test_only;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::test_support::c::CSource;
use syn::parse::Parser as _;
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, ExprCall, ExprMethodCall, ImplItemFn, ItemFn, Lit, Local, Token};
use tree_sitter::Node;

#[derive(Debug)]
pub(super) struct SourceFunction {
    pub(super) path: PathBuf,
    pub(super) name: String,
    pub(super) directly_recursive: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SourceStruct {
    pub(super) path: PathBuf,
    pub(super) name: String,
    pub(super) field_count: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct TestOnlyProductionMutator {
    pub(super) path: PathBuf,
    pub(super) owner: String,
    pub(super) method: String,
}

#[derive(Debug, Default)]
pub(super) struct RustBodyFacts {
    bf_ir_variants: HashSet<String>,
    string_literals: Vec<String>,
    contains_method_call: bool,
    bare_oracle_verify_assertion: bool,
    normalize_assert_eq: bool,
}

impl RustBodyFacts {
    pub(super) fn uses_bf_ir_variant(&self, variant: &str) -> bool {
        self.bf_ir_variants.contains(variant)
    }

    pub(super) fn has_bare_oracle_verify_assertion(&self) -> bool {
        self.bare_oracle_verify_assertion
    }

    pub(super) fn has_normalize_assert_eq(&self) -> bool {
        self.normalize_assert_eq
    }

    pub(super) fn calls_contains(&self) -> bool {
        self.contains_method_call
    }

    pub(super) fn has_diagnostic_literal(&self) -> bool {
        self.string_literals.iter().any(|value| {
            ["frame={", "diff={", "full={"]
                .iter()
                .any(|part| value.contains(part))
        })
    }
}

#[derive(Debug)]
pub(super) struct TestBlock {
    pub(super) path: PathBuf,
    pub(super) name: String,
    pub(super) ignored_reason: Option<String>,
    pub(super) facts: RustBodyFacts,
}

pub(super) fn rust_source_functions(path: &Path, source: &str) -> Vec<SourceFunction> {
    let file = syn::parse_file(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse Rust source {}: {error}", path.display())
    });
    let inherent_methods = inherent_methods(&file);
    let mut collector = RustFunctionCollector {
        path,
        functions: Vec::new(),
        impl_owner: None,
        impl_is_trait: false,
        active_trait: None,
        inherent_methods: &inherent_methods,
    };
    collector.visit_file(&file);
    collector.functions
}

pub(super) fn rust_source_structs(path: &Path, source: &str) -> Vec<SourceStruct> {
    let file = syn::parse_file(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse Rust source {}: {error}", path.display())
    });
    let mut structs = Vec::new();
    collect_rust_structs(&file.items, path, &mut structs);
    structs
}

fn collect_rust_structs(items: &[syn::Item], path: &Path, structs: &mut Vec<SourceStruct>) {
    let mut pending = items.iter().collect::<Vec<_>>();
    while let Some(item) = pending.pop() {
        match item {
            syn::Item::Struct(item_struct) => structs.push(SourceStruct {
                path: path.to_path_buf(),
                name: item_struct.ident.to_string(),
                field_count: item_struct.fields.len(),
            }),
            syn::Item::Mod(item_mod) => {
                if let Some((_, nested)) = &item_mod.content {
                    pending.extend(nested);
                }
            }
            _ => {}
        }
    }
}

pub(super) fn rust_test_only_production_mutators(
    path: &Path,
    source: &str,
) -> Vec<TestOnlyProductionMutator> {
    let file = syn::parse_file(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse Rust source {}: {error}", path.display())
    });
    let mut test_only_types = HashSet::new();
    collect_test_only_types(&file.items, false, &mut test_only_types);
    let mut mutators = Vec::new();
    collect_test_only_mutators(&file.items, false, path, &test_only_types, &mut mutators);
    mutators
}

fn collect_test_only_types(
    items: &[syn::Item],
    inherited_test_only: bool,
    test_only_types: &mut HashSet<String>,
) {
    let mut pending = items
        .iter()
        .map(|item| (item, inherited_test_only))
        .collect::<Vec<_>>();
    while let Some((item, inherited_test_only)) = pending.pop() {
        let test_only = inherited_test_only || attributes_are_test_only(item_attrs(item));
        match item {
            syn::Item::Struct(item_struct) if test_only => {
                test_only_types.insert(item_struct.ident.to_string());
            }
            syn::Item::Enum(item_enum) if test_only => {
                test_only_types.insert(item_enum.ident.to_string());
            }
            syn::Item::Union(item_union) if test_only => {
                test_only_types.insert(item_union.ident.to_string());
            }
            syn::Item::Mod(item_mod) => {
                if let Some((_, nested)) = &item_mod.content {
                    pending.extend(nested.iter().map(|item| (item, test_only)));
                }
            }
            _ => {}
        }
    }
}

fn collect_test_only_mutators(
    items: &[syn::Item],
    inherited_test_only: bool,
    path: &Path,
    test_only_types: &HashSet<String>,
    mutators: &mut Vec<TestOnlyProductionMutator>,
) {
    let mut pending = items
        .iter()
        .map(|item| (item, inherited_test_only))
        .collect::<Vec<_>>();
    while let Some((item, inherited_test_only)) = pending.pop() {
        let test_only = inherited_test_only || attributes_are_test_only(item_attrs(item));
        match item {
            syn::Item::Impl(item_impl) => {
                let Some(owner) = type_name(&item_impl.self_ty) else {
                    continue;
                };
                if item_impl.trait_.is_some() || test_only_types.contains(&owner) {
                    continue;
                }
                for impl_item in &item_impl.items {
                    let syn::ImplItem::Fn(method) = impl_item else {
                        continue;
                    };
                    if !test_only
                        && attributes_are_test_only(&method.attrs)
                        && method.sig.receiver().is_some_and(|receiver| {
                            receiver.mutability.is_some()
                                || matches!(
                                    receiver.kind,
                                    syn::ReceiverKind::Reference(_, _, Some(_))
                                )
                        })
                        && body_mutates_self_state(&method.block)
                    {
                        mutators.push(TestOnlyProductionMutator {
                            path: path.to_path_buf(),
                            owner: owner.clone(),
                            method: method.sig.ident.to_string(),
                        });
                    }
                }
            }
            syn::Item::Mod(item_mod) => {
                if let Some((_, nested)) = &item_mod.content {
                    pending.extend(nested.iter().map(|item| (item, test_only)));
                }
            }
            _ => {}
        }
    }
}

fn body_mutates_self_state(body: &syn::Block) -> bool {
    #[derive(Default)]
    struct MutationVisitor {
        found: bool,
    }

    impl<'ast> Visit<'ast> for MutationVisitor {
        fn visit_expr_assign(&mut self, expression: &'ast syn::ExprAssign) {
            self.found |= expression_is_self_state(&expression.left);
            visit::visit_expr_assign(self, expression);
        }

        fn visit_expr_reference(&mut self, expression: &'ast syn::ExprReference) {
            self.found |=
                expression.mutability.is_some() && expression_is_self_state(&expression.expr);
            visit::visit_expr_reference(self, expression);
        }

        fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
            self.found |= expression_is_self_state(&expression.receiver);
            visit::visit_expr_method_call(self, expression);
        }
    }

    let mut visitor = MutationVisitor::default();
    visitor.visit_block(body);
    visitor.found
}

fn expression_is_self_state(expression: &Expr) -> bool {
    let mut current = expression;
    loop {
        match current {
            Expr::Field(field) => current = &field.base,
            Expr::Index(index) => current = &index.expr,
            Expr::Paren(paren) => current = &paren.expr,
            Expr::Group(group) => current = &group.expr,
            Expr::Path(path) => return path.path.is_ident("self"),
            _ => return false,
        }
    }
}

fn item_attrs(item: &syn::Item) -> &[Attribute] {
    match item {
        syn::Item::Const(item) => &item.attrs,
        syn::Item::Enum(item) => &item.attrs,
        syn::Item::ExternCrate(item) => &item.attrs,
        syn::Item::Fn(item) => &item.attrs,
        syn::Item::ForeignMod(item) => &item.attrs,
        syn::Item::Impl(item) => &item.attrs,
        syn::Item::Macro(item) => &item.attrs,
        syn::Item::Mod(item) => &item.attrs,
        syn::Item::Static(item) => &item.attrs,
        syn::Item::Struct(item) => &item.attrs,
        syn::Item::Trait(item) => &item.attrs,
        syn::Item::TraitAlias(item) => &item.attrs,
        syn::Item::Type(item) => &item.attrs,
        syn::Item::Union(item) => &item.attrs,
        syn::Item::Use(item) => &item.attrs,
        syn::Item::Verbatim(_) | _ => &[],
    }
}

struct RustFunctionCollector<'a> {
    path: &'a Path,
    functions: Vec<SourceFunction>,
    impl_owner: Option<String>,
    impl_is_trait: bool,
    active_trait: Option<Vec<String>>,
    inherent_methods: &'a HashSet<(String, String)>,
}

impl RustFunctionCollector<'_> {
    fn add_function(&mut self, function: &ItemFn) {
        let name = function.sig.ident.to_string();
        self.add(&name, &function.sig, &function.block, false);
    }

    fn add_method(&mut self, method: &ImplItemFn) {
        let name = method.sig.ident.to_string();
        self.add(&name, &method.sig, &method.block, true);
    }

    fn add(&mut self, name: &str, signature: &syn::Signature, body: &syn::Block, is_method: bool) {
        let delegates_to_inherent = self.impl_is_trait
            && self.impl_owner.as_ref().is_some_and(|owner| {
                self.inherent_methods
                    .contains(&(owner.clone(), name.to_string()))
            });
        let mut calls = RustSelfCallVisitor {
            name,
            is_method,
            impl_owner: self.impl_owner.as_deref(),
            active_trait: self.active_trait.as_deref(),
            delegates_to_inherent,
            scopes: vec![signature_bindings(signature)],
            found: false,
        };
        calls.visit_block(body);
        self.functions.push(SourceFunction {
            path: self.path.to_path_buf(),
            name: name.to_string(),
            directly_recursive: calls.found,
        });
    }
}

impl<'ast> Visit<'ast> for RustFunctionCollector<'_> {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.add_function(node);
        visit::visit_item_fn(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let previous = self.impl_owner.take();
        let previous_is_trait = self.impl_is_trait;
        let previous_trait = self.active_trait.take();
        self.impl_owner = type_name(&node.self_ty);
        self.impl_is_trait = node.trait_.is_some();
        self.active_trait = node
            .trait_
            .as_ref()
            .map(|(path, _)| rust_path_segments(path));
        visit::visit_item_impl(self, node);
        self.impl_owner = previous;
        self.impl_is_trait = previous_is_trait;
        self.active_trait = previous_trait;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        self.add_method(node);
        visit::visit_impl_item_fn(self, node);
    }

    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        if let Some(body) = &node.default {
            let name = node.sig.ident.to_string();
            self.add(&name, &node.sig, body, true);
        }
        visit::visit_trait_item_fn(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast syn::ItemTrait) {
        let previous_owner = self.impl_owner.take();
        let previous_is_trait = self.impl_is_trait;
        let previous_trait = self.active_trait.replace(vec![node.ident.to_string()]);
        self.impl_is_trait = true;
        visit::visit_item_trait(self, node);
        self.impl_owner = previous_owner;
        self.impl_is_trait = previous_is_trait;
        self.active_trait = previous_trait;
    }
}

fn type_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn rust_path_segments(path: &syn::Path) -> Vec<String> {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect()
}

fn inherent_methods(file: &syn::File) -> HashSet<(String, String)> {
    struct Collector(HashSet<(String, String)>);

    impl<'ast> Visit<'ast> for Collector {
        fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
            if node.trait_.is_none()
                && let Some(owner) = type_name(&node.self_ty)
            {
                for item in &node.items {
                    if let syn::ImplItem::Fn(method) = item {
                        self.0.insert((owner.clone(), method.sig.ident.to_string()));
                    }
                }
            }
            visit::visit_item_impl(self, node);
        }
    }

    let mut collector = Collector(HashSet::new());
    collector.visit_file(file);
    collector.0
}

struct RustSelfCallVisitor<'a> {
    name: &'a str,
    is_method: bool,
    impl_owner: Option<&'a str>,
    active_trait: Option<&'a [String]>,
    delegates_to_inherent: bool,
    scopes: Vec<HashSet<String>>,
    found: bool,
}

impl RustSelfCallVisitor<'_> {
    fn path_prefix_matches_active_trait(&self, segments: &[&syn::PathSegment]) -> bool {
        self.active_trait.is_some_and(|active_trait| {
            let prefix = &segments[..segments.len().saturating_sub(1)];
            prefix.len() == active_trait.len()
                && prefix
                    .iter()
                    .zip(active_trait)
                    .all(|(segment, expected)| segment.ident == expected)
        })
    }

    fn matching_path_call(&self, call: &ExprCall) -> bool {
        let Expr::Path(function) = call.func.as_ref() else {
            return false;
        };
        let segments = function.path.segments.iter().collect::<Vec<_>>();
        if let Some(qself) = &function.qself {
            let qualified_type = type_name(&qself.ty);
            return self.is_method
                && segments
                    .last()
                    .is_some_and(|segment| segment.ident == self.name)
                && qualified_type.as_deref().is_some_and(|name| {
                    name == "Self" || self.impl_owner.is_some_and(|owner| owner == name)
                })
                && self.path_prefix_matches_active_trait(&segments);
        }
        if segments.len() == 1 {
            return !self.is_method
                && segments[0].ident == self.name
                && !self.binding_is_in_scope();
        }
        if self.is_method
            && segments
                .last()
                .is_some_and(|segment| segment.ident == self.name)
            && self.path_prefix_matches_active_trait(&segments)
        {
            return true;
        }
        self.is_method
            && !self.delegates_to_inherent
            && segments.len() == 2
            && segments[1].ident == self.name
            && (segments[0].ident == "Self"
                || self
                    .impl_owner
                    .is_some_and(|owner| segments[0].ident == owner))
    }

    fn binding_is_in_scope(&self) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|scope| scope.contains(self.name))
    }

    fn with_bindings(&mut self, bindings: HashSet<String>, visit: impl FnOnce(&mut Self)) {
        self.scopes.push(bindings);
        visit(self);
        self.scopes.pop();
    }

    fn visit_let_condition(&mut self, expression: &Expr) -> HashSet<String> {
        let mut pending = vec![expression];
        let mut conditions = Vec::new();
        while let Some(condition) = pending.pop() {
            match condition {
                Expr::Binary(binary) if matches!(binary.op, syn::BinOp::And(_)) => {
                    pending.push(&binary.right);
                    pending.push(&binary.left);
                }
                Expr::Group(group) => pending.push(&group.expr),
                Expr::Paren(paren) => pending.push(&paren.expr),
                _ => conditions.push(condition),
            }
        }

        self.scopes.push(HashSet::new());
        for condition in conditions {
            if let Expr::Let(let_expression) = condition {
                self.visit_expr(&let_expression.expr);
                self.scopes
                    .last_mut()
                    .or_invariant("let condition should have an active scope")
                    .extend(pattern_bindings(&let_expression.pat));
            } else {
                self.visit_expr(condition);
            }
        }
        self.scopes
            .pop()
            .or_invariant("let condition should have an active scope")
    }
}

fn signature_bindings(signature: &syn::Signature) -> HashSet<String> {
    signature
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            syn::FnArg::Receiver(_) => None,
            syn::FnArg::Typed(argument) => Some(pattern_bindings(&argument.pat)),
        })
        .flatten()
        .collect()
}

fn pattern_bindings(pattern: &syn::Pat) -> HashSet<String> {
    struct Collector(HashSet<String>);

    impl<'ast> Visit<'ast> for Collector {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            self.0.insert(pattern.ident.to_string());
            visit::visit_pat_ident(self, pattern);
        }
    }

    let mut collector = Collector(HashSet::new());
    collector.visit_pat(pattern);
    collector.0
}

impl<'ast> Visit<'ast> for RustSelfCallVisitor<'_> {
    fn visit_block(&mut self, node: &'ast syn::Block) {
        let item_bindings = node
            .stmts
            .iter()
            .filter_map(|statement| match statement {
                syn::Stmt::Item(syn::Item::Fn(function)) => Some(function.sig.ident.to_string()),
                _ => None,
            })
            .collect();
        self.with_bindings(item_bindings, |visitor| {
            for statement in &node.stmts {
                visitor.visit_stmt(statement);
            }
        });
    }

    fn visit_local(&mut self, node: &'ast Local) {
        if let Some(initializer) = &node.init {
            self.visit_expr(&initializer.expr);
            if let Some((_, diverge)) = &initializer.diverge {
                self.visit_expr(diverge);
            }
        }
        self.scopes
            .last_mut()
            .or_invariant("Rust function visitor should have an active scope")
            .extend(pattern_bindings(&node.pat));
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        self.found |= self.matching_path_call(node);
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.found |= self.is_method
            && !self.delegates_to_inherent
            && node.method == self.name
            && matches!(node.receiver.as_ref(), Expr::Path(path) if path.path.is_ident("self"));
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_closure(&mut self, node: &'ast syn::ExprClosure) {
        let bindings = node.inputs.iter().flat_map(pattern_bindings).collect();
        self.with_bindings(bindings, |visitor| visitor.visit_expr(&node.body));
    }

    fn visit_arm(&mut self, node: &'ast syn::Arm) {
        self.with_bindings(pattern_bindings(&node.pat), |visitor| {
            if let syn::Pat::Guard(guard) = &node.pat {
                visitor.visit_expr(&guard.guard);
            }
            visitor.visit_expr(&node.body);
        });
    }

    fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
        self.visit_expr(&node.expr);
        self.with_bindings(pattern_bindings(&node.pat), |visitor| {
            visitor.visit_block(&node.body);
        });
    }

    fn visit_expr_if(&mut self, node: &'ast syn::ExprIf) {
        let bindings = self.visit_let_condition(&node.cond);
        self.with_bindings(bindings, |visitor| {
            visitor.visit_block(&node.then_branch);
        });
        if let Some((_, otherwise)) = &node.else_branch {
            self.visit_expr(otherwise);
        }
    }

    fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
        let bindings = self.visit_let_condition(&node.cond);
        self.with_bindings(bindings, |visitor| {
            visitor.visit_block(&node.body);
        });
    }

    // A nested item is a separate function scope and must not make its parent recursive.
    fn visit_item_fn(&mut self, _node: &'ast ItemFn) {}

    fn visit_impl_item_fn(&mut self, _node: &'ast ImplItemFn) {}
}

pub(super) fn c_source_functions(path: &Path, source: &str) -> Vec<SourceFunction> {
    let syntax = CSource::try_parse(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse C source {}: {error}", path.display())
    });

    let mut functions = Vec::new();
    collect_c_functions(syntax.root(), source.as_bytes(), path, &mut functions);
    functions
}

pub(super) fn c_source_structs(path: &Path, source: &str) -> Vec<SourceStruct> {
    let syntax = CSource::try_parse(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse C source {}: {error}", path.display())
    });

    let mut structs = Vec::new();
    let mut pending = vec![syntax.root()];
    while let Some(node) = pending.pop() {
        if node.kind() == "struct_specifier"
            && let Some(body) = node.child_by_field_name("body")
        {
            let mut cursor = body.walk();
            let field_count = body
                .named_children(&mut cursor)
                .filter(|child| child.kind() == "field_declaration")
                .map(c_field_declarator_count)
                .sum();
            if field_count != 0 {
                let name = node
                    .child_by_field_name("name")
                    .and_then(|name| name.utf8_text(source.as_bytes()).ok())
                    .map_or_else(
                        || format!("anonymous struct at line {}", node.start_position().row + 1),
                        str::to_owned,
                    );
                structs.push(SourceStruct {
                    path: path.to_path_buf(),
                    name,
                    field_count,
                });
            }
            continue;
        }
        let mut cursor = node.walk();
        pending.extend(node.named_children(&mut cursor));
    }
    structs
}

fn c_field_declarator_count(declaration: Node<'_>) -> usize {
    let declarators = (0..declaration.child_count())
        .filter(|&index| declaration.field_name_for_child(index) == Some("declarator"))
        .count();
    declarators.max(1)
}

fn collect_c_functions(node: Node<'_>, source: &[u8], path: &Path, out: &mut Vec<SourceFunction>) {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if node.kind() == "function_definition" {
            let declarator = node
                .child_by_field_name("declarator")
                .or_invariant("C function definition should have a declarator");
            let name_node = c_declarator_identifier(declarator)
                .or_invariant("C function definition should have an identifier");
            let name = name_node
                .utf8_text(source)
                .or_invariant("C identifiers should be UTF-8")
                .to_string();
            let body = node
                .child_by_field_name("body")
                .or_invariant("C function definition should have a body");
            out.push(SourceFunction {
                path: path.to_path_buf(),
                directly_recursive: c_body_calls_name(body, source, &name),
                name,
            });
            continue;
        }

        let mut cursor = node.walk();
        pending.extend(node.children(&mut cursor));
    }
}

fn c_declarator_identifier(node: Node<'_>) -> Option<Node<'_>> {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if node.kind() == "identifier" {
            return Some(node);
        }
        if let Some(declarator) = node.child_by_field_name("declarator") {
            pending.push(declarator);
            continue;
        }
        let mut cursor = node.walk();
        pending.extend(node.named_children(&mut cursor));
    }
    None
}

fn c_body_calls_name(node: Node<'_>, source: &[u8], name: &str) -> bool {
    let mut pending = vec![node];
    while let Some(node) = pending.pop() {
        if node.kind() == "call_expression"
            && let Some(function) = node.child_by_field_name("function")
            && function.kind() == "identifier"
            && function
                .utf8_text(source)
                .is_ok_and(|called| called == name)
            && !c_call_name_is_shadowed(node, source, name)
        {
            return true;
        }
        let mut cursor = node.walk();
        pending.extend(node.children(&mut cursor));
    }
    false
}

fn c_call_name_is_shadowed(call: Node<'_>, source: &[u8], name: &str) -> bool {
    let mut child = call;
    while let Some(parent) = child.parent() {
        if parent.kind() == "compound_statement" {
            let mut cursor = parent.walk();
            if parent
                .named_children(&mut cursor)
                .take_while(|sibling| sibling.end_byte() <= child.start_byte())
                .any(|sibling| {
                    sibling.kind() == "declaration" && c_declaration_names(sibling, source, name)
                })
            {
                return true;
            }
        } else if parent.kind() == "function_definition" {
            let declarator = parent
                .child_by_field_name("declarator")
                .or_invariant("C function definition should have a declarator");
            return c_parameters_name(declarator, source, name);
        }
        child = parent;
    }
    false
}

fn c_declaration_names(declaration: Node<'_>, source: &[u8], name: &str) -> bool {
    let mut pending = vec![declaration];
    while let Some(node) = pending.pop() {
        if matches!(node.kind(), "init_declarator" | "declaration")
            && let Some(declarator) = node.child_by_field_name("declarator")
            && c_declarator_identifier(declarator)
                .and_then(|identifier| identifier.utf8_text(source).ok())
                .is_some_and(|identifier| identifier == name)
        {
            return true;
        }
        let mut cursor = node.walk();
        pending.extend(node.named_children(&mut cursor));
    }
    false
}

fn c_parameters_name(declarator: Node<'_>, source: &[u8], name: &str) -> bool {
    let mut pending = vec![declarator];
    while let Some(node) = pending.pop() {
        if node.kind() == "parameter_declaration" {
            if let Some(parameter) = node.child_by_field_name("declarator")
                && c_declarator_identifier(parameter)
                    .and_then(|identifier| identifier.utf8_text(source).ok())
                    .is_some_and(|identifier| identifier == name)
            {
                return true;
            }
            continue;
        }
        let mut cursor = node.walk();
        pending.extend(node.named_children(&mut cursor));
    }
    false
}

pub(super) fn rust_test_blocks(path: &Path, source: &str) -> Vec<TestBlock> {
    let file = syn::parse_file(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse Rust tests {}: {error}", path.display())
    });
    let mut collector = RustTestCollector {
        path,
        tests: Vec::new(),
    };
    collector.visit_file(&file);
    collector.tests
}

struct RustTestCollector<'a> {
    path: &'a Path,
    tests: Vec<TestBlock>,
}

impl Visit<'_> for RustTestCollector<'_> {
    fn visit_item_fn(&mut self, node: &ItemFn) {
        if node
            .attrs
            .iter()
            .any(|attribute| attribute.path().is_ident("test"))
        {
            let mut facts = RustBodyFacts::default();
            facts.visit_block(&node.block);
            self.tests.push(TestBlock {
                path: self.path.to_path_buf(),
                name: node.sig.ident.to_string(),
                ignored_reason: node.attrs.iter().find_map(ignore_reason),
                facts,
            });
        }
        visit::visit_item_fn(self, node);
    }
}

fn ignore_reason(attribute: &Attribute) -> Option<String> {
    if !attribute.path().is_ident("ignore") {
        return None;
    }
    match &attribute.meta {
        syn::Meta::NameValue(value) => match &value.value {
            syn::Expr::Lit(literal) => match &literal.lit {
                syn::Lit::Str(reason) => Some(reason.value()),
                _ => Some(String::new()),
            },
            _ => Some(String::new()),
        },
        _ => Some(String::new()),
    }
}

impl RustBodyFacts {
    fn inspect_assert_macro(&mut self, node: &syn::Macro) {
        let Some(name) = node
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
        else {
            return;
        };
        if name != "assert" && name != "assert_eq" {
            return;
        }
        let Ok(arguments) =
            Punctuated::<Expr, Token![,]>::parse_terminated.parse2(node.tokens.clone())
        else {
            return;
        };
        let Some(first) = arguments.first() else {
            return;
        };
        if name == "assert" {
            self.bare_oracle_verify_assertion |= matches!(first, Expr::MethodCall(call)
                if call.method.to_string().starts_with("verify_")
                    && matches!(call.receiver.as_ref(), Expr::Path(path) if path.path.is_ident("oracle")));
        } else {
            self.normalize_assert_eq |= is_call_named(first, "normalize");
        }
    }
}

impl<'ast> Visit<'ast> for RustBodyFacts {
    fn visit_path(&mut self, node: &'ast syn::Path) {
        let segments = node.segments.iter().collect::<Vec<_>>();
        if segments.len() >= 2 && segments[segments.len() - 2].ident == "BfIr" {
            self.bf_ir_variants
                .insert(segments[segments.len() - 1].ident.to_string());
        }
        visit::visit_path(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.contains_method_call |= node.method == "contains";
        visit::visit_expr_method_call(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        self.inspect_assert_macro(node);
        if let Ok(arguments) =
            Punctuated::<Expr, Token![,]>::parse_terminated.parse2(node.tokens.clone())
        {
            for argument in arguments {
                self.visit_expr(&argument);
            }
        }
        visit::visit_macro(self, node);
    }

    fn visit_lit(&mut self, node: &'ast Lit) {
        if let Lit::Str(value) = node {
            self.string_literals.push(value.value());
        }
        visit::visit_lit(self, node);
    }
}

fn is_call_named(expression: &Expr, expected: &str) -> bool {
    matches!(expression, Expr::Call(call)
        if matches!(call.func.as_ref(), Expr::Path(path)
            if path.path.segments.last().is_some_and(|segment| segment.ident == expected)))
}
