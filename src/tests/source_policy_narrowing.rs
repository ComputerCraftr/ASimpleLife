use std::collections::HashSet;
use std::path::{Path, PathBuf};

use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};
use syn::{
    Attribute, Expr, ExprCall, ExprCast, ExprMethodCall, FnArg, ImplItemFn, ItemFn, Lit, Local,
    Pat, Token, Type,
};

#[derive(Debug, PartialEq, Eq)]
pub(super) struct NarrowingViolation {
    pub(super) path: PathBuf,
    pub(super) function: String,
    pub(super) target: String,
}

pub(super) fn rust_wide_narrowing_violations(path: &Path, source: &str) -> Vec<NarrowingViolation> {
    let file = syn::parse_file(source).unwrap_or_else(|error| {
        crate::invariant_failure!("failed to parse Rust source {}: {error}", path.display())
    });
    let mut visitor = NarrowingVisitor {
        path,
        function: None,
        wide_bindings: HashSet::new(),
        violations: Vec::new(),
    };
    visitor.visit_file(&file);
    visitor.violations
}

fn is_checked_narrowing_boundary(attribute: &Attribute) -> bool {
    if !attribute.path().is_ident("doc") {
        return false;
    }
    let syn::Meta::NameValue(value) = &attribute.meta else {
        return false;
    };
    let Expr::Lit(literal) = &value.value else {
        return false;
    };
    matches!(&literal.lit, Lit::Str(value) if value.value() == "source-policy: checked-narrowing-boundary")
}

struct NarrowingVisitor<'a> {
    path: &'a Path,
    function: Option<String>,
    wide_bindings: HashSet<String>,
    violations: Vec<NarrowingViolation>,
}

impl NarrowingVisitor<'_> {
    fn enter_function(&mut self, name: String, inputs: &Punctuated<FnArg, Token![,]>) {
        self.function = Some(name);
        self.wide_bindings.clear();
        for input in inputs {
            let FnArg::Typed(argument) = input else {
                continue;
            };
            if type_is_wide(&argument.ty) {
                collect_pat_bindings(&argument.pat, &mut self.wide_bindings);
            }
        }
    }

    fn record(&mut self, target: String) {
        self.violations.push(NarrowingViolation {
            path: self.path.to_path_buf(),
            function: self
                .function
                .clone()
                .unwrap_or_else(|| "<module>".to_string()),
            target,
        });
    }
}

impl<'ast> Visit<'ast> for NarrowingVisitor<'_> {
    fn visit_item_fn(&mut self, function: &'ast ItemFn) {
        if function.attrs.iter().any(is_checked_narrowing_boundary) {
            return;
        }
        self.enter_function(function.sig.ident.to_string(), &function.sig.inputs);
        visit::visit_block(self, &function.block);
        self.function = None;
        self.wide_bindings.clear();
    }

    fn visit_impl_item_fn(&mut self, function: &'ast ImplItemFn) {
        if function.attrs.iter().any(is_checked_narrowing_boundary) {
            return;
        }
        self.enter_function(function.sig.ident.to_string(), &function.sig.inputs);
        visit::visit_block(self, &function.block);
        self.function = None;
        self.wide_bindings.clear();
    }

    fn visit_local(&mut self, local: &'ast Local) {
        if let Pat::Type(typed) = &local.pat
            && type_is_wide(&typed.ty)
        {
            collect_pat_bindings(&typed.pat, &mut self.wide_bindings);
        }
        visit::visit_local(self, local);
    }

    fn visit_expr_cast(&mut self, expression: &'ast ExprCast) {
        if let Some(target) = narrowing_target(&expression.ty) {
            let direct_integer_cast_is_forbidden = matches!(target.as_str(), "i64" | "u64");
            if direct_integer_cast_is_forbidden
                || expression_is_known_wide(&expression.expr, &self.wide_bindings)
            {
                self.record(target);
            }
        }
        visit::visit_expr_cast(self, expression);
    }

    fn visit_expr_call(&mut self, expression: &'ast ExprCall) {
        if let Expr::Path(callee) = expression.func.as_ref() {
            let segments = callee.path.segments.iter().collect::<Vec<_>>();
            if segments.len() == 2
                && segments[1].ident == "try_from"
                && matches!(
                    segments[0].ident.to_string().as_str(),
                    "usize" | "u64" | "i64"
                )
                && expression
                    .args
                    .first()
                    .is_some_and(|argument| expression_is_known_wide(argument, &self.wide_bindings))
            {
                self.record(segments[0].ident.to_string());
            }
        }
        visit::visit_expr_call(self, expression);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        if expression.method == "try_into"
            && expression_is_known_wide(&expression.receiver, &self.wide_bindings)
        {
            self.record("untyped try_into".to_string());
        }
        visit::visit_expr_method_call(self, expression);
    }
}

fn type_is_wide(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.path.segments.last().is_some_and(|segment| matches!(segment.ident.to_string().as_str(), "u128" | "i128")))
}

fn narrowing_target(ty: &Type) -> Option<String> {
    let Type::Path(path) = ty else { return None };
    let target = path.path.segments.last()?.ident.to_string();
    matches!(target.as_str(), "usize" | "u64" | "i64").then_some(target)
}

fn expression_is_known_wide(expression: &Expr, bindings: &HashSet<String>) -> bool {
    let mut expression = expression;
    loop {
        match expression {
            Expr::Paren(paren) => expression = &paren.expr,
            Expr::Group(group) => expression = &group.expr,
            _ => break,
        }
    }
    match expression {
        Expr::Path(path) => path
            .path
            .get_ident()
            .is_some_and(|ident| bindings.contains(&ident.to_string())),
        Expr::Lit(literal) => {
            matches!(&literal.lit, Lit::Int(value) if matches!(value.suffix(), "u128" | "i128"))
        }
        _ => false,
    }
}

fn collect_pat_bindings(pattern: &Pat, bindings: &mut HashSet<String>) {
    let mut pending = vec![pattern];
    while let Some(pattern) = pending.pop() {
        match pattern {
            Pat::Ident(ident) => {
                bindings.insert(ident.ident.to_string());
            }
            Pat::Paren(paren) => pending.push(&paren.pat),
            Pat::Reference(reference) => pending.push(&reference.pat),
            Pat::Tuple(tuple) => pending.extend(&tuple.elems),
            _ => {}
        }
    }
}
