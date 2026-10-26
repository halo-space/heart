//! Compile-time validation for Workflow Node registration.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::{quote, quote_spanned};
use syn::{
    FnArg, GenericArgument, Ident, ImplItem, Item, ItemFn, ItemImpl, PathArguments, ReturnType,
    Signature, Type,
    parse::{Parse, ParseStream},
    spanned::Spanned,
    visit::{self, Visit},
};

enum Mode {
    Function,
    Call(Ident),
    Signal(Ident),
}

impl Parse for Mode {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self::Function);
        }
        let key: Ident = input.parse()?;
        input.parse::<syn::Token![=]>()?;
        let method: Ident = input.parse()?;
        if !input.is_empty() {
            return Err(input.error("use exactly one of call = method or signal = method"));
        }
        match key.to_string().as_str() {
            "call" => Ok(Self::Call(method)),
            "signal" => Ok(Self::Signal(method)),
            _ => Err(syn::Error::new(
                key.span(),
                "expected call = method or signal = method",
            )),
        }
    }
}

#[proc_macro_attribute]
pub fn node(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let result = syn::parse::<Mode>(attributes).and_then(|mode| {
        syn::parse::<Item>(item).and_then(|item| {
            let root = root()?;
            expand(mode, item, &root)
        })
    });
    result
        .unwrap_or_else(|error| error.into_compile_error())
        .into()
}

fn root() -> syn::Result<Tokens> {
    use proc_macro_crate::{FoundCrate, crate_name};
    match crate_name("halo-agent") {
        Ok(FoundCrate::Itself) => Ok(quote!(::halo)),
        // Cargo exposes this package's explicit `[lib] name = "halo"` when
        // the dependency has not been renamed by its consumer.
        Ok(FoundCrate::Name(name)) if name == "halo_agent" => Ok(quote!(::halo)),
        Ok(FoundCrate::Name(name)) => {
            let name = syn::parse_str::<Ident>(&name)
                .or_else(|_| syn::parse_str::<Ident>(&format!("r#{name}")))?;
            Ok(quote!(::#name))
        }
        Err(error) => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            error.to_string(),
        )),
    }
}

fn expand(mode: Mode, item: Item, root: &Tokens) -> syn::Result<Tokens> {
    match (mode, item) {
        (Mode::Function, Item::Fn(function)) => expand_function(function, root),
        (Mode::Call(method), Item::Impl(implementation)) => {
            expand_impl(implementation, method, false, root)
        }
        (Mode::Signal(method), Item::Impl(implementation)) => {
            expand_impl(implementation, method, true, root)
        }
        (Mode::Function, item) => Err(syn::Error::new(
            item.span(),
            "#[node] requires a free async function; use #[node(call = method)] or #[node(signal = method)] on an inherent impl",
        )),
        (_, item) => Err(syn::Error::new(
            item.span(),
            "call/signal must mark the containing inherent impl, not a standalone method",
        )),
    }
}

struct Io {
    input: Type,
    output: Type,
    error: Type,
}

fn check_signature(signature: &Signature, method: bool, signal: bool) -> syn::Result<Io> {
    if signature.asyncness.is_none() {
        return Err(syn::Error::new(
            signature.span(),
            "node functions and selected methods must be async",
        ));
    }
    if signature.constness.is_some()
        || signature.unsafety.is_some()
        || signature.abi.is_some()
        || signature.variadic.is_some()
    {
        return Err(syn::Error::new(
            signature.span(),
            "use a regular safe Rust async signature for #[node]",
        ));
    }
    if !signature.generics.params.is_empty() || signature.generics.where_clause.is_some() {
        return Err(syn::Error::new(
            signature.generics.span(),
            "generic node signatures require an explicit Node implementation or a concrete ordinary function",
        ));
    }
    let expected = if method { 3 } else { 2 };
    if signature.inputs.len() != expected {
        return Err(syn::Error::new(
            signature.inputs.span(),
            if method {
                "selected method must take &self, one owned business input, and &Cancellation"
            } else {
                "node function must take one owned business input and Cancellation"
            },
        ));
    }
    let mut arguments = signature.inputs.iter();
    if method {
        let Some(FnArg::Receiver(receiver)) = arguments.next() else {
            return Err(syn::Error::new(
                signature.inputs.span(),
                "selected method must take &self",
            ));
        };
        if receiver.reference.is_none()
            || receiver.mutability.is_some()
            || receiver.colon_token.is_some()
        {
            return Err(syn::Error::new(
                receiver.span(),
                "selected method must take an immutable &self receiver",
            ));
        }
    }
    let Some(FnArg::Typed(input)) = arguments.next() else {
        return Err(syn::Error::new(
            signature.inputs.span(),
            "expected one owned business input",
        ));
    };
    owned(&input.ty)?;
    let Some(FnArg::Typed(cancellation)) = arguments.next() else {
        return Err(syn::Error::new(
            signature.inputs.span(),
            "missing Cancellation parameter",
        ));
    };
    let correct_cancellation = if method {
        matches!(&*cancellation.ty, Type::Reference(reference)
            if reference.mutability.is_none() && named(&reference.elem, "Cancellation"))
    } else {
        named(&cancellation.ty, "Cancellation")
    };
    if !correct_cancellation {
        return Err(syn::Error::new(
            cancellation.ty.span(),
            if method {
                "selected component method must take &Cancellation"
            } else {
                "free node function must take Cancellation by value"
            },
        ));
    }
    let ReturnType::Type(_, result) = &signature.output else {
        return Err(syn::Error::new(
            signature.output.span(),
            "node must return Result<O, E>",
        ));
    };
    let types = arguments_of(result, "Result", 2)?;
    let output = if signal {
        arguments_of(&types[0], "Signal", 1)?[0].clone()
    } else {
        if named(&types[0], "Signal") {
            return Err(syn::Error::new(
                types[0].span(),
                "Signal results require #[node(signal = method)] on an inherent impl",
            ));
        }
        types[0].clone()
    };
    owned(&output)?;
    owned(&types[1])?;
    Ok(Io {
        input: (*input.ty).clone(),
        output,
        error: types[1].clone(),
    })
}

fn named(ty: &Type, name: &str) -> bool {
    matches!(ty, Type::Path(path) if path.qself.is_none()
        && path.path.segments.last().is_some_and(|segment| segment.ident == name))
}

fn arguments_of(ty: &Type, name: &str, count: usize) -> syn::Result<Vec<Type>> {
    if let Type::Path(path) = ty
        && path.qself.is_none()
        && let Some(segment) = path.path.segments.last()
        && segment.ident == name
        && let PathArguments::AngleBracketed(arguments) = &segment.arguments
    {
        let types = arguments
            .args
            .iter()
            .filter_map(|argument| match argument {
                GenericArgument::Type(ty) => Some(ty.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        if types.len() == count && arguments.args.len() == count {
            return Ok(types);
        }
    }
    Err(syn::Error::new(
        ty.span(),
        if name == "Result" {
            "node must explicitly return Result<O, E>"
        } else {
            "signal method must explicitly return Result<node::Signal<O>, E>"
        },
    ))
}

fn owned(ty: &Type) -> syn::Result<()> {
    struct Borrowed(Option<proc_macro2::Span>);
    impl<'ast> Visit<'ast> for Borrowed {
        fn visit_type_reference(&mut self, reference: &'ast syn::TypeReference) {
            self.0.get_or_insert(reference.span());
        }
        fn visit_lifetime(&mut self, lifetime: &'ast syn::Lifetime) {
            self.0.get_or_insert(lifetime.span());
        }
        fn visit_type_path(&mut self, path: &'ast syn::TypePath) {
            if path
                .path
                .segments
                .first()
                .is_some_and(|segment| segment.ident == "Self")
            {
                self.0.get_or_insert(path.span());
            } else {
                visit::visit_type_path(self, path);
            }
        }
    }
    let mut borrowed = Borrowed(None);
    borrowed.visit_type(ty);
    if let Some(span) = borrowed.0 {
        Err(syn::Error::new(
            span,
            "use concrete owned node I/O and error types; borrowed/lifetime/Self-projection signatures require an explicit Node impl",
        ))
    } else {
        Ok(())
    }
}

/// An anonymous, never-executed type-check assertion. Referencing the existing
/// registration method checks precisely its Schema/Serde/error bounds without
/// executing it, constructing a Node, or inventing a node_id or helper API.
fn check(io: &Io, value: Option<&Ident>, ty: Option<&Type>, root: &Tokens) -> Tokens {
    let Io {
        input,
        output,
        error,
    } = io;
    // A user type named N (or any internal-looking name) must not be captured
    // by the assertion's type parameter. Pick a name absent from all I/O tokens.
    let types = quote!(#input #output #error).to_string();
    let index = (0u32..)
        .find(|index| !types.contains(&format!("__HeartNodeDeclaration{index}")))
        .expect("finite type tokens");
    let parameter = Ident::new(
        &format!("__HeartNodeDeclaration{index}"),
        proc_macro2::Span::mixed_site(),
    );
    let value_argument = if value.is_some() {
        quote!(_: #parameter)
    } else {
        quote!()
    };
    let binding = if let Some(value) = value {
        quote!(let callable = #value;)
    } else {
        quote!()
    };
    let assertion = if value.is_some() {
        quote!(let _ = verify(callable);)
    } else {
        let ty = ty.expect("impl type");
        quote!(let _ = verify::<#ty>;)
    };
    quote_spanned! {input.span()=>
        const _: fn() = || {
            #binding
            {
            fn verify<#parameter: #root::agent::node::Node<#input, #output, #error>>(
                #value_argument
            ) {
                let _: fn(
                    &mut #root::runtime::workflow::WorkflowBuilder,
                    ::std::string::String, #parameter,
                ) -> ::std::result::Result<
                    &mut #root::runtime::workflow::WorkflowBuilder,
                    #root::runtime::workflow::GraphError,
                > = #root::runtime::workflow::WorkflowBuilder::add_node::<#input, #output, #error, #parameter>;
            }
            #assertion
            }
        };
    }
}

fn expand_function(function: ItemFn, root: &Tokens) -> syn::Result<Tokens> {
    let io = check_signature(&function.sig, false, false)?;
    let assertion = check(&io, Some(&function.sig.ident), None, root);
    Ok(quote!(#function #assertion))
}

fn expand_impl(
    implementation: ItemImpl,
    method: Ident,
    signal: bool,
    root: &Tokens,
) -> syn::Result<Tokens> {
    if implementation.trait_.is_some()
        || implementation.unsafety.is_some()
        || !implementation.generics.params.is_empty()
        || implementation.generics.where_clause.is_some()
    {
        return Err(syn::Error::new(
            implementation.span(),
            "mark a concrete inherent impl; generic or trait impls require an explicit Node impl",
        ));
    }
    let selected = implementation
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == method => Some(function),
            _ => None,
        })
        .ok_or_else(|| {
            syn::Error::new(
                method.span(),
                "selected node method must be defined in this inherent impl",
            )
        })?;
    let io = check_signature(&selected.sig, true, signal)?;
    let Io {
        input,
        output,
        error,
    } = &io;
    let ty = &implementation.self_ty;
    let result = if signal {
        quote!(<#ty>::#method(self, input, &cancellation).await)
    } else {
        quote!(<#ty>::#method(self, input, &cancellation).await.map(#root::agent::node::Signal::Complete))
    };
    let assertion = check(&io, None, Some(ty), root);
    Ok(quote!(
        #implementation
        impl #root::agent::node::Node<#input, #output, #error> for #ty {
            async fn call(&self, input: #input, cancellation: &#root::Cancellation)
                -> ::std::result::Result<#root::agent::node::Signal<#output>, #error>
            { #result }
        }
        #assertion
    ))
}
