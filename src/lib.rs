//! A Rust implementation of micrograd, a tiny autograd engine
//! 
//! This crate provides a minimal implementation of an autograd engine,
//! allowing for backpropagation through a computational graph.

#![deny(missing_docs)]
#[doc = include_str!("../README.md")]
/// Module containing the core value and expression types
pub mod value;
/// Module containing neural network layers and operations
pub mod nn;
/// Module containing the compiled graph and its operations
pub mod compiled;
/// Module containing operation types and enums
pub mod operations;
/// Module containing multi-value (matrices) operations and types
pub mod multi_value;