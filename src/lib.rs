#[derive(Debug)]
struct Value {
    data: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_be_instantiated() {
        let value = Value { data: 3.0 };
        assert_eq!(value.data, 3.0);
    }
}
