use std::ops::Add;
use std::ops::Mul;

#[derive(Debug)]
struct Value {
    data: f64,
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        Value {
            data: self.data + other.data,
        }
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        Value {
            data: self.data * other.data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_be_instantiated() {
        let value = Value { data: 3.0 };
        assert_eq!(value.data, 3.0);
    }

    #[test]
    fn can_add() {
        let value1 = Value { data: 3.0 };
        let value2 = Value { data: 4.0 };
        let result = value1 + value2;
        assert_eq!(result.data, 7.0);
    }

    #[test]
    fn can_multiply() {
        let value1 = Value { data: 3.0 };
        let value2 = Value { data: 4.0 };
        let result = value1 * value2;
        assert_eq!(result.data, 12.0);
    }
}
