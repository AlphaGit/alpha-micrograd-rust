use std::ops::Add;
use std::ops::Mul;

#[derive(Debug)]
struct Value {
    data: f64,
    previous: Vec<Value>,
}

macro_rules! new_value {
    ($data:expr) => {
        Value::new($data, vec![])
    };
    ($data: expr, $($rest: expr),+) => {
        Value::new($data, vec![$($rest),+])
    };
}

impl Value {
    fn new(data: f64, previous: Vec<Value>) -> Value {
        Value { data, previous }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        Value::new(self.data + other.data, vec![self, other])
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        Value::new(self.data * other.data, vec![self, other])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_be_instantiated() {
        let value = new_value![3.0];
        assert_eq!(value.data, 3.0);
    }

    #[test]
    fn can_be_instantiated_with_previous() {
        let value1 = new_value![3.0];
        let value2 = new_value![4.0, value1];
        assert_eq!(value2.data, 4.0);
        assert_eq!(value2.previous[0].data, 3.0);
    }

    #[test]
    fn can_be_instantiated_with_multiple_previous() {
        let value1 = new_value![3.0];
        let value2 = new_value![4.0];
        let value3 = new_value![5.0, value1, value2];
        assert_eq!(value3.data, 5.0);
        assert_eq!(value3.previous[0].data, 3.0);
        assert_eq!(value3.previous[1].data, 4.0);
    }

    #[test]
    fn can_add() {
        let value1 = new_value![3.0];
        let value2 = new_value![4.0];
        let result = value1 + value2;
        assert_eq!(result.data, 7.0);
    }

    #[test]
    fn can_multiply() {
        let value1 = new_value![3.0];
        let value2 = new_value![4.0];
        let result = value1 * value2;
        assert_eq!(result.data, 12.0);
    }
}
