// https://projecteuler.net/problem=735

// Let f(n) be the number of divisors of 2 n 2 that are no greater than n. For example, f ( 15 ) = 8 because there are 8 such divisors: 1,2,3,5,6,9,10,15.
// Note that 18 is also a divisor of 2 × 15 2 but it is not counted because it is greater than 15.

use std::collections::HashMap;
use std::vec::Vec;
use std::iter::*;

fn solve(n: Option<i64>) -> usize {
    let mut factors_cache: HashMap<i64, Vec<i64>> = HashMap::new();

    fn factors_of(x: i64) -> () {
        let mut prime_factors = Vec::new();
        prime_factors.push(2);

        let array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    }

    // factors_of(n).len()
    return 0;
}

fn example() {
    let strings = vec!["tofu", "93", "18"];
    let numbers: Vec<_> = strings
        .into_iter()
        .map(|s| s.parse::<i32>())
        .filter_map(Result::ok)
        .collect();
    println!("Results: {:?}", numbers);
}


fn main() {
    println!("{:?}", solve(Option::Some(200000)));
}
