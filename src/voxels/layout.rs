/// Layout system based on linear constraints

/// Ideas
/// - 

/// 
enum Operator { ADDITION, SUBTRACTION, MULTIPLICATION }

/// 
enum Operand<Variable, Number> {
    CONSTANT, VARIABLE(Variable, Number)
}

/// 
// TODO: if we use a type alias, it might be difficult or inconvenient to implement traits?
type Constraint = (Operand, Operator, Operand)
//struct Constraint<Variable> {
//    let lhs: Operator
//    let rhs: Operator
//    let op: Operand
//}