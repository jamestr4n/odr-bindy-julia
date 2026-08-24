println("Hello World!")
1 + 1 
2^3
3 < 2
1 == 1 && 2 > 3
1 == 1 || 2 > 3
true + true * 2

x = 1
y = 2
z = x + y

x = 2

y

z

z  = x + y

i = 1
i += 1

typeof(-3)
typeof(3.14)
typeof(pi)
typeof("Hello World!")
round(pi; digits = 2)
1000000 == 1_000_000

div(4, 2)
4 ÷ 2
4 / 2


typeof('a')
typeof('a' + 1)
typeof("char")

s3 = "doggo"
println("$s3 is a good boy!")

s1 = "Hello"
s1_s3 = s1 * s3

#10.12 arrays

col_vec = [5, 2, 3]
typeof(col_vec)

row_vec = Float32[1 2 3]
typeof(row_vec)

sort(col_vec)
col_vec
sort!(col_vec)

matrix = [1 2 3; 4 5 6; 7 8 9]

#10.13 Tuples

#10.14 Named Tuples

dog = (
    name = "doggo",
    age = 3,
    breed = "golden retriever"
)
dog[1]

dog.name
dog.breed

#10.15 Dictionaries
dog = Dict(
    "name" => "doggo",
    "age" => 3,
    "breed" => "golden retriever"
)
dog["name"]

#10.16 Struct

mutable struct Dog
    name::String
    age::Int
    breed::String
end

mydog = Dog(
    "doggo",
    3,
    "golden retriever"
)

typeof(mydog)

mydog.name
mydog.age   
mydog.breed

mydog.name = "doggo2"
mydog.age = 4
mydog

# 10.17 Control flow: if statements

x, y = 1, 2

task_1() = println("$x > $y")
task_2() = println("$x < $y")
task_3() = println("$x == $y")

if x > y 
    task_1()
elseif x < y
    task_2()
else
    task_3()
end

# 10.18 Control flow: ternary operator
x, y = 1, 2

t1() = println("$x > $y")
t2() = println("$x < $y")
t3() = println("$x == $y")

x > y ? t1() : (x < y ? t2() : t3())

# 10.19 Control flow: while loops
i = 1
while i <= 5
    println(i)
    i += 1
end

# 10.20 Control flow: for loops
for i in 1:2:10
    println(i)
end

#10.21 Control flow: For loop over collection

myarray = [10, 20, 30, 40, 50]

for element in myarray
    println(element)
end

mydog = Dict(
    "name" => "doggo",
    "age" => 3,
    "breed" => "golden retriever"
)

for (key, value) in mydog
    println("$key: $value")
end

# 10.22 Comprehension

cubed = [x^3 for x in 1:5]

# 10.23 Functions   

function myadd(x, y)
    return x + y
end

myadd(1, 2)

f(a, b) = sqrt(a^2 + b^2)
f(2, 3)

# 10.24 Multiple dispatch

function mytypeof(x::Int64)
    return "This is an Int64"
end

function mytypeof(x::Float64)
    return "This is a Float64"
end

function mytypeof(x::Number)
    return "This is a Number"
end

function mytypeof(x::Any)
    return "This is something else"
end



function mygenericfunction(x)
    println("$x is type: ",
        mytypeof(x)
        )
end

mygenericfunction(pi)
mygenericfunction([1, 2, 3])

struct Dog
    name::String
end

function mytypeof(x::Dog)
    return "This is a Dog"
end

mydog = Dog("doggo")
mygenericfunction(mydog)

methods(mytypeof)
methods(mygenericfunction)

# 10.25 Anonymous functions

firstname = [
"John", "Jane", "Jim"
]

map(length, firstname)
map(x -> x * "Doggo", firstname)

# 10.26 Julia's standard libraries

rand(10)

using Random
Random.seed!(1)
rand(10)

Random.seed!(42)

data = randn(1000)

function average(x::Vector)
    return sum(x) / length(x)
end

using Statistics

mean(data)

# 10.27 External packages

using Plots

f(x) = x^3 -2x

plot(f)

plot(f;
    legend = false,
    linewidth = 3,
    color = :red,
    lims = (-2, 2),
    aspect_ratio = 1
)

using Random
Random.seed!(42)

xs = randn(1000)
ys = randn(1000)

scatter(xs, ys;
    legend = false,
    color = :blue,
    alpha = 0.5
)