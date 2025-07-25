**AUTO**
```cpp
// lets C++ deduce the type of the variable
int fibonacci[5] = {0, 1, 1, 2, 3};  
for (auto number : fibonacci) {  
  std::cout << number;  
}
```


**FUNCTIONS** 
1. Declaration
```cpp
//default params follow non-default params
<return type> name(args, default_arg=0){

}
```
2. Method Overloading
	1. Each function has different types of parameters
					OR
	2. Each function has a different number of parameters
```cpp
//OVERLOADING 
int add(int a, int b) {  
  return a + b;  
}  
 
double add(double a, double b) {  
  return a + b;  
}

int main() {  
  cout << add(3, 2);   // Calls add(int, int)  
  cout << "\n";  
  cout << add(5.3, 1.4);   // Calls add(double, double)  
}
```

[[Basics 2.1 - References]] For Reference Params