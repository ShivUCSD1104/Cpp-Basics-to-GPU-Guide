**BASICS**

1. Preprocessor directive:
	1. Pre compilation
	2. ```#include <iostream>```
		1. <> , "" for standard , user defined libs
2. namespace std;
	1. use standard lib objs without std::
3. int Main(){}
	1. returns int, takes no args except CLI args
	```cpp
	int main(int argc, char *argv[]){
	}
	OR
	int main(int argc, char **argv){
	}
	```
	2. argc ----> number of args (default 1  ---> ./the exec file)
	3. char argv -----> CLI args passed as chars, 
		1. ```argv[0] = exec file```
4. cout << x = print x, cin >> x = input x
	1. ``` int x, y;  
		std::cin >> x >> y;  
		std::cout << x << y; ```
5. ```const <type> var = val;``` same as JS const
6. Datatype - Space - Limit
	1. integer - 4 bytes - $[-2^{31}, 2^{31}]$
	2. double - 8 bytes - 15 decimal digits
	3. char - 1 byte 
	4. bool - 1 byte
	5. std::string - requires ```<string>```
		1. str.length()
		2. access chars by ```[]```
	6. (type) --> ``` double x = (double) x```
7. Operators
	1. Logic --> && , || , !
	2. Bitwise 
		1. x << y - left shift - x bits by y 
		2. x >> y - right shift - x bits by y
		3. ~x - bitwise NOT - flip bits in x
		4. & - bitwise AND - each bit in x AND y
		5. | - bitwise OR - each bit in x OR y
		6. ^ - bitwise XOR - each bit in x XOR y
	3. Ternary --> ? :
8. Switch Case 6
```cpp
switch (boolean) {  
  case 1:  
    cout << "Case 1";  
    break;  
  case 2:  
    cout << "Case 2";
  default:
	cout << "Default";
 }
```
9. Do-while
```cpp
do{
	cout<<"Did"; // will still exec once
}while(False);
```
10. For each 
```cpp
int fibonacci[5] = {0, 1, 1, 2, 3};  
for (int number : fibonacci) {  
  std::cout << number;  
}
```

