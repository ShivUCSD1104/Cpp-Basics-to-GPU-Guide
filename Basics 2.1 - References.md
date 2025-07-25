
**Address of Operator**: &
```cpp
// Print the value of message (Hello World!)  
cout << message << endl;  
  
// Print the memory address of message (0x7ffee9b21af0)  
cout << &message << endl;
```

**References**: Alias to an existing variable declared using ```&```
1. References must be initialized 
2. References should not be reassigned 
```cpp
int& length; // THIS IS NOT VALID
int x = 5, y = 2; 
int& xref = x; 
xref = y; // DON'T DO THIS!
```
3. Main use is function args as the functions can modify args passed as ref args

```cpp
// PASS BY REFERENCE IN FUNCTIONS
void change(int& x){
	x = 3;
}
int main(){
	int x = 5;
	cout << x; //prints 5
	change(x);
	cout << x; //prints 3
}
```