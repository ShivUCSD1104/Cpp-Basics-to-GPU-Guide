```cpp
#include <iostream>  
  
// Base class  
class Animal {  
private:  
  std::string gender;  
  int age;  
  
public:  
  Animal(std::string new_gender, int new_age)  
    : gender(new_gender), age(new_age) {}  
};  
  
// Derived class  
class Dog: public Animal  {  
private:  
  std::string breed;  
  
public:  
  // Call base class constructor  
  Dog(std::string new_gender, int new_age, std::string new_breed)  
    : Animal(new_gender, new_age), breed(new_breed) {}  
  
  void sound() {  
    std::cout << "Woof\n";  
  }  
};  
  
int main() {  
  // Calls Dog(string, int, string) constructor  
  Dog buddy("male", 8, "Husky");  
    
  // Output: Woof  
  buddy.sound();  
  
  return 0;  
}
```

#### Multilevel Inheritence
```cpp
#include <iostream>  
  
class A {   // A is the base class  
public:  
  int a;  
  
  A() { std::cout << "Constructing A\n"; }  
};  
  
class B: public A { // class B inherits from class A  
public:  
  int b;  
  
  B() { std::cout << "Constructing B\n"; }  
};  
  
class C: public B { // class C inherits from class B  
public:  
  int c;  
  
  C() { std::cout << "Constructing C\n"; }  
};  
  
int main() {  
  C example;  
  
  return 0;  
}
```

