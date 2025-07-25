#### Classes & Objects

1. **Classes**
By default, all class members are declared `private`
```cpp
class School {  
// Can access directly outside
public:  
  std::string name;  
  int age;  
  
  void getName();
};

void School::getName(){
	std::cout << 'Name';
}
```

2. Objects
```cpp
School yeet;
yeet.name = "YeetCamp";  
yeet.age = 21;
yeet.getName;
//Expected Output: 'Name'
```

#### Access Specifiers:
|Access|`public`|`protected`|`private`|
|---|---|---|---|
|Inside the class|yes|yes|yes|
|Inside derived classes|yes|yes|no|
|Outside the class|yes|no|no|

[[Basics 3.1.2 - Encapsulation]]

#### Constructors & Destructors:
Constructor is called when object is instantiated
Destructor is called when object lifecycle is terminated
```cpp
class House {  
private:  
  std::string location;  
  int rooms;  
  
public:  
  // Constructor with default parameters  
  House(std::string loc = "New York", int num = 5) {  
    location = loc;  
    rooms = num;  
  }  
  
  void summary() {  
    std::cout<<location<<" house with "<<rooms<<" rooms"; 
  }  

  // Destructor  
  ~House() {  
    std::cout << "Moved away from " << location;  
  }
};


House('Florida') //works
House(5) //does not work because it skips a param
```

**Member Initializer List**
Initialize vals for constructor. Main Use: To initialize consts without using '=' 
```cpp
class Book {  
private:  
  const std::string title;  
  const int pages;  
public:  
  Book() : title("Diary"), pages(100) {} 
  //       ^^^^ Member initializer list  
};
```

[[Basics 3.1.5 - Inheritance]]

[[Basics 3.1.7 - Polymorphism]]
