### **Getter(Accessor) Functions**

```cpp
class Clock {  
private:  
  int time = 1200;  
  
public:  
  // Getter function for time  
  int getTime() {  
    return time;  
  }  
};  
  
int main() {  
  Clock alarm;  
  std::cout << alarm.getTime(); // Output: 1200  
}
```

### **Setter (Mutator) Functions**
```cpp
class Clock {  
private:  
  int time = 1200;  
  
public:  
  // Getter function for time  
  int getTime() {  
    return time;  
  }  
  
  // Setter function for time  
  void setTime(int new_time) {  
    time = new_time;  
  }  
};  
  
int main() {  
  Clock alarm;  
  alarm.setTime(930);  
  std::cout << alarm.getTime(); // Output: 930  
}
```