#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;

template <typename T>
void test(T a){
    if(typeid(T) == typeid(float))  cout<<"float"<<endl;
    if(typeid(T) == typeid(int))    cout<<"int"<<endl;
}

int main(){
    float a = 0;
    int b = 0;
    test(a);
    test(b);
}