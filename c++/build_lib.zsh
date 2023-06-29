g++ -c -std=c++11 -fPIC glauber.cpp -o glauberC.o
g++ -shared -o libGlauberC.so  glauberC.o 