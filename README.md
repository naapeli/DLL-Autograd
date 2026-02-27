Compile c++ with the following command:

cmake -S . -B build -G "MinGW Makefiles" && cmake --build build


To run the tests, use

python -m pytest Tests/
