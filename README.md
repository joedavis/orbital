# Quick and Dirty N-Body Newtonian Gravity Simulator

I was bored and decided to learn how to use scipy.odeint. This was the 
result. 

[Demo on YouTube](https://www.youtube.com/watch?v=dbfz2HLqo9w)

# Build and Running Instructions

Install `numpy`, `scipy`, `matplotlib` and `cython`. Then compile the 
cython modules with 

    python3 setup.py build_ext --inplace

To run the program, use:
       
    python3 -m orbital

