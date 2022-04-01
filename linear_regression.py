

def lr(b0, b1, X):
    print("Linear Regression with One variable.")

    #b0 = the intercept
    #b1 = the slope
    #X = what do you want to predict
    #Y = the result
    Y = b0 + b1 * X
    return Y

def lr_multiple(b0, b1, x1, b2, x2, b3, x3, b4, x4):
    print("Linear regression with multiple variables selected.")

    # b0 = intercept (X=0)
    # b1 = the coef or parameter of x1
    # b2 = the coef or parameter of x2 and so on..

    Yhat = b0 + b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4

    return Yhat





def main():
    print("What do you want to do?")
    print("1. Linear Regression with one variable.\n2. Linear Regression with multiple variables(4 variables.).\n3. Classification Problem")
    choice = int(input("Escolha sua opção: "))

    if choice == 1:
        print("Linear Regression with one variable selected.")
        print("Please submit the required inputs below.")
        
        b0 = int(input("The b0: "))
        b1 = int(input("The b1: "))
        X = int(input("And then the X: "))
        
        Y = lr(b0, b1, X)
        
        print("Your output is {}".format(Y))

    if choice == 2:
        print("Linear Regression with multiple variables selected.")
        print("Please submit the required inputs below.")
        
        b0 = int(input("The b0: "))
        b1 = int(input("The b1: "))
        x1 = int(input("The x1: "))
        b2 = int(input("The b2: "))
        x2 = int(input("The x2: "))
        b3 = int(input("The b3: "))
        x3 = int(input("The x3: "))
        b4 = int(input("The b4: "))
        x4 = int(input("The x4: "))
        
        Yhat = lr_multiple(b0, b1, x1, b2, x2, b3, x3, b4, x4)
        
        print("Your output is {}".format(Yhat))


if __name__ == '__main__':
    main()