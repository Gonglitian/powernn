from Tensor import *
# training data
x = Tensor(np.random.normal(0, 1.0, (100, 3)))
coef = Tensor(np.random.randint(0, 10, (3,)))
y = x * coef - 3 

params = {
    "w": Tensor(np.random.normal(0, 1.0, (3, 3)), requires_grad=True),
    "b": Tensor(np.random.normal(0, 1.0, 3), requires_grad=True)
}

learng_rate = 3e-4
loss_list = []
for e in range(100):
    # set gradient to zero
    for param in params.values():
        param.zero_grad()
    
    # forward
    predicted = x @ params["w"] + params["b"]
    err = predicted - y
    loss = (err * err).sum()
    
    # backward automatically
    loss.backward()
    
    # updata parameters
    for param in params.values():
        param -= learng_rate * param.grad
        
    loss_list.append(loss.values)
    if e % 10 == 0:
        print("epoch-%i \tloss: %.4f" % (e, loss.values))