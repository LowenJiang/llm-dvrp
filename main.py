from src import sim, nodify

# generate requests -> encode in a (V, A) network
requests = sim.simulation()
enc_net = nodify.create_network(requests)
node, matrix, requests = enc_net['map'], enc_net['distance'], enc_net['requests']
print(len(node))
print(matrix)
print(requests)

# network -> cost solver 
