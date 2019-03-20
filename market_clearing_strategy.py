import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
import sys



def rank_sellers(buyer,
	valuations,
	prices,
	preference):
	

	gain = np.zeros_like(prices)
	for s,price in enumerate(prices):
		gain[s] = valuations[buyer,s] - price

	
	best_gain = np.max(gain)
	best_sellers = np.where(gain == best_gain)[0]
	for bs in best_sellers:
		preference[buyer,bs] = 1.
	
	return preference



def find_r_set(num_sellers,
	num_buyers,
	preference,
	prices):
	print preference
	print prices
	new_prices = np.copy(prices)
	s = []
	n_s = []
	edge_list = []
	for seller in range(num_sellers):
		for buyer in range(num_buyers):
			if preference[buyer,seller] == 1.:
				edge_list.append((buyer,seller))

	new_preference = np.copy(preference)
	while True:
		edge_to_remove = []
		for (buyer,seller) in edge_list:
			seller_rank = sum(preference[:,seller])
			buyer_rank = sum(preference[buyer,:])
			if seller_rank > 1.:
				if buyer_rank > 1.:
					alternative_sellers = np.where(preference[buyer,:] == 1.)[0].tolist()
					alternative_sellers = [alt_s for alt_s in alternative_sellers if alt_s != seller]
					for alt_s in alternative_sellers:
						alt_seller_rank = sum(preference[:,alt_s])
						if alt_seller_rank == 1.:
							new_preference[buyer,seller] = 0.
							edge_to_remove.append((buyer,seller))

		edge_list = [edge for edge in edge_list if edge not in edge_to_remove]
		preference = np.copy(new_preference)
		if len(edge_to_remove) == 0:
			break

	for seller in range(num_sellers): 
		if sum(preference[:,seller]) > 1.:
			buyers = np.where(preference[:,seller] == 1.)[0].tolist()
			s += buyers 


	s = list(set(s))
	for buyer in s:
		sellers = np.where(preference[buyer,:] == 1.)[0].tolist()
		n_s += sellers
	
	
	n_s = list(set(n_s))
	print preference
	print 'S:',s,'N(S):',n_s
	if len(s) > len(n_s):
		for seller in n_s:
			new_prices[seller] += 1

	
	return new_prices, preference




def clear_market(num_sellers,
	num_buyers,
	valuations,
	demo_mode=False):



	prices = np.zeros((num_sellers))
	preference = np.zeros((num_sellers,num_buyers))
	matching = np.zeros((num_sellers,num_buyers))
	perfect_match = False

	for i in range(1000):
		for b in range(num_buyers):
			preference = rank_sellers(b,valuations,prices,preference)

		if demo_mode:
			print 'iter:',i
			print 'preferences:',preference
			print 'old prices:',prices
		
		old_preference = preference
		new_prices,matching = find_r_set(num_sellers,num_buyers,preference,prices)
		#matching = np.copy(preference)

		if demo_mode:
			print 'matching:',matching
			print 'new_prices:',new_prices
			raw_input('press key to continue')

		if np.array_equal(prices,new_prices):
			perfect_match = True
			break
		else:
			prices = np.copy(new_prices)

		
		# If min price > 0 then we want to reduce everything by min price.
		min_price = np.min(prices)
		if min_price > 0:
			prices -= min_price
		
		preference = np.zeros((num_sellers,num_buyers))
		matching = np.zeros((num_sellers,num_buyers))

	return prices,old_preference,matching,perfect_match,i




def draw_graph(matching):
	num_buyers = matching.shape[0]
	num_sellers = matching.shape[1]
	node_list_buyers = [i for i in range(num_buyers)]
	node_list_sellers = [i+num_buyers for i in range(num_sellers)]
	edge_list = []
	for i in range(num_buyers):
		for j in range(num_sellers):
			if matching[i,j] == 1.:
				edge_list.append((i,num_buyers+j+1))

	print node_list_buyers, node_list_sellers, edge_list
	
	print node_list_buyers, node_list_sellers, edge_list

	G = nx.Graph()
	G.add_nodes_from([1, 2, 3, 4], bipartite=0)
	G.add_nodes_from([5, 6, 7], bipartite=1)

	G.add_edges_from([(1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (4, 5)])
	#G.add_edges_from(edge_list)
	print nx.is_connected(G)
	x,y = nx.bipartite.sets(G)
	pos = dict()
	pos.update( (n, (1,i)) for i,n in enumerate(x))
	pos.update( (n, (2,i)) for i,n in enumerate(y))
	nx.draw(G, pos=pos, with_labels=True)
	plt.show()
	return



def evaluation():

	success = 0
	fail = 0
	total_iterations = 0
	s_time = time.time()
	num_iter = []
	conv_times = []
	for j in range(2):
		start_time = time.time()
		num_sellers = 3
		num_buyers = 3

		assert(num_sellers >= num_buyers)

		
		valuations = np.array([[12,4,2],
			[8,7,6],
			[7,5,2]])
		
		'''

		valuations = np.random.randint(low=1,high=40,size=(num_sellers,num_buyers))
		
		valuations = np.array([[ 2, 33, 30,  5, 15],
		 [34,  7, 14, 20, 33],
		 [21, 20, 39,  7,  6],
		 [23, 18, 30,  4, 14],
		 [21, 15, 29,  5, 22]])
		
		
		valuations = np.array([[3,1,1],
			[2,1,2],
			[4,2,3]])
		
		
		valuations = np.array([[24, 29, 38,  3, 20, 10, 13],
		 [ 3, 17,  1, 15, 34, 16, 25],
		 [34, 25, 34, 32, 29, 17,  8],
		 [36, 38, 38, 36, 31,  6,  7],
		 [12,  2, 24, 26, 36, 29, 10],
		 [20, 20, 37, 24, 13,  1, 16],
		 [24, 28, 34,  7, 30,  3, 21]])
		'''
		
		print valuations

		prices,preference,matching,perfect_match,i = clear_market(num_sellers,num_buyers,valuations)


		if perfect_match:
			print 'Finished in iterations:',i
			total_iterations += i
			num_iter.append(i)
			end_time = time.time()
			conv_times.append((end_time - start_time))

			print prices
			print preference
			print matching
			success += 1
		else:
			print 'Did not converge'
			fail += 1

	e_time = time.time()

	dims = np.arange(len(num_iter))
	dims += 2

	#print dims,num_iter

	plt.plot(dims,num_iter)
	
	plt.xlabel('number of buyers/sellers')
	plt.ylabel('convergence in iterations')

	plt.show()

	plt.plot(dims,conv_times)
	
	plt.xlabel('number of buyers/sellers')
	plt.ylabel('convergence in seconds')

	plt.show()
	print 'Successful:',success,'Failed:',fail 
	print 'Average convergence:',(total_iterations / float(j+1)),'iterations'
	print 'Average convergence:',((e_time - s_time) / float(j+1)),'seconds'

	return







def demo_1():
	'''
	'''
	num_sellers = 5
	num_buyers = 5


	
	valuations = np.array([[ 2, 33, 30,  5, 15],
	 [34,  7, 14, 20, 33],
	 [21, 20, 39,  7,  6],
	 [23, 18, 30,  4, 14],
	 [21, 15, 29,  5, 22]])
	
	
	print valuations

	raw_input('press key to continue')

	prices,preference,matching,perfect_match,i = clear_market(num_sellers,num_buyers,valuations,demo_mode=True)


	if perfect_match:
		print 'Finished in iterations:',i
		print 'final prices:',prices
		print 'final preferences:',preference
		print 'final matching:',matching
	else:
		print 'Did not converge'



	return








if __name__ == '__main__':
	mode = 'eval'
	if mode == 'eval':
		evaluation()
	elif mode == 'demo':
		demo_1()
