import random
import numpy as np
import matplotlib.pyplot as plt
import time





class Grid_MDP:

	def __init__(self,gridsize,points,Side_Wall_Coords,dst=None):
		self.N=gridsize
		V=[(i,j) for i in range (gridsize) for j in range (gridsize)]
		V1=[(e,f,0) for e in V for f in V]
		V2=[(e,e,1) for e in V]

		self.states=V1+V2
		self.depots=points
		self.Wall_pos=Side_Wall_Coords
		self.map={}
		

		if(dst==None):
			X=random.sample(range(0,len(points)),1)
			self.dest=(points[X[0]],points[X[0]],2)
		else:
			self.dest=(dst,dst,2)

		self.states.append(self.dest)

		L=self.states
		for i in range(0,len(L)):
			self.map[L[i]]=i


		self.Actions_list=["North","South","East","West","Pickup","Putdown"]
		self.act_map={}
		for i in range(0,len(self.Actions_list)):
			self.act_map[self.Actions_list[i]]=i

		


	def isEnd(self,state):
		if(state==self.dest):
			return True
		return False


	def get_Actions(self,state):
		if(state==self.dest):
			return []
		else:
			return self.Actions_list


	def is_Valid_Pos(self,x,y,action):
		if(x<0 or x>=self.N or y<0 or y>=self.N):
			return False
		else:
			if((action=="East" and (x-0.5,y) in self.Wall_pos) or (action=="West"  and (x+0.5,y) in self.Wall_pos) or (action=="North" and (x,y-0.5) in self.Wall_pos) or (action=="South" and (x,y+0.5) in self.Wall_pos)):
				return False
			else:
				return True






	def final_pos(self,x,y,action):
		x_tent=0
		y_tent=0
		z_tent=0
		if(action=="North"):
			(x_tent,y_tent)=(x,y+1)
		elif(action=="South"):
			(x_tent,y_tent)=(x,y-1)
		elif(action=="West"):
			(x_tent,y_tent)=(x-1,y)
		elif(action=="East"):
			(x_tent,y_tent)=(x+1,y)
		else:
			(x_tent,y_tent)=(x,y)

		if(self.is_Valid_Pos(x_tent,y_tent,action)):
			return (x_tent,y_tent)
		else:
			return (x,y)

		




	def get_Successor_State_Reward_Unrandomized(self,state,action):

		u=state[0]
		v=state[1]
		w=state[2]	
		(x,y)=(u[0],u[1])
		Transitions=[]

		p1=self.final_pos(x,y,"North")
		p2=self.final_pos(x,y,"East")
		p3=self.final_pos(x,y,"South")
		p4=self.final_pos(x,y,"West")

		if(w==0):
			pn=(p1,v,w)
			pe=(p2,v,w)
			ps=(p3,v,w)
			pw=(p4,v,w)
		else:
			pn=(p1,p1,w)
			pe=(p2,p2,w)
			ps=(p3,p3,w)
			pw=(p4,p4,w)

		
		if(action=="North"):
			Transitions.append((pn,0.85,-1))
			Y=[(ps,0.05,-1),(pw,0.05,-1),(pe,0.05,-1)]
			Transitions+=Y
		elif(action=="South"):
			Transitions.append((ps,0.85,-1))
			Y=[(pn,0.05,-1),(pw,0.05,-1),(pe,0.05,-1)]
			Transitions+=Y
		elif(action=="East"):
			Transitions.append((pe,0.85,-1))
			Y=[(pn,0.05,-1),(ps,0.05,-1),(pw,0.05,-1)]
			Transitions+=Y
		elif(action=="West"):
			Transitions.append((pw,0.85,-1))
			Y=[(pn,0.05,-1),(ps,0.05,-1),(pe,0.05,-1)]
			Transitions+=Y

		else:
			if(action=="Pickup"):
				u=state[0]
				v=state[1]
				w=state[2]
				if(u==v):
					new_state=(u,v,1)
					Transitions.append((new_state,1,-1))
				else:
					Transitions.append((state,1,-10))

			else:
				t_state=(self.dest[0],self.dest[1],1)
				if(state==t_state):
					Transitions.append((self.dest,1,20))

				else:
					if(u==v and state[2]==1):
						Transitions.append(((u,v,0),1,-1))
					else:
						Transitions.append(((u,v,state[2]),1,-10))


		
		return Transitions



	def get_succ_state(self,state,action):
		return self.get_Successor_State_Reward_Unrandomized(state,action)





class Value_Iteration:

	def __init__(self,MDP_Prob,discount_factor,eps):
		self.MDP_Prob=MDP_Prob
		self.disc=discount_factor
		self.epsilon=eps


	def value_iter(self,key=0):
		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		Value=np.zeros(m,dtype="float64")
		New_Value=np.zeros(m,dtype="float64")
		P=np.empty(m-1,dtype="U16")
		M=self.MDP_Prob.map
		iters=0
		req_data=[]
		while True:
			for e in self.MDP_Prob.states:
				temp=0
				Acts=self.MDP_Prob.get_Actions(e)
				if(not self.MDP_Prob.isEnd(e)):
					T=[sum([prob*(r+self.disc*Value[M[s1]]) for (s1,prob,r) in self.MDP_Prob.get_succ_state(e,act)]) for act in Acts]
					T1=np.array(T) 
					temp=np.max(T1)
					polcy=Acts[np.argmax(T1)]
					P[M[e]]=polcy
					New_Value[M[e]]=temp
					
					


			max_diff=np.amax(np.absolute(Value-New_Value))
			req_data.append(max_diff)
			if(max_diff<=self.epsilon):
				if(key==0):
					return (P,len(req_data))
				else:
					return req_data



			np.copyto(Value,New_Value)
			iters+=1





class Policy_Iteration:

	def __init__(self,MDP_Prob,discount_factor,eps):
		self.MDP_Prob=MDP_Prob
		self.disc=discount_factor
		self.epsilon=eps


	
	def policy_iter(self,key=0):
		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		Value=np.zeros(m,dtype="float64")		
		iters=0
		req_data=[]
		State_space=self.MDP_Prob.states
		Req_states=State_space[:-1]
		M=self.MDP_Prob.map
		P=np.empty(m-1,dtype="U16")
		P_new=np.empty(m-1,dtype="U16")
		for e in Req_states:
			P_new[M[e]]=random.choice(self.MDP_Prob.get_Actions(e))

		np.copyto(P,P_new)
		iters=0
		Policy_loss_calc=[]

		while True:
			Updated_val=self.policy_evalaution(Req_states,P,Value)
			Policy_loss_calc.append(Updated_val)
			for e in Req_states:
				temp=0
				Acts=self.MDP_Prob.get_Actions(e)
				T=[sum([prob*(r+self.disc*Updated_val[M[s1]]) for (s1,prob,r) in self.MDP_Prob.get_succ_state(e,act)]) for act in Acts]
				T1=np.array(T) 
				temp=np.max(T1)
				polcy=Acts[np.argmax(T1)]
				P_new[M[e]]=polcy


			if((P==P_new).all()):
				if(key==0):
					return (P,len(Policy_loss_calc))
				else:
					U=[]
					d1=0
					f=np.array(Updated_val)
					for e in Policy_loss_calc:
						d1=np.amax(np.absolute(np.array(e)-f))
						U.append(d1)
					return U

				

			np.copyto(P,P_new)
			iters+=1
							
		




	def policy_evalaution(self,R,P,Val):
		n=(self.MDP_Prob).N
		M=self.MDP_Prob.map
		m=n**4+n**2+1
		New_Value=np.zeros(m,dtype="float64")
		Value=np.zeros(m,dtype="float64")
		np.copyto(Value,Val)
		iters=0
		while True:
			for e in R:
				temp=0	
				corr_act=P[M[e]]
				temp=sum([prob*(r+self.disc*Value[M[s1]]) for (s1,prob,r) in self.MDP_Prob.get_succ_state(e,corr_act)])
				New_Value[M[e]]=temp



			max_diff=np.amax(np.absolute(Value-New_Value))
			if(max_diff<=self.epsilon):
				return New_Value


			np.copyto(Value,New_Value)
			iters+=1




class Policy_Iteration_Linalg:

	def __init__(self,MDP_Prob,discount_factor,eps):
		self.MDP_Prob=MDP_Prob
		self.disc=discount_factor
		self.epsilon=eps


	
	def policy_iter(self,key=0):
		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		Value=np.zeros(m,dtype="float64")		
		iters=0
		req_data=[]
		State_space=self.MDP_Prob.states
		Req_states=State_space[:-1]
		M=self.MDP_Prob.map
		P=np.empty(m-1,dtype="U16")
		P_new=np.empty(m-1,dtype="U16")
		for e in Req_states:
			P_new[M[e]]=random.choice(self.MDP_Prob.get_Actions(e))

		np.copyto(P,P_new)
		iters=0
		Policy_loss_calc=[]

		while True:
			Updated_val=self.policy_evalaution(Req_states,P,Value)
			Policy_loss_calc.append(Updated_val)
			for e in Req_states:
				temp=0
				Acts=self.MDP_Prob.get_Actions(e)
				T=[sum([prob*(r+self.disc*Updated_val[M[s1]]) for (s1,prob,r) in self.MDP_Prob.get_succ_state(e,act)]) for act in Acts]
				T1=np.array(T) 
				temp=np.max(T1)
				polcy=Acts[np.argmax(T1)]
				P_new[M[e]]=polcy


			if((P==P_new).all()):
				if(key==0):
					return (P,len(Policy_loss_calc))
				else:
					U=[]
					d1=0
					f=np.array(Updated_val)
					for e in Policy_loss_calc:
						d1=np.amax(np.absolute(np.array(e)-f))
						U.append(d1)
					return U

				

			np.copyto(P,P_new)
			iters+=1
							
		




	def policy_evalaution(self,R,P,Val):
		n=(self.MDP_Prob).N
		M=self.MDP_Prob.map
		m=n**4+n**2+1
		A=np.zeros((m,m),dtype="float64")
		b=np.zeros(m,dtype="float64")
		for i in range (0,m):
			A[i][i]=1

		for e in R:
			corr_act=P[M[e]]
			U0=self.MDP_Prob.get_succ_state(e,corr_act)
			for z in U0:
				b[M[e]]+=z[2]*z[1]
				A[M[e]][M[z[0]]]-=self.disc*z[1]


		

		X=np.dot(np.linalg.inv(A),b.reshape(m,))
		return X.reshape(m).tolist()

		


# ###################################################################################




class Q_Learning:

	def __init__(self,MDP_Prob,explor_rate,learn,discount_factor,eps,dec=None,iters=500):

		self.MDP_Prob=MDP_Prob
		self.disc=discount_factor
		self.explor_rate=explor_rate
		self.learning_rate=learn
		self.disc=discount_factor
		self.epsilon=eps
		self.iters=iters

		if(dec==None):
			self.decay=False
		else:
			self.decay=True


	def computed(self,L):
		s=0
		for i in range(len(L)-1,-1,-1):
			s=s*self.disc+L[i]
		return s

	



	def epsilon_greedy_choose(self,state,Val,explor_rate):
		A=self.MDP_Prob.get_Actions(state)
		M=self.MDP_Prob.map
		M1=self.MDP_Prob.Actions_list
		l=len(A)
		Weights=[1-explor_rate]+[(explor_rate)/(l-1)]*(l-1)				
		act1=M1[np.argmax(Val[M[state]])]
		Acts=[act1]
		for e in A:
			if(e!=act1):
				Acts.append(e)

		act=random.choices(Acts,Weights,k=1)[0]
		return act





	def Q_val_iteration(self,episodes,start_state=None):
		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		QA_Val=np.zeros((m,6),dtype="float64")
		New_QA_Val=np.zeros((m,6),dtype="float64")
		M=self.MDP_Prob.map
		M1=self.MDP_Prob.act_map


		QA_Val=np.zeros((m,6),dtype="float64")
		R=[]

		if(start_state==None):
			for j in range(0,episodes):
				QA_Val=self.episodic_iteration(QA_Val)
				if((j%80==0)):
					R.append(self.Evaluate(QA_Val))
		else:
			for j in range(0,episodes):
				QA_Val=self.episodic_iteration(QA_Val)
				if((j%80==0)):
					R.append(self.Evaluate(QA_Val,start_state))




		Policy=[None]*(m)
		for i in range(0,m-1):
			Policy[i]=self.MDP_Prob.Actions_list[np.argmax(QA_Val[i])]


		return (R,Policy)






	def episodic_iteration(self,QA_Val):

		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		M=self.MDP_Prob.map
		M1=self.MDP_Prob.act_map


		state=random.choice(self.MDP_Prob.states[:-1])
		steps=0
		alpha=self.learning_rate
		# =a0
		R=[]
		explor_rate=self.explor_rate
		

		while(True):

			if(self.MDP_Prob.isEnd(state) or steps==self.iters):
				return QA_Val


			act=None
			if(steps==0):
				act=random.choice(self.MDP_Prob.get_Actions(state))
			else:
				act=self.epsilon_greedy_choose(state,QA_Val,explor_rate)




			T=self.MDP_Prob.get_succ_state(state,act)
			u=random.choice(T)


			i1=M[state]
			j1=M1[act]

			fin_state=u[0]
			prob=u[1]
			rew=u[2]

				
			temp=self.disc*np.max(QA_Val[M[fin_state]])+rew
			QA_Val[i1][j1]=(1-alpha)*QA_Val[i1][j1]+(alpha)*temp
			state=fin_state
			
				
			steps+=1
			if(self.decay):
				explor_rate=(self.explor_rate/(steps+1))

		



	def Evaluate(self,V,start_state=None):
		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		M=self.MDP_Prob.map
		M1=self.MDP_Prob.act_map	
		P=np.argmax(V,axis=1).tolist()
		P=[self.MDP_Prob.Actions_list[e] for e in P]
		r=0
		for j in range(0,10):
			R1=[]
			if(start_state==None):
				state=random.choice(self.MDP_Prob.states)
			else:
				state=start_state
			steps=0
			while(not (self.MDP_Prob.isEnd(state) or steps==500)):
				act=P[M[state]]
				T=self.MDP_Prob.get_succ_state(state,act)
				u=random.choice(T)
				state=u[0]
				R1.append(u[2])
				steps+=1

			r+=self.computed(R1)


		return 0.1*r


################################################################################################################################################

class SARSA:

	def __init__(self,MDP_Prob,explor_rate,learn,discount_factor,eps,dec=None,iters=500):

		self.MDP_Prob=MDP_Prob
		self.disc=discount_factor
		self.explor_rate=explor_rate
		self.learning_rate=learn
		self.disc=discount_factor
		self.epsilon=eps
		self.iters=iters

		if(dec==None):
			self.decay=False
		else:
			self.decay=True


	def computed(self,L):
		s=0
		for i in range(len(L)-1,-1,-1):
			s=s*self.disc+L[i]
		return s



	def epsilon_greedy_choose(self,state,Val,explor_rate):
		A=self.MDP_Prob.get_Actions(state)
		M=self.MDP_Prob.map
		M1=self.MDP_Prob.Actions_list
		l=len(A)
		Weights=[1-explor_rate]+[(explor_rate)/(l-1)]*(l-1)				
		act1=M1[np.argmax(Val[M[state]])]
		Acts=[act1]
		for e in A:
			if(e!=act1):
				Acts.append(e)

		act=random.choices(Acts,Weights,k=1)[0]
		return act





	def SARSA_iteration(self,episodes,start_state=None):
		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		QA_Val=np.zeros((m,6),dtype="float64")
		New_QA_Val=np.zeros((m,6),dtype="float64")
		M=self.MDP_Prob.map
		M1=self.MDP_Prob.act_map


		QA_Val=np.zeros((m,6),dtype="float64")
		R=[]


		if(start_state==None):
			for j in range(0,episodes):
				QA_Val=self.episodic_iteration(QA_Val)
				if((j%80==0)):
					R.append(self.Evaluate(QA_Val))
		else:
			for j in range(0,episodes):
				QA_Val=self.episodic_iteration(QA_Val)
				if((j%80==0)):
					R.append(self.Evaluate(QA_Val,start_state))

		Policy=[None]*(m)
		for i in range(0,m-1):
			Policy[i]=self.MDP_Prob.Actions_list[np.argmax(QA_Val[i])]


		return (R,Policy)





	def episodic_iteration(self,QA_Val):

		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		M=self.MDP_Prob.map
		M1=self.MDP_Prob.act_map

		

		state=random.choice(self.MDP_Prob.states[:-1])
		steps=0
		s=0
		act=random.choice(self.MDP_Prob.get_Actions(state))
		alpha=self.learning_rate
		explor_rate=self.explor_rate



		while(True):

			if(self.MDP_Prob.isEnd(state) or steps==self.iters):
				return QA_Val

			if(steps!=0):
				act=self.epsilon_greedy_choose(state,QA_Val,explor_rate)
			else:
				act=random.choice(self.MDP_Prob.get_Actions(state))


			
			
			T=self.MDP_Prob.get_succ_state(state,act)
			u=random.choice(T)


			i1=M[state]
			j1=M1[act]

			fin_state=u[0]
			prob=u[1]
			rew=u[2]


			act_temp=self.epsilon_greedy_choose(fin_state,QA_Val,explor_rate)



			temp=self.disc*QA_Val[M[fin_state]][M1[act_temp]]+rew
			QA_Val[i1][j1]=(1-alpha)*QA_Val[i1][j1]+(alpha)*temp
			state=fin_state
			act=act_temp
			
			
		
			steps+=1
			if(self.decay):
				explor_rate=(self.explor_rate/(steps+1))

			



	def Evaluate(self,V,start_state=None):
		n=(self.MDP_Prob).N
		m=n**4+n**2+1
		M=self.MDP_Prob.map
		M1=self.MDP_Prob.act_map	
		P=np.argmax(V,axis=1).tolist()
		P=[self.MDP_Prob.Actions_list[e] for e in P]
		r=0
		for j in range(0,10):
			R1=[]
			if(start_state==None):
				state=random.choice(self.MDP_Prob.states)
			else:
				state=start_state
			steps=0
			while(not (self.MDP_Prob.isEnd(state) or steps==500)):
				act=P[M[state]]
				T=self.MDP_Prob.get_succ_state(state,act)
				u=random.choice(T)
				state=u[0]
				R1.append(u[2])
				steps+=1

			r+=self.computed(R1)


		return 0.1*r








#########################################################################################################################################

def prob_A_2a():
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)])	
	V=Value_Iteration(Taxi_Car_small,0.9,1e-10)
	ans_1=V.value_iter()
	print("Epsilon chosen was 1e-10")
	print("Number of Iterations for Convergence are: "+str(ans_1[1]))
	
	



def prob_A_2b():
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)])
	Disc=[0.01,0.1,0.5,0.8,0.99]
	L=[]
	for din in Disc:
		V=Value_Iteration(Taxi_Car_small,din,1e-10)
		Y_val=np.array(V.value_iter(1))
		X_val=np.array([i+1 for i in range(len(Y_val))])
		plt.plot(X_val,Y_val,label='H')
		plt.xlabel("Number of iterations")
		plt.ylabel("Max-Norm")
		plt.title("Plot of Max-Norm vs Iterations with discount-factor "+str(din))
		plt.savefig(str(din)+" discount_factor value_iteration.png")
		plt.close() 



def prob_A_2c():
	
	Disc=[0.1,0.99]
	pass_desti=(4,4)
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)],pass_desti)
	M=Taxi_Car_small.map

	Mr=[((0,0),(0,4)),((0,0),(3,0)),((3,0),(0,4))]

	for (taxi_start,pass_start) in Mr:

		ini_state=(taxi_start,pass_start,0)
		fin_state=(pass_desti,pass_desti,2)

		for din in Disc:
			print("First 20 steps till episode completion with discount-factor "+str(din)+" for problem instance with initial positions of taxi and passenger at "+str((taxi_start,pass_start))+" are:")
			V=Value_Iteration(Taxi_Car_small,din,1e-10)
			ans=V.value_iter()
			P=ans[0]
			count=0
			state=ini_state
			while(count<20 and state!=fin_state):
				ind=M[state]
				act=P[ind]
				print(act)
				Trans=Taxi_Car_small.get_succ_state(state,act)[0]
				state=Trans[0]
				count+=1

			


		


def prob_A_3a():
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)],(4,4))
	Disc=[0.01,0.3,0.5,0.8,0.99]
	Y1=[]
	Y2=[]
	for din in Disc:
		V=Policy_Iteration_Linalg(Taxi_Car_small,din,1e-10)
		V1=Policy_Iteration(Taxi_Car_small,din,1e-10)
		t1=time.time()
		V.policy_iter()
		t2=time.time()
		V1.policy_iter()
		t3=time.time()
		Y1.append(t2-t1)
		Y2.append(t3-t2)

	X=Disc
	plt.plot(X,Y1,label='With Gaussian Elimination')
	plt.plot(X,Y2,label='With Fixed Point Iteration')
	plt.xlabel("Discount factor")
	plt.ylabel("Time taken in seconds")
	plt.title("Plot of time-taken vs discount-factor")
	plt.legend()
	plt.savefig("plot_A_3a.png")
	plt.close() 






def prob_A_3b():
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)])
	Disc=[0.01,0.1,0.5,0.8,0.99]
	L=[]
	for din in Disc:
		V=Policy_Iteration(Taxi_Car_small,din,1e-10)
		Y_val=np.array(V.policy_iter(1))
		X_val=np.array([i+1 for i in range(len(Y_val))])
		plt.plot(X_val,Y_val,label='H')
		plt.xlabel("Number of iterations")
		plt.ylabel("Policy-Loss")
		plt.title("Plot of Policy-Loss vs Iterations with discount-factor "+str(din))
		plt.savefig(str(din)+" discount_factor policy_iteration.png")
		plt.close() 




def prob_B_2():
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)])
	V=Q_Learning(Taxi_Car_small,0.1,0.25,0.99,1e-10)
	V1=Q_Learning(Taxi_Car_small,0.1,0.25,0.99,1e-10,True)
	V2=SARSA(Taxi_Car_small,0.1,0.25,0.99,1e-10)
	V3=SARSA(Taxi_Car_small,0.1,0.25,0.99,1e-10,True)
	Y1=V.Q_val_iteration(10000)[0]
	X_val=[80*i+1 for i in range(len(Y1))]
	plt.plot(X_val,Y1)
	plt.xlabel("Number of episodes")
	plt.ylabel("Discounted Accumulated Reward")
	plt.title("Plot for Q_Learning")
	plt.savefig("Q_Learning.png")
	plt.close() 
	Y2=V1.Q_val_iteration(10000)[0]
	X_val=[80*i+1 for i in range(len(Y2))]
	plt.plot(X_val,Y2)
	plt.xlabel("Number of episodes")
	plt.ylabel("Discounted Accumulated Reward")
	plt.title("Plot for Q_Learning with decay")
	plt.savefig("Decaying_Q_Learning.png")
	plt.close() 
	Y3=V2.SARSA_iteration(10000)[0]
	X_val=[80*i+1 for i in range(len(Y3))]
	plt.plot(X_val,Y3)
	plt.xlabel("Number of episodes")
	plt.ylabel("Discounted Accumulated Reward")
	plt.title("Plot for SARSA")
	plt.savefig("SARSA.png")
	plt.close() 
	Y4=V3.SARSA_iteration(10000)[0]
	X_val=[80*i+1 for i in range(len(Y4))]
	plt.plot(X_val,Y4)
	plt.xlabel("Number of episodes")
	plt.ylabel("Discounted Accumulated Reward")
	plt.title("Plot for SARSA with decay")
	plt.savefig("Decaying_SARSA.png")
	plt.close() 



def prob_B_3():
	
	pass_desti=(4,4)
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)],pass_desti)
	M=Taxi_Car_small.map

	Mr=[((0,0),(0,4)),((0,0),(3,0)),((0,0),(4,4))]

	for e in Mr:
		ini_state=(e[0],e[1],0)
		fin_state=(pass_desti,pass_desti,2)		
		V=SARSA(Taxi_Car_small,0.1,0.25,0.99,1e-10,True)
		Y=V.SARSA_iteration(10000,start_state=ini_state)[0]
		X_val=[80*i+1 for i in range(len(Y))]
		plt.plot(X_val,Y)
		plt.xlabel("Number of episodes")
		plt.ylabel("Discounted Accumulated Reward")
		plt.title("Passenger at "+str(e[1])+" Decaying_SARSA")
		plt.savefig("Passenger at "+str(e[1])+" Decaying_SARSA.png")
		plt.close() 
		

	



def prob_B_4_i():
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)])
	E=[0,0.05,0.1,0.5,0.9]
	for exo in E:		
		V=Q_Learning(Taxi_Car_small,exo,0.1,0.99,1e-10)
		Y1=V.Q_val_iteration(10000)[0]
		X_val=[80*i+1 for i in range(len(Y1))]
		plt.plot(X_val,Y1)
		plt.xlabel("Number of episodes")
		plt.ylabel("Discounted Accumulated Reward")
		plt.title("Plot of epsilon "+str(exo))
		plt.savefig(str(exo)+" varying_epsilon.png")
		plt.close() 



def prob_B_4_ii():
	Taxi_Car_small=Grid_MDP(5,[(0,0),(3,0),(0,4),(4,4)],[(0.5,0),(0.5,1),(2.5,0),(2.5,1),(1.5,3),(1.5,4)])
	E=[0.1,0.2,0.3,0.4,0.5]
	for lrn in E:		
		V=Q_Learning(Taxi_Car_small,0.1,lrn,0.99,1e-10)
		Y1=V.Q_val_iteration(10000)[0]
		X_val=[80*i+1 for i in range(len(Y1))]
		plt.plot(X_val,Y1)
		plt.xlabel("Number of episodes")
		plt.ylabel("Discounted Accumulated Reward")
		plt.title("Plot of alpha "+str(lrn))
		plt.savefig(str(lrn)+" varying_alpha.png")
		plt.close() 



def prob_B_5():
	Depos=[(0,9),(0,1),(3,6),(4,0),(6,5),(5,9),(8,9),(9,0)]
	Walls=[(0.5,0),(0.5,1),(0.5,2),(0.5,3),(3.5,0),(3.5,1),(3.5,2),(3.5,3),(7.5,0),(7.5,1),(7.5,2),(7.5,3),(7.5,6),(7.5,7),(7.5,8),(7.5,9),
	(2.5,9),(2.5,8),(2.5,7),(2.5,6),(5.5,4),(5.5,5),(5.5,6),(5.5,7)]
	pass_desti=(3,6)
	Taxi_Car_large=Grid_MDP(10,Depos,Walls,pass_desti)
	

	Mr=[((0,1),(0,9)),((0,1),(6,5)),((0,1),(9,0)),((0,1),(5,9)),((0,1),(8,9))]

	for e in Mr:
		ini_state=(e[0],e[1],0)
		fin_state=(pass_desti,pass_desti,2)		
		V=SARSA(Taxi_Car_large,0.1,0.25,0.5,1e-10,True,2000)
		Y=V.SARSA_iteration(2000,start_state=ini_state)[0]
		X_val=[80*i+1 for i in range(len(Y))]
		plt.plot(X_val,Y)
		plt.xlabel("Number of episodes")
		plt.ylabel("Discounted Accumulated Reward")
		plt.title("Passenger at "+str(e[1])+" SARSA_Decay")
		plt.savefig("Passenger at "+str(e[1])+" SARSA.png")
		plt.close() 







###############################MAIN_FUNCTION####################################

def main():
	part=str(input("Please enter the part of assignment: "))
	qno=str(input("Please enter the question number: "))
	

	if(part=="A"):

		if(qno=="2a"):
			prob_A_2a()
		elif(qno=="2b"):
			prob_A_2b()
		elif(qno=="2c"):
			prob_A_2c()
		elif(qno=="3a"):
			prob_A_3a()
		elif(qno=="3b"):
			prob_A_3b()
		elif(qno=="1"):
			print("Taxi_MDP object already written")
		else:
			print("No such question and subpart")



	elif(part=="B"):
		if(qno=="2"):
			prob_B_2()
			
		elif(qno=="1"):
			print("Q-Learning and SARSA Implemnted")
		

		elif(qno=="4"):
			prob_B_4_i()
			prob_B_4_ii()

		elif(qno=="3"):
			prob_B_3()

		elif(qno=="5"):
			prob_B_5()


		else:
			print("No such question and subpart")



	else:
		print("Invalid Input for part of assignment")







main()
