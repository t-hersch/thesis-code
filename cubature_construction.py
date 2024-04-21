import numpy as np
from free_lie_algebra import *

def lie_sum(A):
  total = Elt([{}])
  for a in A:
    total += a

  return total

def truncateToLevel(e, level):
  elt_data = e.data[:2*level+1]
  for i in range(len(elt_data)):
    i_data = elt_data[i]
    for key in i_data.keys():
      total_d = i + key.letters.count(0)
      if total_d > level:
        e -= Elt([{}]*i+[{key : i_data[key]}])
        #i_data[key] = 0
  return e

def gaussian_cubature(d):
  if d == 1:
    z = [[0], [0.958572464613819], [2.02018287045609], [-0.958572464613819], [-2.02018287045609]]
    weights = [0.945308720482942, 0.393619323152241, 0.0199532420590459, 0.393619323152241, 0.0199532420590459]
    z = [[np.sqrt(2)*y for y in x] for x in z]
    weights = [x*(1/np.sqrt(np.pi)) for x in weights]

    return z, weights
  elif d in range(2,8):
    if d == 2:  # the table of values given in Stroud
      eta = 0.446103183094540
      lam = 1.36602540378444
      xi = -0.366025403784439
      mu =  1.98167882945871
      gam = 0
      A =   0.328774019778636
      B =   1/12
      C =   0.00455931355469736
    elif d == 3:
      eta = 0.476731294622796
      lam = 0.935429018879534
      xi = -0.731237647787132
      mu =  0.433155309477649
      gam = 2.66922328697744
      A =   0.242000000000000
      B =   0.081000000000000
      C =   0.005000000000000
    elif d == 4:
      eta = 0.523945658287507
      lam = 1.19433782552719
      xi = -0.398112608509063
      mu = -0.318569372920112
      gam = 1.85675837424096
      A =   0.155502116982037
      B =   0.0777510584910183
      C =   0.00558227484231506
    elif d == 5:
      eta = 2.14972564378798
      lam = 4.64252986016289
      xi = -0.623201054093728
      mu = -0.447108700673434
      gam = 0.812171426076331
      A =   0.000487749259189752
      B =   0.000487749259189752
      C =   0.0497073504444862
    elif d == 6:
      eta = 1.000000000000000
      lam = np.sqrt(2)
      xi =  0.000000000000000
      mu = -1.000000000000000
      gam = 1.000000000000000
      A =   0.007812500000000
      B =   0.062500000000000
      C =   0.007812500000000
    elif d == 7:
      eta = 0.000000000000000
      lam = 0.959724318748357
      xi = -0.772326488820521
      mu = -1.41214270131942
      gam = 0.319908106249452
      A =   1/9
      B =   1/72
      C =   1/72
    z = [] # initiate variables and append values
    weights = []

    z.append([eta]*d)
    weights.append(A)

    z.append([-eta]*d)
    weights.append(A)
    
    for i in range(d):
      z.append([xi]*i+[lam]+[xi]*(d-i-1))
      weights.append(B)

      z.append([-xi]*i+[-lam]+[-xi]*(d-i-1))
      weights.append(B)

      for j in range(i+1, d):
        z.append([gam]*i+[mu]+[gam]*(j-i-1)+[mu]+[gam]*(d-j-1))
        weights.append(C)

        z.append([-gam]*i+[-mu]+[-gam]*(j-i-1)+[-mu]+[-gam]*(d-j-1))
        weights.append(C)

    z = [[np.sqrt(2)*y for y in x] for x in z]
    return z, weights
  elif d >= 8:
    r = np.sqrt((d+2)/2)
    s = np.sqrt((d+2)/4)
    A = 2/(d+2)
    B = (4-d)/(2*(d+2)**2)
    C = 1/((d+2)**2)

    z = [[0]*d] # initiate variables and append values
    weights = [A]

    for i in range(d):
      z.append([0]*i+[r]+[0]*(d-i-1))
      weights.append(B)

      z.append([0]*i+[-r]+[0]*(d-i-1))
      weights.append(B)

      for j in range(i+1, d):
        z.append([0]*i+[s]+[0]*(j-i-1)+[s]+[0]*(d-j-1))
        weights.append(C)

        z.append([0]*i+[s]+[0]*(j-i-1)+[-s]+[0]*(d-j-1))
        weights.append(C)

        z.append([0]*i+[-s]+[0]*(j-i-1)+[s]+[0]*(d-j-1))
        weights.append(C)

        z.append([0]*i+[-s]+[0]*(j-i-1)+[-s]+[0]*(d-j-1))
        weights.append(C)

    z = [[np.sqrt(2)*y for y in x] for x in z]
    return z, weights
  else:
    print("Invalid dimension")

def gaussian_cubature_7(d = 3):
  if d == 2:
    r = np.sqrt(3)
    s = np.sqrt((9-3*np.sqrt(5))/8)
    t = np.sqrt((9+3*np.sqrt(5))/8)
    A = 1/36
    B = (5+2*np.sqrt(5))/45
    C = (5-2*np.sqrt(5))/45

    z = [] # initiate variables and append values
    weights = []

    z.append([r,0])
    weights.append(A)
    z.append([-r,0])
    weights.append(A)
    z.append([0,r])
    weights.append(A)
    z.append([0,-r])
    weights.append(A)

    z.append([s,s])
    weights.append(B)
    z.append([s,-s])
    weights.append(B)
    z.append([-s,s])
    weights.append(B)
    z.append([-s,-s])
    weights.append(B)

    z.append([t,t])
    weights.append(C)
    z.append([t,-t])
    weights.append(C)
    z.append([-t,t])
    weights.append(C)
    z.append([-t,-t])
    weights.append(C)

    z = [[np.sqrt(2)*y for y in x] for x in z]
    return z, weights
  
  elif d == 3:
    u = 0.524647623275290
    v = 1.65068012388578
    B0 = -16.6705761599566
    B1 = 10.0296981655678
    B2 = 0.161699246687754
    B3 = -6.04719151221535
    B4 = 0.0234381399489666
    B5 = 4.17194501880647
  elif d == 4:
    u = 0.339981043584856
    v = 0.861136311594053
    B0 = -1387.37239302779
    B1 = 535.678956451906
    B2 = -0.969864622038531
    B3 = -184.343295988627
    B4 = 0.625450563856360
    B5 = 47.9664983932297

  z = [] # initiate variables and append values
  weights = []

  z.append([0]*d)
  weights.append(B0)
    
  for i in range(d):
    z.append([0]*i+[u]+[0]*(d-i-1))
    weights.append(B1)

    z.append([0]*i+[-u]+[0]*(d-i-1))
    weights.append(B1)

    z.append([0]*i+[v]+[0]*(d-i-1))
    weights.append(B2)

    z.append([0]*i+[-v]+[0]*(d-i-1))
    weights.append(B2)

    for j in range(i+1, d):
      z.append([0]*i+[u]+[0]*(j-i-1)+[u]+[0]*(d-j-1))
      weights.append(B3)

      z.append([0]*i+[u]+[0]*(j-i-1)+[-u]+[0]*(d-j-1))
      weights.append(B3)

      z.append([0]*i+[-u]+[0]*(j-i-1)+[u]+[0]*(d-j-1))
      weights.append(B3)

      z.append([0]*i+[-u]+[0]*(j-i-1)+[-u]+[0]*(d-j-1))
      weights.append(B3)

      z.append([0]*i+[v]+[0]*(j-i-1)+[v]+[0]*(d-j-1))
      weights.append(B4)

      z.append([0]*i+[v]+[0]*(j-i-1)+[-v]+[0]*(d-j-1))
      weights.append(B4)

      z.append([0]*i+[-v]+[0]*(j-i-1)+[v]+[0]*(d-j-1))
      weights.append(B4)

      z.append([0]*i+[-v]+[0]*(j-i-1)+[-v]+[0]*(d-j-1))
      weights.append(B4)

      for k in range(j+1, d):
        z.append([0]*i+[u]+[0]*(j-i-1)+[u]+[0]*(k-j-1)+[u]+[0]*(d-k-1))
        weights.append(B5)

        z.append([0]*i+[-u]+[0]*(j-i-1)+[u]+[0]*(k-j-1)+[u]+[0]*(d-k-1))
        weights.append(B5)

        z.append([0]*i+[u]+[0]*(j-i-1)+[-u]+[0]*(k-j-1)+[u]+[0]*(d-k-1))
        weights.append(B5)

        z.append([0]*i+[u]+[0]*(j-i-1)+[u]+[0]*(k-j-1)+[-u]+[0]*(d-k-1))
        weights.append(B5)

        z.append([0]*i+[-u]+[0]*(j-i-1)+[-u]+[0]*(k-j-1)+[u]+[0]*(d-k-1))
        weights.append(B5)

        z.append([0]*i+[-u]+[0]*(j-i-1)+[u]+[0]*(k-j-1)+[-u]+[0]*(d-k-1))
        weights.append(B5)

        z.append([0]*i+[u]+[0]*(j-i-1)+[-u]+[0]*(k-j-1)+[-u]+[0]*(d-k-1))
        weights.append(B5)

        z.append([0]*i+[-u]+[0]*(j-i-1)+[-u]+[0]*(k-j-1)+[-u]+[0]*(d-k-1))
        weights.append(B5)

  z = [[np.sqrt(2)*y for y in x] for x in z]
  weights = [x*((1/np.pi)**(d/2)) for x in weights]
  return z, weights


def lie_sum(A):
  total = Elt([{}])
  for a in A:
    total += a

  return total

def wiener_cubature(d):
  x = 0 # constant used in Lyons-Victoir construction (assumes any value so taken to be zero for simplicity)

  z, gauss_lam = gaussian_cubature(d = d) # degree 5 cubature on Gaussian

  e = [word2Elt(Word([i])) for i in range(d+1)] # tensor space basis

  lie_poly = [] # initiate lie_poly
  lam = [] # initiate lam
  
  for k, zk in enumerate(z):
    sum1 = lie_sum([(1/12)*(zk[i-1])**2*lieProduct(lieProduct(e[0],e[i]), e[i]) for i in range(1,d+1)])
    sum2 = lie_sum([zk[i-1]*e[i] for i in range(1,d+1)])
    sum3 = lie_sum([lie_sum([(1/2)*zk[i-1]*zk[j-1]*lieProduct(e[i],e[j]) for j in range(i+1,d+1)]) for i in range(1,d+1)])
    sum4 = lie_sum([lie_sum([(1/6)*zk[i-1]*zk[j-1]**2*lieProduct(lieProduct(e[i],e[j]),e[j]) for j in range(i+1,d+1)]) for i in range(1,d+1)])
    sum5 = lie_sum([lie_sum([(1/6)*zk[j-1]*zk[i-1]**2*lieProduct(lieProduct(e[j],e[i]),e[i]) for j in range(i+1,d+1)]) for i in range(1,d+1)])
    
    lie_poly.append(e[0]+sum1+sum2+sum3+x*sum4+(1-x)*sum5) # add eta = +1 lie polynomials
    lam.append((1/2)*gauss_lam[k])

    lie_poly.append(e[0]+sum1+sum2-sum3+(1-x)*sum4+x*sum5) # add eta = -1 lie polynomials
    lam.append((1/2)*gauss_lam[k])

  return lie_poly, lam

def truncateToLevel(e, level):
  elt_data = e.data[:2*level+1]
  for i in range(len(elt_data)):
    i_data = elt_data[i]
    for key in i_data.keys():
      total_d = i + key.letters.count(0)
      if total_d > level:
        e -= Elt([{}]*i+[{key : i_data[key]}])
        #i_data[key] = 0
  return e

def verify(m, d, lie_poly, lam):
  rhs = Elt([{}]) # evaluate the rhs of the sum given in Lyons-Victoir
  for i in range(len(lam)):
    rhs += lam[i]*exp(lie_poly[i], maxLevel = m)

  rhs = truncateToLevel(rhs, m)

def verify_lhs(m, d):
  e = [word2Elt(Word([i])) for i in range(d+1)] # evaulate the lhs of the sum given in Lyons-Victoir
  lhs = e[0]
  for i in range(1,d+1):
    lhs += (0.5*concatenationProduct(e[i], e[i]))

  lhs = exp(lhs, maxLevel = m)
  lhs = truncateToLevel(lhs, m)

  return lhs

def verify_rhs(m, d, lie_poly, lam):
  rhs = lie_sum([lam[i]*exp(lie_poly[i], maxLevel = m) for i in range(len(lam))]) # evaluate the rhs of the sum given in Lyons-Victoir
  rhs = truncateToLevel(rhs, m)

  return rhs

def dim_1_wiener_cubature(m):
  if m == 3:
    z = [[-1],[1]]
    gauss_lam = [1/2,1/2]
    e = [word2Elt(Word([i])) for i in range(m+1)]
    lie_poly = []
    lam = []
    for k, zk in enumerate(z):
      lie_poly.append(e[0]+zk[0]*e[1])
      lam.append(gauss_lam[k])
    return lie_poly, lam
  elif m == 5:
    e = [word2Elt(Word([i])) for i in range(m+1)]
    z = [[np.sqrt(3)], [0], [-np.sqrt(3)]]
    gauss_lam = [1/6, 2/3, 1/6]
    
    lie_poly = []
    lam = []
    
    for k, zk in enumerate(z):
      sum1 = lie_sum([zk[i-1]*e[i] for i in range(1,2)])
      sum4 = -zk[0]*lieProduct(e[0],e[1])
      sum5 = lie_sum([lieProduct(lieProduct(e[0], e[i]), e[i]) for i in range(1,2)])
    
      for gamma_1 in [-1,1]:
        lie_poly.append(e[0]
                        +sum1
                        +gamma_1*np.sqrt(1/12)*sum4
                        +(1/12)*sum5
                        )
        lam.append((1/2)*gauss_lam[k])
    return lie_poly, lam
  elif m == 7:
    e = [word2Elt(Word([i])) for i in range(m+1)]
    #z = [[-1.6506801], [-0.5246476],  [0.5246476],  [1.6506801]]
    #gauss_lam = [0.08131284, 0.80491409, 0.80491409, 0.08131284]
    z = [[np.sqrt((3-np.sqrt(6))/2)],[-np.sqrt((3-np.sqrt(6))/2)],[np.sqrt((3+np.sqrt(6))/2)],[-np.sqrt((3+np.sqrt(6))/2)]]
    gauss_lam = [np.sqrt(np.pi)/(4*(3-np.sqrt(6))),np.sqrt(np.pi)/(4*(3-np.sqrt(6))),np.sqrt(np.pi)/(4*(3+np.sqrt(6))),np.sqrt(np.pi)/(4*(3+np.sqrt(6)))]

    z = [[np.sqrt(2)*y for y in x] for x in z]
    gauss_lam = [x*((1/np.pi)**(1/2)) for x in gauss_lam]
    lie_poly = []
    lam = []
    
    for k, zk in enumerate(z):
      sum1 = lie_sum([zk[i-1]*e[i] for i in range(1,2)])
      sum4 = -zk[0]*lieProduct(e[0],e[1])
      sum5 = lie_sum([lieProduct(lieProduct(e[0], e[i]), e[i]) for i in range(1,2)])
      sum10 = lie_sum([lieProduct(lieProduct(lieProduct(lieProduct(e[0],e[i]),e[i]),e[i]),e[i]) for i in range(1,2)])
    
      for gamma_1 in [-1,1]:
        lie_poly.append(e[0]
                        +sum1
                        +gamma_1*np.sqrt(1/12)*sum4
                        +(1/12)*sum5
                        +(1/360)*sum10
                        )
        lam.append((1/2)*gauss_lam[k])
    return lie_poly, lam


def dim_2_wiener_cubature(m):
  if m == 3:
    z, gauss_lam = [[np.sqrt(2),0],[0,np.sqrt(2)],[-np.sqrt(2),0],[0,-np.sqrt(2)]], [(1/4)]*4 # degree 3 cubature on Gaussian

    e = [word2Elt(Word([i])) for i in range(3)] # tensor space basis

    lie_poly = [] # initiate lie_poly
    lam = [] # initiate lam
    
    for k, zk in enumerate(z):
      lie_poly.append(e[0]+zk[0]*e[1]+zk[1]*e[2])
      lam.append(gauss_lam[k])

    return lie_poly, lam
  
  elif m == 5:
    return wiener_cubature(d = 2)
  
  elif m == 7:
    d = 2
    z, gauss_lam = gaussian_cubature_7(d = d) # degree 7 cubature on Gaussian
    
    e = [word2Elt(Word([i])) for i in range(d+1)] # tensor space basis
    lie_poly = [] # initiate lie_poly
    lam = [] # initiate lam

    for k, zk in enumerate(z):
      sum1 = lie_sum([zk[i-1]*e[i] for i in range(1,d+1)])
      sum2 = lieProduct(e[1],e[2])
      sum3 = lie_sum([lie_sum([zk[i-1]*lieProduct(lieProduct(e[i],e[j]),e[j])+zk[j-1]*lieProduct(lieProduct(e[j],e[i]),e[i]) for j in range(i+1,d+1)]) for i in range(1,d+1)]) 
      sum4 = -zk[0]*lieProduct(e[0],e[1])+zk[1]*lieProduct(e[0],e[2])
      sum5 = lie_sum([lieProduct(lieProduct(e[0], e[i]), e[i]) for i in range(1, d+1)])
      sum6 = zk[0]*zk[1]*(lieProduct(lieProduct(lieProduct(e[1],e[2]),e[1]),e[1])+lieProduct(lieProduct(lieProduct(e[1],e[2]),e[2]),e[2]))
      sum7 = zk[0]*lieProduct(lieProduct(lieProduct(e[1],e[2]),e[1]),lieProduct(e[1],e[2])) + zk[1]*lieProduct(lieProduct(lieProduct(e[2],e[1]),e[2]),lieProduct(e[2],e[1])) 
      sum8 = zk[0]*lieProduct(lieProduct(lieProduct(lieProduct(e[1],e[2]),e[1]),e[1]),e[2])+zk[1]*lieProduct(lieProduct(lieProduct(lieProduct(e[2],e[1]),e[2]),e[2]),e[1])
      sum9 = zk[1]*lieProduct(lieProduct(lieProduct(lieProduct(e[1],e[2]),e[1]),e[1]),e[1])+zk[0]*lieProduct(lieProduct(lieProduct(lieProduct(e[2],e[1]),e[2]),e[2]),e[2])
      sum10 = lie_sum([lieProduct(lieProduct(lieProduct(lieProduct(e[0],e[i]),e[i]),e[i]),e[i]) for i in range(1,d+1)])
      sum11 = lieProduct(lieProduct(lieProduct(lieProduct(e[0],e[2]),e[1]),e[1]),e[2])+lieProduct(lieProduct(lieProduct(lieProduct(e[0],e[1]),e[1]),e[2]),e[2])
      
      for gamma_1 in [-1,1]:
        for gamma_2 in [-1,1]:
          lie_poly.append(e[0]
                          +sum1
                          +gamma_1*(np.sqrt(1/12)*zk[0]*zk[1]+gamma_2*np.sqrt(1/6))*sum2
                          +(1/12)*(1+2*gamma_2)*sum3
                          +gamma_1*np.sqrt(1/12)*sum4
                          +(1/12)*sum5
                          +(1/(12*np.sqrt(3)))*gamma_1*sum6
                          -(1/90)*sum7
                          +(1/360)*sum8
                          -(1/360)*sum9
                          +(1/360)*sum10
                          +(1/360)*sum11
                          -(1/360)*lieProduct(lieProduct(lieProduct(e[1],e[2]),e[2]),lieProduct(e[0],e[1]))
                          +(1/180)*lieProduct(lieProduct(lieProduct(e[1],e[2]),e[1]),lieProduct(e[0],e[2]))
                          -(1/60)*lieProduct(lieProduct(lieProduct(e[0],e[2]),e[1]),lieProduct(e[1],e[2]))
                          +(1/90)*lieProduct(lieProduct(lieProduct(e[0],e[1]),e[2]),lieProduct(e[1],e[2]))
                          )
          lam.append((1/4)*gauss_lam[k])
    return lie_poly, lam
  

def verify(m, d, lie_poly, lam):
  rhs = verify_rhs(m, d, lie_poly, lam)
  lhs = verify_lhs(m, d)

  return distance(rhs,lhs) < 1e-14

print(verify_lhs(7,3))
