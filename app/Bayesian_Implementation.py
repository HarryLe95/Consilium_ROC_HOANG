"""
Nomenclature:
S1 - state_dying  - the dying state 
S0 - state_normal - the normal state 
delta_V - voltage drop - changes between the minimum Voltage between different days 
S - state r.v, takes discrete values from {S0, S1}
dV - voltage drop r.v take continuous values 

Goal - find:
P(S = S1|dV = delta_V)
The idea is to provide a pseduo label for the dying state and their probability of being in that state 

P(S=S1|dV) = P(dV|S=S1) * P(S=S1)/P(dV) = P(dV|S=S1)P(S=S1)/[P(dV|S=S1)P(S=S1) + P(dV|S=S0)P(S=S0)]
"""

